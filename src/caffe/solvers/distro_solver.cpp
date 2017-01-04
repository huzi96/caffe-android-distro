#include <vector>

#include "caffe/sgd_solvers.hpp"

namespace caffe {

template <typename Dtype>
int DistroSolver<Dtype>::Step_stage_0(int &average_loss, const int start_iter) {
	// zero-init the params
	this->net_->ClearParamDiffs();
	if (this->param_.test_interval() && this->iter_ % this->param_.test_interval() == 0
	    && (this->iter_ > 0 || this->param_.test_initialization())
	    && Caffe::root_solver()) {
	  this->TestAll();
	  if (this->requested_early_exit_) {
	    // Break out of the while loop because stop was requested while testing.
	    return -1;
	  }
	}

	for (int i = 0; i < Solver<Dtype>::callbacks_.size(); ++i) {
	  Solver<Dtype>::callbacks_[i]->on_start();
	}
	const bool display = this->param_.display() && this->iter_ % this->param_.display() == 0;
	this->net_->set_debug_info(display && this->param_.debug_info());
	// accumulate the loss and gradient
	Dtype loss = 0;

	for (int i = 0; i < this->param_.iter_size(); ++i) {
	  loss += this->net_->ForwardBackward();
	}
	loss /= this->param_.iter_size();
	// average the loss across iterations for smoothed reporting
	this->UpdateSmoothedLoss(loss, start_iter, average_loss);
	if (display) {
	  LOG_IF(INFO, Caffe::root_solver()) << "Iteration " << this->iter_
	      << ", loss = " << this->smoothed_loss_;
	  const vector<Blob<Dtype>*>& result = this->net_->output_blobs();
	  int score_index = 0;
	  for (int j = 0; j < result.size(); ++j) {
	    const Dtype* result_vec = result[j]->cpu_data();
	    const string& output_name =
	        this->net_->blob_names()[this->net_->output_blob_indices()[j]];
	    const Dtype loss_weight =
	        this->net_->blob_loss_weights()[this->net_->output_blob_indices()[j]];
	    for (int k = 0; k < result[j]->count(); ++k) {
	      ostringstream loss_msg_stream;
	      if (loss_weight) {
	        loss_msg_stream << " (* " << loss_weight
	                        << " = " << loss_weight * result_vec[k] << " loss)";
	      }
	      LOG_IF(INFO, Caffe::root_solver()) << "    Train net output #"
	          << score_index++ << ": " << output_name << " = "
	          << result_vec[k] << loss_msg_stream.str();
	    }
	  }
	}
	for (int i = 0; i < Solver<Dtype>::callbacks_.size(); ++i) {
	  Solver<Dtype>::callbacks_[i]->on_gradients_ready();
	}
	return 0;
}

template <typename Dtype>
int DistroSolver<Dtype>::Step_stage_1() {
    this->ApplyUpdate();

    // Increment the internal iter_ counter -- its value should always indicate
    // the number of times the weights have been updated.
    this->iter_++;

    SolverAction::Enum request = this->GetRequestedAction();

    // Save a snapshot if needed.
    if ((this->param_.snapshot()
         && this->iter_ % this->param_.snapshot() == 0
         && Caffe::root_solver()) ||
         (request == SolverAction::SNAPSHOT)) {
      this->Snapshot();
    }
    if (SolverAction::STOP == request) {
      this->requested_early_exit_ = true;
      // Break out of training loop.
      return -1;
    }
    return 0;
}

template <typename Dtype>
void DistroSolver<Dtype>::Step(int iters) {
	const int start_iter = this->iter_;
	const int stop_iter = this->iter_ + iters;
	int average_loss = this->param_.average_loss();
	this->losses_.clear();
	this->smoothed_loss_ = 0;

	while (this->iter_ < stop_iter) {
		if (Step_stage_0(average_loss, start_iter) == -1) break;
		if (Step_stage_1() == -1) break;
	}
}

template <typename Dtype>
int DistroSolver<Dtype>::Half_iter(ostream *outstream) {
	const int start_iter = this->iter_;
	int average_loss = this->param_.average_loss();
	this->losses_.clear();
	this->smoothed_loss_ = 0;
	Step_stage_0(average_loss, start_iter);
	NetParameter export_param;
	this->net_->ToProto(&export_param, true);
	export_param.SerializeToOstream(outstream);
	return 0;
}

template <typename Dtype>
int DistroSolver<Dtype>::Cont_iter(istream *instream) {
	ZeroCopyInputStream *inputstream = new IstreamInputStream(instream);
	CodedInputStream* coded_input = new CodedInputStream(inputstream);
	// coded_input->SetTotalBytesLimit(kProtoReadBytesLimit, 536870912);
	NetParameter *proto = new NetParameter();
	bool success = proto->ParseFromCodedStream(coded_input);

	if(success) {
		this->net_->CopyTrainedLayersFrom(*proto);
	}
	delete proto;
	delete coded_input;
	delete inputstream;
	Step_stage_1();
	return 0;
}

INSTANTIATE_CLASS(DistroSolver);
REGISTER_SOLVER_CLASS(Distro);

}  // namespace caffe
