#include <vector>

#include "caffe/sgd_solvers.hpp"

namespace caffe {

/* 
 * Two separate stage for one iteration
 * Stage 0 is for one (or more) ForwardBackward operation
 */
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

/*Stage 1 is for applying update according to blobs in the net*/
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

/*For testing, one stage 0 and one stage 1*/
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

/*Do stage 0 and return the net parameters*/
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

/*Do stage 1 with the parameter*/
template <typename Dtype>
int DistroSolver<Dtype>::Cont_iter(istream *instream) {
	LOG(INFO) << "Cont_iter";
	ZeroCopyInputStream *inputstream = new IstreamInputStream(instream);
	CodedInputStream* coded_input = new CodedInputStream(inputstream);
	// coded_input->SetTotalBytesLimit(kProtoReadBytesLimit, 536870912);
	NetParameter *proto = new NetParameter();
	bool success = proto->ParseFromCodedStream(coded_input);
	const int timeout = 32;
	for (int i = 0; i < timeout; ++i)
	{
		if(success) {
			// LOG(INFO) << "Success in decoding";
			this->net_->CopyTrainedLayersFrom(*proto);
			if (i != 0) {
				LOG(INFO) << "Try again worked";
			}
			break;
		}
		else {
			LOG(INFO) << "Fail decoding";
			sleep(1);
		}
	}
	if (!success)
	{
		CHECK(success == true) << "Tried 64 times and fail";
	}
	delete proto;
	delete coded_input;
	delete inputstream;
	Step_stage_1();
	return 0;
}


/*Accumulate diff in the pair_net.*/
template <typename Dtype>
int DistroSolver<Dtype>::Accumulate_diff(istream *instream) {
	LOG(INFO) << "Accumulate";
	ZeroCopyInputStream *inputstream = new IstreamInputStream(instream);
	CodedInputStream* coded_input = new CodedInputStream(inputstream);
	// coded_input->SetTotalBytesLimit(kProtoReadBytesLimit, 536870912);
	NetParameter *proto = new NetParameter();
	//////////////////////////////////////////////////////////////////////////
	// merged_cnt = 0;
	//////////////////////////////////////////////////////////////////////////
	const int timeout = 32;
	for (int i = 0; i < timeout; ++i)
	{
		bool success = proto->ParseFromCodedStream(coded_input);
		if(success) {
			// LOG(INFO) << "Success in parsing";
			if (i != 0)
			{
				LOG(INFO) << "Succeed at last";
			}
			if(merged_cnt == 0) {
				// this->pair_net = new Net<Dtype>(*proto, this->net_.get());
				// this->pair_net = new Net<Dtype>(*proto);
				this->pair_net = this->net_;
				this->pair_net->CopyTrainedLayersFrom(*proto);
				// LOG(INFO) << "Build pair net successfully";
				merged_cnt = 1;
				delete proto;
				delete coded_input;
				delete inputstream;
			}
			else
			{
				merged_cnt++;
				// CHECK_EQ(pair_net.layers().size(), this->net_->layers().size());
				int num_source_layers = proto->layer_size();
				for (int i = 0; i < num_source_layers; ++i) {
					const LayerParameter& source_layer = proto->layer(i);
					const string& source_layer_name = source_layer.name();
					int target_layer_id = 0;
					while (target_layer_id != this->pair_net->layer_names().size() &&
					    this->pair_net->layer_names()[target_layer_id] != source_layer_name) {
					  	++target_layer_id;
					}
					if (target_layer_id == this->pair_net->layer_names().size()) {
						LOG(INFO) << "Ignoring source layer " << source_layer_name;
						continue;
					}
					DLOG(INFO) << "Copying source layer " << source_layer_name;
					vector<shared_ptr<Blob<Dtype> > >& target_blobs =
					    this->pair_net->layers()[target_layer_id]->blobs();
					CHECK_EQ(target_blobs.size(), source_layer.blobs_size())
					    << "Incompatible number of blobs for layer " << source_layer_name;
					for (int j = 0; j < target_blobs.size(); ++j) {
					 	if (!target_blobs[j]->ShapeEquals(source_layer.blobs(j))) {
						    Blob<Dtype> source_blob;
						    const bool kReshape = true;
						    source_blob.FromProto(source_layer.blobs(j), kReshape);
						    LOG(FATAL) << "Cannot copy param " << j << " weights from layer '"
						        << source_layer_name << "'; shape mismatch.  Source param shape is "
						        << source_blob.shape_string() << "; target param shape is "
						        << target_blobs[j]->shape_string() << ". "
						        << "To learn this layer's parameters from scratch rather than "
						        << "copying from a saved net, rename the layer.";
				  		}
						// target_blobs[j]->FromProto(source_layer.blobs(j), kReshape);
					 	Blob<Dtype> source_blob;
					    const bool kReshape = true;
					    source_blob.FromProto(source_layer.blobs(j), kReshape);
						caffe_cpu_axpby<Dtype>(source_blob.count(), 1.,
			              source_blob.cpu_diff(), 1.,
			              target_blobs[j]->mutable_cpu_data());
					}
				}
				delete proto;
				delete coded_input;
				delete inputstream;
			}
			break;
		}
		else {
			LOG(INFO) << "Error in parsing";
		}
	}
	return 0;
}

/*Get the pair_net*/
template <typename Dtype>
int DistroSolver<Dtype>::GetAccumulatedNet(ostream* outstream) {
	LOG(INFO) << "GetAccumulatedNet";
	NetParameter export_param;
	this->pair_net->ToProto(&export_param, true);
    // LOG(INFO) << "SerializeToOstream";
	export_param.SerializeToOstream(outstream);
	merged_cnt = 0;
    // LOG(INFO) << "Delete pair_net";
	// delete this->pair_net;
	return 0;
}

/*Set the parameters according to an incoming net*/
template <typename Dtype>
int DistroSolver<Dtype>::SetNet(istream* instream) {
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
	return 0;
}

/*Oveeriding normalize to do the normalization according to the preset normlize scale*/
template <typename Dtype>
void DistroSolver<Dtype>::Normalize(int param_id) {
	// Scale gradient to counterbalance accumulation.
	// LOG(INFO)<<"Distro Normalization called";
	const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
	const Dtype accum_normalization = Dtype(1.) / this->normalize_scale;
	switch (Caffe::mode()) {
		case Caffe::CPU: {
			caffe_scal(net_params[param_id]->count(), accum_normalization,
			    net_params[param_id]->mutable_cpu_diff());
		break;
		}
		case Caffe::GPU: {
	#ifndef CPU_ONLY
			caffe_gpu_scal(net_params[param_id]->count(), accum_normalization,
		    	net_params[param_id]->mutable_gpu_diff());
	#else
			NO_GPU;
	#endif
			break;
		}
		default:
		LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
	}
}

template <typename Dtype>
void DistroSolver<Dtype>::SetNormalizeScale(int scale)
{
	this->normalize_scale = scale;
}



INSTANTIATE_CLASS(DistroSolver);
REGISTER_SOLVER_CLASS(Distro);

}  // namespace caffe
