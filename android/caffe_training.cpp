#include "caffe_training.hpp"

DEFINE_string(gpu, "",
    "Optional; run in GPU mode on given device IDs separated by ','."
    "Use '-gpu all' to run on all available GPUs. The effective training "
    "batch size is multiplied by the number of devices.");
DEFINE_string(solver, "",
    "The solver definition protocol buffer text file.");
DEFINE_string(model, "",
    "The model definition protocol buffer text file.");
DEFINE_string(phase, "",
    "Optional; network phase (TRAIN or TEST). Only used for 'time'.");
DEFINE_int32(level, 0,
    "Optional; network level.");
DEFINE_string(stage, "",
    "Optional; network stages (not to be confused with phase), "
    "separated by ','.");
DEFINE_string(snapshot, "",
    "Optional; the snapshot solver state to resume training.");
DEFINE_string(weights, "",
    "Optional; the pretrained weights to initialize finetuning, "
    "separated by ','. Cannot be set simultaneously with snapshot.");
DEFINE_int32(iterations, 50,
    "The number of iterations to run.");
DEFINE_string(sigint_effect, "stop",
             "Optional; action to take when a SIGINT signal is received: "
              "snapshot, stop or none.");
DEFINE_string(sighup_effect, "snapshot",
             "Optional; action to take when a SIGHUP signal is received: "
             "snapshot, stop or none.");

namespace caffe {

CaffeTrain *CaffeTrain::caffe_train_ = 0;
string CaffeTrain::solver_path_ = "";

CaffeTrain *CaffeTrain::Get() {
  CHECK(caffe_train_);
  return caffe_train_;
}

CaffeTrain *CaffeTrain::Get(const string &solver_path) {
  if (!caffe_train_ || solver_path != solver_path_) {
    caffe_train_ = new CaffeTrain(solver_path);
    solver_path_ = solver_path;
  }
  return caffe_train_;
}

caffe::SolverAction::Enum GetRequestedAction(
    const std::string& flag_value) {
  if (flag_value == "stop") {
    return caffe::SolverAction::STOP;
  }
  if (flag_value == "snapshot") {
    return caffe::SolverAction::SNAPSHOT;
  }
  if (flag_value == "none") {
    return caffe::SolverAction::NONE;
  }
  LOG(FATAL) << "Invalid signal effect \""<< flag_value << "\" was specified";
}


CaffeTrain::CaffeTrain(const string &solver_path) 
{
  CHECK_GT(solver_path.size(), 0) << "Need solver descriptor file.";
  Caffe::set_mode(Caffe::CPU);
  caffe::ReadSolverParamsFromTextFileOrDie(solver_path, &solver_param);
  solver_param.mutable_train_state()->set_level(0);

  caffe::SignalHandler signal_handler(
          GetRequestedAction(FLAGS_sigint_effect),
          GetRequestedAction(FLAGS_sighup_effect));
  solver = shared_ptr<caffe::Solver<float> >(caffe::SolverRegistry<float>::CreateSolver(solver_param));

  solver->SetActionFunction(signal_handler.GetActionFunction());
}

CaffeTrain::~CaffeTrain() {}

int CaffeTrain::solve() {return 0;}

int CaffeTrain::solve_test()
{
  LOG(INFO) << ("simple train");
  OneIter();
  return 0;
}

char *CaffeTrain::ForwardBackward() {
  boost::asio::streambuf buf;
  std::ostream outstream(&buf);
  solver->Half_iter(&outstream);
  char* outbytes = new char[buf.size()];
  memcpy(outbytes, boost::asio::buffer_cast<const void*>(buf.data()), buf.size());
  net_size = buf.size();
  // LOG(INFO) << net_size;
  return outbytes;
}

int CaffeTrain::UpdateWith(std::vector<char> raw_vector) {
  int length = raw_vector.size();
  char *raw_stream = new char[length];
  std::copy(raw_vector.begin(), raw_vector.end(), raw_stream);
  membuf buf(raw_stream, raw_stream + length);
  std::istream instream(&buf);
  solver->Cont_iter(&instream);
  delete raw_stream;
  return 0;
}

int CaffeTrain::Accumulate(std::vector<char> raw_vector) {
  int length = raw_vector.size();
  char *raw_stream = new char[length];
  std::copy(raw_vector.begin(), raw_vector.end(), raw_stream);
  membuf buf(raw_stream, raw_stream + length);
  std::istream instream(&buf);
  solver->Accumulate_diff(&instream);
  delete raw_stream;
  return 0;
}

char *CaffeTrain::GetNewNet() {
  boost::asio::streambuf buf;
  std::ostream outstream(&buf);

  // LOG(INFO) << "GetAccumulatedNet";
  solver->GetAccumulatedNet(&outstream);

  // LOG(INFO) << "Generate block buffer";
  char* outbytes = new char[buf.size()];
  memcpy(outbytes, boost::asio::buffer_cast<const void*>(buf.data()), buf.size());
  net_size = buf.size();
  // LOG(INFO) << net_size;
  return outbytes;
}

void CaffeTrain::SetNormalizeScale(int scale)
{
  this->solver->SetNormalizeScale(scale);
}

float CaffeTrain::getAcc()
{
  return this->solver->stored_accuracy;
}

void CaffeTrain::OneIter() {
  LOG(INFO) << "Solving ";
  // LOG(INFO) << "Learning Rate Policy: " << param_.lr_policy();
  solver->SetNormalizeScale(1);
  for(int i=0; i<1000; i++)
  {
    // LOG(INFO) << "ForwardBackward";
    char *p = ForwardBackward();
    // LOG(INFO) << "Accumulate";
    // Accumulate(p, net_size);
    // // LOG(INFO) << "GetNewNet";
    // p = GetNewNet();
    // LOG(INFO) << "UpdateWith";
    // UpdateWith(p, net_size);
  }
}

}
