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
  solver->Solve();
  return 0;
}


}
