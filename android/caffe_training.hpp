#ifndef CAFFE_MOBILETRAINING_HPP_
#define CAFFE_MOBILETRAINING_HPP_
#include <algorithm>
#include <string>
#include <vector>
#include <cstring>
#include <map>

#include "boost/algorithm/string.hpp"

#include "caffe/caffe.hpp"
#include "caffe/layers/memory_data_layer.hpp"


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <gflags/gflags.h>
#include <glog/logging.h>

#include "caffe/util/signal_handler.h"

#include "boost/asio.hpp"

using std::clock;
using std::clock_t;
using std::string;
using std::vector;

using caffe::Blob;
using caffe::Caffe;
using caffe::Datum;
using caffe::Net;
using caffe::MemoryDataLayer;

using caffe::Layer;
using caffe::Solver;
using caffe::shared_ptr;
using caffe::string;
using caffe::Timer;
using caffe::vector;


using std::string;
using std::vector;

namespace caffe {

template <typename Dtype>
class DistriSolver: public Solver<Dtype>
{

};

class CaffeTrain {
public:
  ~CaffeTrain();

  static CaffeTrain *Get();
  static CaffeTrain *Get(const string &solver_path);

  int solve_test();
  int solve();
  void OneIter();

private:
  static CaffeTrain *caffe_train_;
  static string solver_path_;

  CaffeTrain(const string &solver_path);

  /*My new solver object*/
  SolverParameter solver_param;

  shared_ptr<caffe::Solver<float> > solver;
};

} // namespace caffe

#endif
