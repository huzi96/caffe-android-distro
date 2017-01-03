#include <vector>

#include "caffe/sgd_solvers.hpp"

namespace caffe {

INSTANTIATE_CLASS(DistroSolver);
REGISTER_SOLVER_CLASS(Distro);

}  // namespace caffe
