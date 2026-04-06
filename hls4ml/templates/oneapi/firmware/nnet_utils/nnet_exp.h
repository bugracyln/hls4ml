#ifndef NNET_EXP_H_
#define NNET_EXP_H_

#include "nnet_common.h"
#include "nnet_dense.h"
#include "nnet_helpers.h"
#include "nnet_mult.h"
#include "nnet_transpose.h"
#include <type_traits>
#include <limits>
#include <sycl/ext/oneapi/device_global/device_global.hpp>

/*
This implements kernel to calculate e^x directly using a mixture of reduction and Taylor approximation 
to calculate e^x. The main use will be in the softmax layer
*/






#endif