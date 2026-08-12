#ifndef NNET_DENSE_STREAM_H_
#define NNET_DENSE_STREAM_H_

#include "nnet_common.h"
#include "nnet_dense.h"
#include "nnet_types.h"

namespace nnet {

// Note:  DataPack logic removed, at least in the initial version
template <class data_pipe, class res_pipe, typename CONFIG_T> void dense_resource_stream() {

#ifdef AUTOREG
    using data_pipe_T = typename ExtractPipeType<data_pipe>::value_type;
    using res_pipe_T = typename ExtractPipeType<res_pipe>::value_type;
    using data_arr_T = typename data_pipe_T::data_type;
    using res_arr_T = typename res_pipe_T::data_type;
#else
    using data_arr_T = typename ExtractPipeType<data_pipe>::value_type;
    using res_arr_T = typename ExtractPipeType<res_pipe>::value_type;
#endif

    [[intel::fpga_register]] res_arr_T res;
    
#ifdef AUTOREG
    [[intel::fpga_register]] res_pipe_T out_pipe;
    while (true){
    [[intel::fpga_register]] data_pipe_T data_pack = data_pipe::read();
    out_pipe.feedback = data_pack.feedback;
    if (data_pack.exit_task) {
        out_pipe.exit_task = true;
        res_pipe::write(out_pipe);
        break;
    }
    [[intel::fpga_register]] data_arr_T data = data_pack.data;
#else
    [[intel::fpga_register]] data_arr_T data = data_pipe::read();
#endif

        dense_resource<data_arr_T, res_arr_T, CONFIG_T>(data, res);

#ifdef AUTOREG
        out_pipe.exit_task = data_pack.exit_task;
        out_pipe.data = res;
        res_pipe::write(out_pipe);
    }
#else
    res_pipe::write(res);
#endif
}

} // namespace nnet

#endif
