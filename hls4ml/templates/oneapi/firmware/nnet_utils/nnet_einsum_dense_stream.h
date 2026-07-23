#ifndef NNET_EINSUM_DENSE_H_
#define NNET_EINSUM_DENSE_H_

#include "nnet_common.h"
#include "nnet_dense.h"
#include "nnet_helpers.h"
#include "nnet_mult.h"
#include "nnet_transpose.h"

namespace nnet {

struct einsum_dense_config {
    // Internal data type definitions
    typedef void tpose_inp_conf;
    typedef void tpose_out_conf;
    typedef void dense_conf;

    // Layer Sizes
    static const unsigned n_free_data = 1;
    static const unsigned n_free_kernel = 1;
    static const unsigned n_contract = 1;
    static const unsigned n_inplace = 1;

    // Resource reuse info
    static const unsigned io_type = io_parallel;
    static const unsigned reuse_factor = 1;
    static const unsigned parallelization_factor = 1000;

    // Product function to use
    template <class x_T, class y_T> using product = nnet::product::mult<x_T, y_T>;
};

// Read tokens from input stream into a buffer
template <class Dense_in_T, class data_pipe, typename CONFIG_T> 
void read_token(Dense_in_T &token_buffer
    #ifdef AUTOREG
    , bool &exit_task
    #endif
    ) {

    using data_buff_T = typename ExtractPipeType<data_pipe>::value_type;

    // constexpr unsigned L1 = CONFIG_T::n_free_kernel;
    constexpr unsigned C = CONFIG_T::n_contract;
    // constexpr unsigned I = CONFIG_T::n_inplace;

    data_buff_T buff = data_pipe::read();
    if (buff.exit_task) {
        exit_task = true;
        return;
    }
    
    #pragma unroll 4
    for (unsigned c = 0; c < C; c++) {
    #ifdef AUTOREG
        token_buffer[c] = buff.data[c];
    #else
        token_buffer[c] = buff[c];
    #endif
    }
}

// weights are already transposed during compile-time in the config
template <class data_pipe, class res_pipe, typename CONFIG_T> void einsum_dense_stream() {

    constexpr unsigned L0 = CONFIG_T::n_free_data;
    constexpr unsigned L1 = CONFIG_T::n_free_kernel;
    constexpr unsigned C = CONFIG_T::n_contract;
    constexpr unsigned I = CONFIG_T::n_inplace;
    constexpr unsigned HEAD_DIM_IN = static_cast<unsigned>(CONFIG_T::n_contract / CONFIG_T::n_head);
    constexpr unsigned HEAD_DIM_OUT = static_cast<unsigned>(CONFIG_T::n_free_kernel / CONFIG_T::n_head);

#ifdef AUTOREG
    using Dense_in_pipe_T = typename ExtractPipeType<data_pipe>::value_type;
    using Dense_in_T = typename Dense_in_pipe_T::data_type;
    using Dense_out_pipe_T = typename ExtractPipeType<res_pipe>::value_type;
    using Dense_out_T = typename Dense_out_pipe_T::data_type;
    bool exit_task = 0;
#else
    using Dense_in_T = typename ExtractPipeType<data_pipe>::value_type;
    using Dense_out_T = typename ExtractPipeType<res_pipe>::value_type;
#endif

    using Dense_in_data_T = typename Dense_in_T::value_type;
    using Dense_concat_T = nnet::array<Dense_in_data_T, C>;
    using Dense_heads_T = nnet::array<Dense_in_data_T, L1>;

    [[intel::fpga_register]] Dense_in_T dense_in;
    [[intel::fpga_register]] Dense_out_T dense_out;
    [[intel::fpga_register]] Dense_concat_T dense_in_concat;
    [[intel::fpga_register]] Dense_heads_T dense_out_head;

#ifdef AUTOREG
    Dense_out_pipe_T dense_out_pipe;
    while(true){
        if (exit_task){
            dense_out_pipe.exit_task = true;
            res_pipe::write(dense_out_pipe);
            return;
        }
#endif
        //#pragma unroll CONFIG_T::parallelization_factor
        for (unsigned l0 = 0; l0 < L0; l0++) {

            #pragma unroll 4
            for (unsigned i = 0; i < I; i++) {

                if constexpr (!CONFIG_T::opt_dense){
                #ifdef AUTOREG
                    read_token<Dense_in_T, data_pipe, CONFIG_T>(dense_in, exit_task); // 1xC read
                    if (exit_task) {
                        dense_out_pipe.exit_task = true;
                        res_pipe::write(dense_out_pipe);
                        return;
                    }
                #else
                    read_token<Dense_in_T, data_pipe, CONFIG_T>(dense_in); // 1xC read
                #endif
                }

                // If this is attention output we join heads first
                if constexpr (CONFIG_T::opt_dense) {
                    for (unsigned h = 0; h < CONFIG_T::n_head; h++) {
                    #ifdef AUTOREG
                        auto dense_in_pipe = data_pipe::read();
                        if(dense_in_pipe.exit_task) {
                            exit_task = true;
                            dense_out_pipe.exit_task = true;
                            res_pipe::write(dense_out_pipe);
                            return;
                        }
                        dense_in = dense_in_pipe.data;
                    #else
                        dense_in = data_pipe::read();
                    #endif
                        for (unsigned c = 0; c < HEAD_DIM_IN; c++) {
                            dense_in_concat[HEAD_DIM_IN * h + c] = dense_in[c];
                        }
                    }
                }

                // Call the dense_resource function with the reordered weights
                if constexpr (!CONFIG_T::opt_dense) {
                    nnet::dense_resource<Dense_in_T, Dense_heads_T, typename CONFIG_T::dense_conf>(dense_in, dense_out_head);

                    for (unsigned h = 0; h < CONFIG_T::n_head; h++) {
                        for (unsigned l = 0; l < HEAD_DIM_OUT; l++) {
                            dense_out[l] = dense_out_head[HEAD_DIM_OUT * h + l];
                        }
                    #ifdef AUTOREG
                        dense_out_pipe.data = dense_out;
                        dense_out_pipe.exit_task = false;
                        res_pipe::write(dense_out_pipe);
                    #else    
                        res_pipe::write(dense_out);
                    #endif
                    }
                } else {
                    nnet::dense_resource<Dense_concat_T, Dense_out_T, typename CONFIG_T::dense_conf>(dense_in_concat, dense_out);
                    #ifdef AUTOREG
                        dense_out_pipe.data = dense_out;
                        dense_out_pipe.exit_task = false;
                        res_pipe::write(dense_out_pipe);
                    #else    
                        res_pipe::write(dense_out);
                    #endif
                }
            }
        }
#ifdef AUTOREG
    }
#endif
}

} // namespace nnet

#endif
