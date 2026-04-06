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

/*
//this will be a seperate kernel that reads from the pipe
template <class data_T, class res_pipe, class res_T, typename CONFIG_T>
void res_tpose_pipe(res_T &res){

    constexpr unsigned L0 = CONFIG_T::n_free_data;
    constexpr unsigned I  = CONFIG_T::n_inplace;
    constexpr unsigned L1 = CONFIG_T::n_free_kernel;

    //read from the pipe
    for (unsigned l0 = 0; l0 < L0; l0++) {
        for (unsigned i = 0; i < I; i++) {
            // Read one dense_out token from the pipe
            typename ExtractPipeType<res_pipe>::value_type dense_out = in_pipe.read();

            //transpose and write into the res array (check indexing)
            #pragma unroll
            for (unsigned j = 0; j < L1; j++) {
                res[(l0 * I + i) * L1 + j] = dense_out[j];
            }
        }
    }
}

#define DMA_BURST_WEIGHTS 1
template <typename CONFIG_T>
void read_weights(const typename CONFIG_T::weight_t &weights, unsigned l0){

    constexpr unsigned L1 = CONFIG_T::n_free_kernel;
    constexpr unsigned C = CONFIG_T::n_contract;
    constexpr unsigned I = CONFIG_T::n_inplace;

    using Dense_weights_T = nnet::array<typename CONFIG_T::weight_t::value_type, DMA_BURST_WEIGHTS * L1 * C>;
    [[intel::fpga_register]] Dense_weights_T dense_weights;

    for(unsigned i = 0; i < I; i++){

        const unsigned weights_offset = i * L1 * C;

        for (unsigned j = 0; j < L1; j++) {
            #pragma unroll
            for (unsigned k = 0; k < C; k++) {
                dense_weights[j * C + k] = weights[weights_offset + (k * L1 + j)];
            }
        }
    }
}
*/

//Read tokens from input stream into a buffer
template <class Dense_in_T, class data_pipe, typename CONFIG_T>
void read_token(Dense_in_T &token_buffer){
    
    using data_buff_T = typename ExtractPipeType<data_pipe>::value_type;

    //constexpr unsigned L1 = CONFIG_T::n_free_kernel;
    constexpr unsigned C = CONFIG_T::n_contract;
    //constexpr unsigned I = CONFIG_T::n_inplace;
    
    data_buff_T buff = data_pipe::read();
    #pragma unroll
    for (unsigned c = 0; c < C; c++) {
        token_buffer[c] = buff[c];
    }
}

//todo stream weights and biases too in the future?
template<class data_pipe, class res_pipe, typename CONFIG_T>
void einsum_dense_stream(
        typename CONFIG_T::weight_t weights,
        typename CONFIG_T::bias_t biases
    ){

    constexpr unsigned L0 = CONFIG_T::n_free_data;
    constexpr unsigned L1 = CONFIG_T::n_free_kernel;
    constexpr unsigned C = CONFIG_T::n_contract;
    constexpr unsigned I = CONFIG_T::n_inplace;

    using Dense_in_T = typename ExtractPipeType<data_pipe>::value_type;
    using Dense_in_data_T = typename Dense_in_T::value_type;
    using Dense_out_T = typename ExtractPipeType<res_pipe>::value_type;
    using Dense_out_data_T = typename Dense_out_T::value_type;
    using Dense_weights_T = nnet::array<typename CONFIG_T::weight_t::value_type, L1 * C>;
    using Dense_biases_T = nnet::array<typename CONFIG_T::bias_t::value_type, L1>;

    [[intel::fpga_register]] Dense_in_T dense_in;
    [[intel::fpga_register]] Dense_out_T dense_out;
    [[intel::fpga_register]] Dense_weights_T dense_weights;
    [[intel::fpga_register]] Dense_biases_T dense_biases;
    
    //we now stream over tokens
    #pragma unroll
    for (unsigned i = 0; i < I; i++) {

        //#pragma unroll CONFIG_T::parallelization_factor
        for (unsigned l0 = 0; l0 < L0; l0++) {

            read_token<Dense_in_T, data_pipe, CONFIG_T>(dense_in);// 1xC read

            // Reorder weights from column-major (source) to row-major (destination) during copy
            const unsigned weights_offset = i * L1 * C;
            #pragma unroll
            for (unsigned j = 0; j < L1; j++) {
                #pragma unroll
                for (unsigned k = 0; k < C; k++) {
                    dense_weights[j * C + k] = weights[weights_offset + (k * L1 + j)];
                }
            }

            const unsigned bias_offset = i * L0 * L1;
            #pragma unroll
            for (unsigned b_idx = 0; b_idx < L1; b_idx++) {
                dense_biases[b_idx] = biases[bias_offset + L1 * l0 + b_idx];
            }

            // Create a temporary config to ensure the types of the local buffers
            // match what dense_resource expects for its weight_t and bias_t.
            struct dense_slice_config : CONFIG_T::dense_conf {
                using weight_t = Dense_weights_T;
                using bias_t = Dense_biases_T;
            };

            // Call the dense_resource function with the reordered weights
            nnet::dense_resource<Dense_in_T, Dense_out_T, dense_slice_config>(dense_in, dense_out, 
                                                                                dense_weights, dense_biases);

            res_pipe::write(dense_out);
        }
    }
}

} // namespace nnet

#endif
