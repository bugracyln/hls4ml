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

//Read tokens from input stream into a buffer
template <class data_T, typename CONFIG_T>
void read_token(
    const data_T &data, 
    typename data_T::value_type token_buffer[CONFIG_T::n_contract * CONFIG_T::n_inplace],
    unsigned &token_idx){

    constexpr unsigned C = CONFIG_T::n_contract;
    constexpr unsigned I = CONFIG_T::n_inplace;

    for(unsigned i = 0; i < I; i++){
        for (unsigned channel = 0; channel < C; channel++) {
            //transpose while reading ?
            token_buffer[C * i + channel] = data[C * (I * token_idx + i) + channel];
        }
    }
    token_idx++;
}


template <class data_pipe, class res_pipe, class data_T, class res_T, typename CONFIG_T>
void causal_einsum_dense(
        data_T &data,
        unsigned &token_idx,
        data_window_T &past_vals,
        const typename CONFIG_T::weight_t &weights,
        const typename CONFIG_T::bias_t &biases
    ) {

    //using data_arr_T = typename ExtractPipeType<data_pipe>::value_type;
    //using data_element_T = typename data_arr_T::value_type;
    //using data_window_T = array<data_element_T, CONFIG_T::filt_width * CONFIG_T::n_chan>;

    //[[intel::fpga_register]] data_T inp_tpose;
    //[intel::fpga_register]] res_T out_tpose;
    
    //NO NEED TO TRANSPOSE AS WE EXECUTE CAUSAL EINSUM PER-L0
    //nnet::transpose<data_T, data_T, typename CONFIG_T::tpose_inp_conf>(data_pipe::read(), inp_tpose);


    constexpr unsigned L0 = CONFIG_T::n_free_data;
    constexpr unsigned L1 = CONFIG_T::n_free_kernel;
    constexpr unsigned C = CONFIG_T::n_contract;
    constexpr unsigned I = CONFIG_T::n_inplace;

    using Dense_in_T = nnet::array<typename data_T::value_type, C>;
    using Dense_out_T = nnet::array<typename res_T::value_type, L1>;
    using Dense_weights_T = nnet::array<typename CONFIG_T::weight_t::value_type, L1 * C>;
    using Dense_biases_T = nnet::array<typename CONFIG_T::bias_t::value_type, L1>;
    
    //create the buffer to store vals 
    //typename data_element_T buffer[I * C];

    //#pragma unroll CONFIG_T::parallelization_factor
    //we now stream over tokens

    //read_token<data_element_T, CONFIG_T>(data_pipe::read(), buffer, token_idx);

    #pragma unroll //- assuming I is reasonable
    for (unsigned i = 0; i < I; i++) {
        [[intel::fpga_register]] Dense_in_T dense_in;
        [[intel::fpga_register]] Dense_out_T dense_out;
        [[intel::fpga_register]] Dense_weights_T dense_weights;
        [[intel::fpga_register]] Dense_biases_T dense_biases;

        #pragma unroll
        for (unsigned c_idx = 0; c_idx < C; c_idx++) {
            dense_in[c_idx] = data[C * i + c_idx];
        }

        // Reorder weights from column-major (source) to row-major (destination) during copy
        const unsigned weights_offset = i * L1 * C;
        #pragma unroll
        for (unsigned j = 0; j < L1; j++) {
            #pragma unroll
            for (unsigned k = 0; k < C; k++) {
                dense_weights[j * C + k] = weights[weights_offset + (k * L1 + j)];
            }
        }

        #pragma unroll
        for (unsigned b_idx = 0; b_idx < L1; b_idx++) {
            dense_biases[b_idx] = biases[((i * L0 + token_idx) * L1) + b_idx]; //token_idx is equivalent to l0 (?)
        }

        // Create a temporary config to ensure the types of the local buffers
        // match what dense_resource expects for its weight_t and bias_t.
        struct dense_slice_config : CONFIG_T::dense_conf {
            using weight_t = Dense_weights_T;
            using bias_t = Dense_biases_T;
        };

        // Call the dense_resource function with the reordered weights
        nnet::dense_resource<Dense_in_T, Dense_out_T, dense_slice_config>(dense_in, dense_out, dense_weights,
                                                                            dense_biases);

        //the output is 1 x L1 of qW kW vW etc.
        #pragma unroll
        for(unsigned l1 = 0; l1 < L1; l1++){
            past_vals[L1 * token_idx + l1] = dense_out[l1];
        }

        //TODO: WRITE ALL HEADS AT ONCE AS A SINGLE ITEM SO OTHER LAYERS CAN READ
        //OR CHANGE EINSUM DEF TO READ PER i (AVOID READING PER I FOR NORMAL EINSUM THOUGH)
        //TODO: CHECK WHAT FORMAT DO WE WRITE IN/EXPECT FOR IN NON-STREAMED VERSIONS
        res_pipe::write(dense_out);

        //#pragma unroll
        //for (unsigned j = 0; j < L1; j++) {
        //    out_tpose[((i * L0 + l0) * L1) + j] = dense_out[j];
        //}
    }

    //nnet::transpose<res_T, res_T, typename CONFIG_T::tpose_out_conf>(out_tpose, res);
}

//works by taking a streamed token, calculating the outpupt vector and appending to the respective context window
#define CONTEXT_LENGTH 256
template<class data_pipe, class res_pipe, typename CONFIG_T>
void stream_causal_einsum_dense(typename CONFIG_T::weight_t weights, typename CONFIG_T::bias_t biases){

    constexpr unsigned L1 = CONFIG_T::n_free_kernel;
    constexpr unsigned C = CONFIG_T::n_contract;
    constexpr unsigned I = CONFIG_T::n_inplace;

    using data_arr_T = typename ExtractPipeType<data_pipe>::value_type;
    using data_element_T = typename data_arr_T::value_type;
    using data_window_T = array<data_element_T, CONTEXT_LENGTH * I * L1>;

    using res_T = typename ExtractPipeType<res_pipe>::value_type;

    //window to keep context TODO: DECLARE THIS GLOBALLY ONCE, NOT PER CALL!
    static data_window_T past_vals;

    //circular pointer to keep track of data in stream
    int token_idx = 0;
    
    //create the buffer to store vals 
    typename data_element_T buffer[I * C];

    int buffer_idx = token_idx % CONTEXT_LENGTH;

    //read token from pipe into a buffer
    read_token<data_arr_T, CONFIG_T>(data_pipe::read(), buffer, token_idx);

    //process token and append to a BRAM location
    causal_einsum_dense<data_pipe, res_pipe, data_element_T, res_T, CONFIG_T>(buffer, buffer_idx,
        past_vals, weights, biases); 
}

} // namespace nnet

#endif
