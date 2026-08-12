#ifndef NNET_ACTIVATION_STREAM_H_
#define NNET_ACTIVATION_STREAM_H_

#include "nnet_activation.h"
#include "nnet_common.h"
#include "nnet_types.h"
#include <cassert>

namespace nnet {

// *************************************************
//       Linear Activation
// *************************************************
template <class data_pipe, class res_pipe, typename CONFIG_T> void linear_stream() {

#ifdef AUTOREG
    using res_arr_T = typename ExtractPipeType<res_pipe>::value_type::data_type;
#else
    using res_arr_T = typename ExtractPipeType<res_pipe>::value_type;
#endif

LinearActLoop:
#ifdef AUTOREG
    while (true){
#else
    [[intel::initiation_interval(1)]] 
    for (int i = 0; i < CONFIG_T::n_in / std::tuple_size<res_arr_T>{}; i++) {
#endif
        auto in_data = data_pipe::read();
        typename ExtractPipeType<res_pipe>::value_type out_data;
    #ifdef AUTOREG
        out_data.feedback = in_data.feedback;
        if (in_data.exit_task){
            out_data.exit_task = true;
            res_pipe::write(out_data);
            return;
        }
    #endif

    LinearPackLoop:
        #pragma unroll
        for (int j = 0; j < std::tuple_size<res_arr_T>{}; j++) {
        #ifdef AUTOREG    
            out_data.data[j] = in_data.data[j];
        #else
            out_data[j] = in_data[j];
        #endif   
        }

        res_pipe::write(out_data);
    }
}

// *************************************************
//       ReLU Activation
// *************************************************
template <class data_pipe, class res_pipe, typename CONFIG_T> void relu_stream() {
ReLUActLoop:
#ifdef AUTOREG
    using res_arr_T = typename ExtractPipeType<res_pipe>::value_type::data_type;
    while (true){
#else
    using res_arr_T = typename ExtractPipeType<res_pipe>::value_type;
    [[intel::initiation_interval(
        1)]] for (int i = 0; i < CONFIG_T::n_in / std::tuple_size<res_arr_T>{}; i++) {
#endif
        auto in_data = data_pipe::read();
        typename ExtractPipeType<res_pipe>::value_type out_data;
    
    #ifdef AUTOREG
        out_data.feedback = in_data.feedback;
        if (in_data.exit_task){
            out_data.exit_task = true;
            res_pipe::write(out_data);
            return;
        }
    #endif

    ReLUPackLoop:
        #pragma unroll
        for (int j = 0; j < std::tuple_size<res_arr_T>{}; j++) {
        #ifdef AUTOREG 
            if (in_data.data[j] > 0)
                out_data.data[j] = in_data.data[j];
            else
                out_data.data[j] = 0;
        #else
            if (in_data[j] > 0)
                out_data[j] = in_data[j];
            else
                out_data[j] = 0;
        #endif   
        }
        res_pipe::write(out_data);
    }
}

// *************************************************
//       Leaky RELU Activation
// *************************************************
template <class data_pipe, class res_pipe, typename CONFIG_T> void leaky_relu_stream(typename CONFIG_T::param_t alpha) {
#ifdef AUTOREG
    auto in_pipe_size = std::tuple_size<typename ExtractPipeType<data_pipe>::value_type::data_type>{};

#else
    auto in_pipe_size = std::tuple_size<typename ExtractPipeType<data_pipe>::value_type>{};
#endif
    constexpr unsigned multiplier_limit = DIV_ROUNDUP(in_pipe_size, CONFIG_T::reuse_factor);
    constexpr unsigned pipeline = in_pipe_size / multiplier_limit;

LeakyReLUActLoop:
#ifdef AUTOREG
    using res_arr_T = typename ExtractPipeType<res_pipe>::value_type::data_type;
    while (true){
#else
    using res_arr_T = typename ExtractPipeType<res_pipe>::value_type;
    [[intel::initiation_interval(pipeline)]] for (int i = 0;
                                                  i < CONFIG_T::n_in /
                                                          std::tuple_size<res_arr_T>{};
                                                  i++) {
#endif
        auto in_data = data_pipe::read();
        typename ExtractPipeType<res_pipe>::value_type out_data;

    #ifdef AUTOREG
        out_data.feedback = in_data.feedback;
        if (in_data.exit_task){
            out_data.exit_task = true;
            res_pipe::write(out_data);
            return;
        }
    #endif

    LeakyReLUPackLoop:
        #pragma unroll
        for (int j = 0; j < std::tuple_size<res_arr_T>{}; j++) {
        #ifdef AUTOREG  
            if (in_data.data[j] > 0)
                out_data.data[j] = in_data.data[j];
            else
                out_data.data[j] = alpha * in_data.data[j];
        #else
            if (in_data[j] > 0)
                out_data[j] = in_data[j];
            else
                out_data[j] = alpha * in_data[j];
        #endif
        }

        res_pipe::write(out_data);
    }
}

// *************************************************
//       Thresholded RELU Activation
// *************************************************
template <class data_pipe, class res_pipe, typename CONFIG_T>
void thresholded_relu_stream(typename CONFIG_T::param_t theta) {
ThresholdedReLUActLoop:
#ifdef AUTOREG
    using res_arr_T = typename ExtractPipeType<res_pipe>::value_type::data_type;
    while (true){
#else
    using res_arr_T = typename ExtractPipeType<res_pipe>::value_type;
    [[intel::initiation_interval(1)]] 
    for (int i = 0; i < CONFIG_T::n_in / std::tuple_size<res_arr_T>{}; i++) {
#endif
        auto in_data = data_pipe::read();
        typename ExtractPipeType<res_pipe>::value_type out_data;

        #ifdef AUTOREG
            out_data.feedback = in_data.feedback;
            if (in_data.exit_task){
                out_data.exit_task = true;
                res_pipe::write(out_data);
                return;
            }
        #endif

    ThresholdedReLUPackLoop:
        #pragma unroll
        for (int j = 0; j < std::tuple_size<res_arr_T>{}; j++) {
        #ifdef AUTOREG
            if (in_data.data[j] > theta)
                out_data.data[j] = in_data.data[j];
            else
                out_data.data[j] = 0;
        #else
            if (in_data[j] > theta)
                out_data[j] = in_data[j];
            else
                out_data[j] = 0;
        #endif
        }

        res_pipe::write(out_data);
    }
}

// *************************************************
//       ELU Activation
// *************************************************
template <class data_pipe, class res_pipe, typename CONFIG_T> void elu_stream(typename CONFIG_T::param_t alpha) {
#include "activation_tables/elu_table.tb"

#ifdef AUTOREG
    auto in_pipe_size = std::tuple_size<typename ExtractPipeType<data_pipe>::value_type::data_type>{};
    using data_T = typename ExtractPipeType<data_pipe>::value_type::data_type::value_type;
#else
    using data_T = typename ExtractPipeType<data_pipe>::value_type::value_type;
    auto in_pipe_size = std::tuple_size<typename ExtractPipeType<data_pipe>::value_type>{};
#endif
    constexpr unsigned multiplier_limit = DIV_ROUNDUP(in_pipe_size, CONFIG_T::reuse_factor);
    constexpr unsigned pipeline = in_pipe_size / multiplier_limit;

EluActLoop:
#ifdef AUTOREG
    using res_arr_T = typename ExtractPipeType<res_pipe>::value_type::data_type;
    while (true){
#else
    using res_arr_T = typename ExtractPipeType<res_pipe>::value_type;
    [[intel::initiation_interval(pipeline)]] for (int i = 0;
                                                  i < CONFIG_T::n_in /
                                                          std::tuple_size<res_arr_T>{};
                                                  i++) {
#endif
        auto in_data = data_pipe::read();
        typename ExtractPipeType<res_pipe>::value_type out_data;

    #ifdef AUTOREG
        out_data.feedback = in_data.feedback;
        if (in_data.exit_task){
            out_data.exit_task = true;
            res_pipe::write(out_data);
            return;
        }
    #endif

    EluPackLoop:
        #pragma unroll
        for (int j = 0; j < std::tuple_size<res_arr_T>{}; j++) {
        #ifdef AUTOREG
            [[intel::fpga_register]] data_T datareg = in_data.data[j];
            if (datareg >= 0) {
                out_data.data[j] = datareg;
        #else
            [[intel::fpga_register]] data_T datareg = in_data[j];
            if (datareg >= 0) {
                out_data[j] = datareg; 
        #endif
            } else {
                int index = (datareg * CONFIG_T::table_size / -8).to_int();
                if (index > CONFIG_T::table_size - 1)
                    index = CONFIG_T::table_size - 1;
            #ifdef AUTOREG
                out_data.data[j] = alpha * elu_table[index];
            #else
                out_data[j] = alpha * elu_table[index];
            #endif
            }
        }

        res_pipe::write(out_data);
    }
}

// *************************************************
//       SeLU Activation
// *************************************************
template <class data_pipe, class res_pipe, typename CONFIG_T> void selu_stream() {
#include "activation_tables/selu_table.tb"

#ifdef AUTOREG
    using data_T = typename ExtractPipeType<data_pipe>::value_type::data_type::value_type;
    using res_arr_T = typename ExtractPipeType<res_pipe>::value_type::data_type;
#else
    using data_T = typename ExtractPipeType<data_pipe>::value_type::value_type;
    using res_arr_T = typename ExtractPipeType<res_pipe>::value_type;
#endif

SeluActLoop:
#ifdef AUTOREG
    while (true){
#else
    [[intel::initiation_interval(
        1)]] for (int i = 0; i < CONFIG_T::n_in / std::tuple_size<res_arr_T>{}; i++) {
#endif
        auto in_data = data_pipe::read();
        typename ExtractPipeType<res_pipe>::value_type out_data;

    #ifdef AUTOREG
        out_data.feedback = in_data.feedback;
        if (in_data.exit_task){
            out_data.exit_task = true;
            res_pipe::write(out_data);
            return;
        }
    #endif

    SeluPackLoop:
        #pragma unroll
        for (int j = 0; j < std::tuple_size<res_arr_T>{}; j++) {
            #ifdef AUTOREG
            [[intel::fpga_register]] data_T datareg = in_data.data[j];
            if (datareg >= 0) {
                out_data.data[j] =
                    static_cast<data_T>(1.0507009873554804934193349852946) * datareg;
            } else {
                int index = (datareg * CONFIG_T::table_size / -8).to_int();
                if (index > CONFIG_T::table_size - 1)
                    index = CONFIG_T::table_size - 1;
            
                out_data.data[j] = selu_table[index];
            #else
            [[intel::fpga_register]] data_T datareg = in_data[j];
            if (datareg >= 0) {
                out_data[j] =
                    static_cast<data_T>(1.0507009873554804934193349852946) * datareg;
            } else {
                int index = (datareg * CONFIG_T::table_size / -8).to_int();
                if (index > CONFIG_T::table_size - 1)
                    index = CONFIG_T::table_size - 1;
                
                out_data[j] = selu_table[index];
            #endif
            }
        }

        res_pipe::write(out_data);
    }
}

// *************************************************
//       PReLU Activation
// *************************************************
template <class data_pipe, class res_pipe, typename CONFIG_T> void prelu_stream(typename CONFIG_T::param_t alpha) {

#ifdef AUTOREG
    using data_T = typename ExtractPipeType<data_pipe>::value_type::data_type::value_type;
    using res_arr_T = typename ExtractPipeType<res_pipe>::value_type::data_type;
    auto in_pipe_size = std::tuple_size<typename ExtractPipeType<data_pipe>::value_type::data_type>{};

#else
    using data_T = typename ExtractPipeType<data_pipe>::value_type::value_type;
    using res_arr_T = typename ExtractPipeType<res_pipe>::value_type;
    auto in_pipe_size = std::tuple_size<typename ExtractPipeType<data_pipe>::value_type>{};
#endif

    constexpr unsigned multiplier_limit = DIV_ROUNDUP(in_pipe_size, CONFIG_T::reuse_factor);
    

PReLUActLoop:
#ifdef AUTOREG
    int i = 0;
    while (true){
#else
    constexpr unsigned pipeline = in_pipe_size / multiplier_limit;
    [[intel::initiation_interval(pipeline)]] for (int i = 0;
                                                  i < CONFIG_T::n_in /
                                                          std::tuple_size<res_arr_T>{};
                                                  i++) {
#endif
        auto in_data = data_pipe::read();
        typename ExtractPipeType<res_pipe>::value_type out_data;

    #ifdef AUTOREG
        out_data.feedback = in_data.feedback;
        if (in_data.exit_task){
            out_data.exit_task = true;
            res_pipe::write(out_data);
            return;
        }
    #endif

    PReLUPackLoop:
        #pragma unroll
        for (int j = 0; j < std::tuple_size<res_arr_T>{}; j++) {
        #ifdef AUTOREG
            if (in_data.data[j] > 0)
                out_data.data[j] = in_data.data[j];
            else
                out_data.data[j] = alpha[i * std::tuple_size<res_arr_T>{} + j] * in_data.data[j];
        #else
            if (in_data[j] > 0)
                out_data[j] = in_data[j];
            else
                out_data[j] = alpha[i * std::tuple_size<res_arr_T>{} + j] * in_data[j];
        #endif
        }
    #ifdef AUTOREG
        i = (i+1 >= (CONFIG_T::n_in / std::tuple_size<res_arr_T>{})) ? (i+1 - CONFIG_T::n_in / std::tuple_size<res_arr_T>{} ) : (i+1); 
    #endif
        res_pipe::write(out_data);
    }
}

// *************************************************
//       Softplus Activation
// *************************************************
template <class data_pipe, class res_pipe, typename CONFIG_T> void softplus_stream() {
#include "activation_tables/softplus_table.tb"

#ifdef AUTOREG
    using res_arr_T = typename ExtractPipeType<res_pipe>::value_type::data_type;
#else
    using res_arr_T = typename ExtractPipeType<res_pipe>::value_type;
#endif

SoftplusActLoop:
#ifdef AUTOREG
    while (true){
#else
    [[intel::initiation_interval(
        1)]] for (int i = 0; i < CONFIG_T::n_in / std::tuple_size<res_arr_T>{}; i++) {
#endif
        auto in_data = data_pipe::read();
        typename ExtractPipeType<res_pipe>::value_type out_data;

    #ifdef AUTOREG
        out_data.feedback = in_data.feedback;
        if (in_data.exit_task){
            out_data.exit_task = true;
            res_pipe::write(out_data);
            return;
        }
    #endif

    SoftplusPackLoop:
        #pragma unroll
        for (int j = 0; j < std::tuple_size<res_arr_T>{}; j++) {
        #ifdef AUTOREG
            [[intel::fpga_register]] int data_round = (in_data.data[j] * CONFIG_T::table_size / 16).to_int();
        #else
            [[intel::fpga_register]] int data_round = (in_data[j] * CONFIG_T::table_size / 16).to_int();
        #endif
            
            [[intel::fpga_register]] int index = data_round + 8 * CONFIG_T::table_size / 16;
            if (index < 0)
                index = 0;
            else if (index > CONFIG_T::table_size - 1)
                index = CONFIG_T::table_size - 1;
        #ifdef AUTOREG
            out_data.data[j] = softplus_table[index];
        #else
            out_data[j] = softplus_table[index];
        #endif
        }

        res_pipe::write(out_data);
    }
}

// *************************************************
//       Softsign Activation
// *************************************************
template <class data_pipe, class res_pipe, typename CONFIG_T> void softsign_stream() {
#include "activation_tables/softsign_table.tb"

#ifdef AUTOREG
    using data_T = typename ExtractPipeType<data_pipe>::value_type::data_type::value_type;
    using res_arr_T = typename ExtractPipeType<res_pipe>::value_type::data_type;
    auto in_pipe_size = std::tuple_size<typename ExtractPipeType<data_pipe>::value_type::data_type>{};

#else
    using data_T = typename ExtractPipeType<data_pipe>::value_type::value_type;
    using res_arr_T = typename ExtractPipeType<res_pipe>::value_type;
    auto in_pipe_size = std::tuple_size<typename ExtractPipeType<data_pipe>::value_type>{};
#endif

    static const int MAX_VALUE = 8;

SoftsignActLoop:
#ifdef AUTOREG
    while (true){
#else
    [[intel::initiation_interval(
        1)]] for (int i = 0; i < CONFIG_T::n_in / std::tuple_size<res_arr_T>{}; i++) {
#endif
        auto in_data = data_pipe::read();
        typename ExtractPipeType<res_pipe>::value_type out_data;

    #ifdef AUTOREG
        out_data.feedback = in_data.feedback;
        if (in_data.exit_task){
            out_data.exit_task = true;
            res_pipe::write(out_data);
            return;
        }
    #endif

    SoftsignPackLoop:
        #pragma unroll
        for (int j = 0; j < std::tuple_size<res_arr_T>{}; j++) {
    
            [[intel::fpga_register]] data_T absValue;
        #ifdef AUTOREG
            if (in_data.data[j] < 0) {
                absValue = -in_data.data[j];
            } else {
                absValue = in_data.data[j];
            }
            ac_int<16> index = (absValue * CONFIG_T::table_size / MAX_VALUE).to_int();
            if (absValue > MAX_VALUE)
                index = CONFIG_T::table_size - 1;
            if (in_data.data[j] < 0) {
                out_data.data[j] =
                    static_cast<typename res_arr_T::value_type>(-softsign_table[index]);
            } else {
                out_data.data[j] = static_cast<typename res_arr_T::value_type>(softsign_table[index]);
            }
        #else
            if (in_data[j] < 0) {
                absValue = -in_data[j];
            } else {
                absValue = in_data[j];
            }
            ac_int<16> index = (absValue * CONFIG_T::table_size / MAX_VALUE).to_int();
            if (absValue > MAX_VALUE)
                index = CONFIG_T::table_size - 1;
            if (in_data[j] < 0) {
                out_data[j] =
                    static_cast<typename ExtractPipeType<res_pipe>::value_type::value_type>(-softsign_table[index]);
            } else {
                out_data[j] = static_cast<typename ExtractPipeType<res_pipe>::value_type::value_type>(softsign_table[index]);
            }
        #endif
        }
        res_pipe::write(out_data);
    }
}

// *************************************************
//       Softmax Activation
// *************************************************

template <class data_pipe, class res_pipe, typename CONFIG_T> void softmax_stable_stream() {

#ifdef AUTOREG
    using data_packet_t = typename ExtractPipeType<data_pipe>::value_type;
    using input_arr_t = typename data_packet_t::data_type;
    using res_arr_T = typename ExtractPipeType<res_pipe>::value_type::data_type;
    using res_T = typename res_arr_T::value_type;
#else
    using input_arr_t = typename ExtractPipeType<data_pipe>::value_type;
    using res_arr_T = typename ExtractPipeType<res_pipe>::value_type;
    using res_T = typename res_arr_T::value_type;
#endif

    using input_t = typename input_arr_t::value_type;
    constexpr unsigned input_arr_size = std::tuple_size<input_arr_t>{};
    constexpr unsigned multiplier_limit = DIV_ROUNDUP(input_arr_size, CONFIG_T::reuse_factor);
    //constexpr unsigned pipeline = input_arr_size / multiplier_limit;

    [[intel::fpga_register]] input_t data_array[input_arr_size];
    // In causal version input_arr_size is ALWAYS equal to ctx_len
    assert(CONFIG_T::ctx_len == input_arr_size);
    [[intel::fpga_register]] unsigned mask_size = (CONFIG_T::ctx_len != 0) ? 1 : input_arr_size;

SoftmaxArrayLoop:

#ifdef AUTOREG
    while (true) {

        typename ExtractPipeType<res_pipe>::value_type out_pack;

        auto in_data_pack = data_pipe::read();
        auto in_pack = in_data_pack.data;
        out_pack.feedback = in_data_pack.feedback;

        if (in_data_pack.exit_task) {
            out_pack.exit_task = true;
            res_pipe::write(out_pack);
            break;
        }
#else
    //[[intel::initiation_interval(pipeline)]]
    for (unsigned i = 0; i < CONFIG_T::n_in / input_arr_size; i++) {

        typename ExtractPipeType<res_pipe>::value_type out_pack;

        auto in_pack = data_pipe::read();
#endif

    SoftmaxArrayPackLoop:
        #pragma unroll
        for (unsigned j = 0; j < input_arr_size; j++) {
            data_array[j] = in_pack[j];
        }

        // Find the max and compute all delta(x_i, x_max)
        Op_max<input_t> op_max;
        [[intel::fpga_register]] input_t x_max = reduce<input_t, input_arr_size, Op_max<input_t>>(data_array, op_max);

        [[intel::fpga_register]] typename CONFIG_T::inp_norm_t d_xi_xmax[input_arr_size];

        #pragma unroll
        for (unsigned j = 0; j < mask_size; j++) {
            d_xi_xmax[j] = x_max - data_array[j];
        }

        // Calculate all the e^x's
        [[intel::fpga_register]] typename CONFIG_T::accum_t exp_res[input_arr_size] = {};

        #pragma unroll
        for (unsigned j = 0; j < mask_size; j++) {
            exp_res[j] =
                CONFIG_T::exp_table[softmax_idx_from_real_val<typename CONFIG_T::inp_norm_t, CONFIG_T::exp_table_size>(
                    d_xi_xmax[j])];
        }

        // Explicitly sum the results with an adder tree.
        // Rounding & Saturation mode, which improve accuracy, prevent Vivado from expression balancing
        Op_add<typename CONFIG_T::accum_t> op_add;
        [[intel::fpga_register]] typename CONFIG_T::inv_inp_t exp_sum =
            reduce<typename CONFIG_T::accum_t, input_arr_size, Op_add<typename CONFIG_T::accum_t>>(exp_res, op_add);

        [[intel::fpga_register]] typename CONFIG_T::inv_table_t inv_exp_sum =
            CONFIG_T::invert_table[softmax_idx_from_real_val<typename CONFIG_T::inv_inp_t, CONFIG_T::inv_table_size>(
                exp_sum)];

        
    SoftmaxInvPackLoop:
        #pragma unroll
        for (unsigned j = 0; j < std::tuple_size<res_arr_T>{}; j++) {
        #ifdef AUTOREG
            if constexpr (CONFIG_T::ctx_len != 0){
                if(j < mask_size)
                    out_pack.data[j] = exp_res[j] * inv_exp_sum;
                else
                    out_pack.data[j] = static_cast<res_T>(0);
            }
            else{
                out_pack.data[j] = exp_res[j] * inv_exp_sum;
            }
        #else
            if constexpr (CONFIG_T::ctx_len != 0){
                if(j < mask_size)
                    out_pack[j] = exp_res[j] * inv_exp_sum;
                else
                    out_pack[j] = static_cast<res_T>(0);
            }
            else{
                out_pack[j] = exp_res[j] * inv_exp_sum;
            }
        #endif
        }
    #ifdef AUTOREG
        out_pack.exit_task = false;
    #endif
        res_pipe::write(out_pack);
        if constexpr (CONFIG_T::ctx_len != 0) mask_size = ((mask_size + 1) > CONFIG_T::ctx_len) ? CONFIG_T::ctx_len : (mask_size + 1);
    }
}

template <class data_pipe, class res_pipe, typename CONFIG_T> void softmax_latency_stream() {

#ifdef AUTOREG
    using data_packet_t = typename ExtractPipeType<data_pipe>::value_type;
    using input_arr_t = typename data_packet_t::data_type;
    using res_pipe_T = typename ExtractPipeType<res_pipe>::value_type;
    using res_arr_T = typename res_pipe_T::data_type;
#else
    using input_arr_t = typename ExtractPipeType<data_pipe>::value_type;
    using res_arr_T = typename ExtractPipeType<res_pipe>::value_type;
#endif

    constexpr unsigned input_arr_size = std::tuple_size<input_arr_t>{};

    constexpr unsigned multiplier_limit =
        DIV_ROUNDUP(input_arr_size, CONFIG_T::reuse_factor);
    constexpr unsigned pipeline = input_arr_size / multiplier_limit;

    // Calculate all the e^x's
    [[intel::fpga_register]]
    typename CONFIG_T::exp_table_t exp_res[input_arr_size];

SoftmaxExpLoop:    

#ifdef AUTOREG
    while (true) {

        typename ExtractPipeType<res_pipe>::value_type out_pack;

        auto in_data_pack = data_pipe::read();
        auto in_pack = in_data_pack.data;
        out_pack.feedback = in_data_pack.feedback;

        if (in_data_pack.exit_task) {
            out_pack.exit_task = true;
            res_pipe::write(out_pack);
            break;
        }
#else
    [[intel::initiation_interval(pipeline)]] 
    for (unsigned i = 0; i < CONFIG_T::n_in / input_arr_size; i++) {
        auto in_pack = data_pipe::read();
        typename ExtractPipeType<res_pipe>::value_type out_pack;
#endif
    SoftmaxExpPackLoop:
        #pragma unroll
        for (unsigned j = 0; j < std::tuple_size<input_arr_t>{}; j++) {
            exp_res[j] =
                CONFIG_T::exp_table[softmax_idx_from_real_val<typename input_arr_t::value_type,
                                                              CONFIG_T::exp_table_size>(in_pack[j])];
        }

        // Explicitly sum the results with an adder tree.
        // Rounding & Saturation mode, which improve accuracy, prevent Vivado from expression balancing
        Op_add<typename CONFIG_T::exp_table_t> op_add;
        [[intel::fpga_register]] typename CONFIG_T::exp_table_t exp_sum =
            reduce<typename CONFIG_T::exp_table_t, CONFIG_T::n_in, Op_add<typename CONFIG_T::exp_table_t>>(exp_res, op_add);

        // Multiply previously calculated exponetials with the reciprocal of the sum
        [[intel::fpga_register]] typename CONFIG_T::inv_table_t inv_exp_sum =
            CONFIG_T::invert_table[softmax_idx_from_real_val<typename CONFIG_T::exp_table_t, CONFIG_T::inv_table_size>(
                exp_sum)];

    SoftmaxInvPackLoop:
        #pragma unroll
        for (unsigned j = 0; j < std::tuple_size<res_arr_T>{}; j++) {
            // #pragma HLS ALLOCATION instances=mul limit=multiplier_limit operation
#ifdef AUTOREG
            out_pack.data[j] = exp_res[j] * inv_exp_sum;
#else
            out_pack[j] = exp_res[j] * inv_exp_sum;
#endif
        }
#ifdef AUTOREG
        out_pack.exit_task = false;
#endif
        res_pipe::write(out_pack);
    }
}

template <class data_pipe, class res_pipe, typename CONFIG_T> void softmax_legacy_stream() {
#ifdef AUTOREG
    using data_pipe_T = typename ExtractPipeType<data_pipe>::value_type;
    using data_arr_T = typename data_pipe_T::data_type;
    using res_pipe_T = typename ExtractPipeType<res_pipe>::value_type;
    using res_arr_T = typename res_pipe_T::data_type;
#else
    using data_arr_T = typename ExtractPipeType<data_pipe>::value_type;
    using res_arr_T = typename ExtractPipeType<res_pipe>::value_type;
#endif
    // Index into the lookup table based on data for exponentials
    [[intel::fpga_register]]
    typename CONFIG_T::table_t exp_res[std::tuple_size<data_arr_T>{}];
    [[intel::fpga_register]] typename CONFIG_T::table_t exp_diff_res;
    [[intel::fpga_register]] typename data_arr_T::value_type
        data_cache[std::tuple_size<data_arr_T>{}];

SoftmaxInitLoop:
#ifdef AUTOREG
    while (true){
#else
    [[intel::initiation_interval(1)]] for (unsigned s = 0;
                                           s < CONFIG_T::n_in /
                                                   std::tuple_size<data_arr_T>{};
                                           s++) {
#endif
        auto in_pack = data_pipe::read();
        typename ExtractPipeType<res_pipe>::value_type out_pack;

    #ifdef AUTOREG
        out_pack.feedback = in_pack.feedback;
        if (in_pack.exit_task){
            out_pack.exit_task = true;
            res_pipe::write(out_pack);
            return;
        }
    #endif       

    SoftmaxInitPackLoop:
        #pragma unroll
        for (unsigned j = 0; j < std::tuple_size<data_arr_T>{}; j++) {
        #ifdef AUTOREG
            data_cache[j] = in_pack.data[j];
        #else
            data_cache[j] = in_pack[j];
        #endif
            exp_res[j] = 0;
        }

    SoftmaxExpLoop:
        #pragma unroll
        for (int i = 0; i < std::tuple_size<data_arr_T>{}; i++) {
        SoftmaxExpInner:
            #pragma unroll
            for (int j = 0; j < std::tuple_size<data_arr_T>{}; j++) {
                if (i == j) {
                    exp_diff_res = 1;
                } else {
                    int data_round = ((data_cache[j] - data_cache[i]) * CONFIG_T::table_size / 16).to_int();
                    int index = data_round + 8 * CONFIG_T::table_size / 16;
                    if (index < 0)
                        index = 0;
                    if (index > CONFIG_T::table_size - 1)
                        index = CONFIG_T::table_size - 1;
                    exp_diff_res = CONFIG_T::exp_table[index];
                }
                exp_res[i] += exp_diff_res;
            }
        }

    SoftmaxInvPackLoop:
        #pragma unroll
        for (unsigned j = 0; j < std::tuple_size<res_arr_T>{}; j++) {
            int exp_res_index = (exp_res[j] * CONFIG_T::table_size / 64).to_int();
            if (exp_res_index < 0)
                exp_res_index = 0;
            if (exp_res_index > CONFIG_T::table_size - 1)
                exp_res_index = CONFIG_T::table_size - 1;
        #ifdef AUTOREG
            out_pack.data[j] = static_cast<typename res_arr_T::value_type>(
                CONFIG_T::invert_table[exp_res_index]);        
        #else
            out_pack[j] = static_cast<typename res_arr_T::value_type>(
                CONFIG_T::invert_table[exp_res_index]);
        #endif 
        }

        res_pipe::write(out_pack);
    }
}

template <class data_pipe, class res_pipe, typename CONFIG_T> void softmax_argmax_stream() {
#ifdef AUTOREG
    using data_pipe_T = typename ExtractPipeType<data_pipe>::value_type;
    using data_arr_T = typename data_pipe_T::data_type;
    using res_pipe_T = typename ExtractPipeType<res_pipe>::value_type;
    using res_arr_T = typename res_pipe_T::data_type;
#else
    using data_arr_T = typename ExtractPipeType<data_pipe>::value_type;
    using res_arr_T = typename ExtractPipeType<res_pipe>::value_type;
#endif

#ifdef AUTOREG
    while (true){
#else
    [[intel::initiation_interval(
        1)]] for (int i = 0; i < CONFIG_T::n_in / std::tuple_size<res_arr_T>{}; i++) {
#endif
        auto in_data = data_pipe::read();
        typename ExtractPipeType<res_pipe>::value_type out_data;

    #ifdef AUTOREG
        out_data.feedback = in_data.feedback;
        if (in_data.exit_task){
            out_data.exit_task = true;
            res_pipe::write(out_data);
            return;
        }
    #endif   

        #pragma unroll
        for (int i = 0; i < std::tuple_size<res_arr_T>{}; i++) {
        #ifdef AUTOREG
            out_data.data[i] = static_cast<typename res_arr_T::value_type>(0);
        #else
            out_data[i] = static_cast<typename res_arr_T::value_type>(0);
        #endif
        }
        
    #ifdef AUTOREG
        [[intel::fpga_register]] typename data_arr_T::value_type maximum = in_data.data[0];
    #else
        [[intel::fpga_register]] typename data_arr_T::value_type maximum = in_data[0];
    #endif
        [[intel::fpga_register]] int idx = 0;

        [[intel::initiation_interval(1)]] for (int i = 1;
                                               i < std::tuple_size<res_arr_T>{}; i++) {
            
            #ifdef AUTOREG
            if (in_data.data[i] > maximum) {
                maximum = in_data.data[i];
            #else 
            if (in_data[i] > maximum) {
                maximum = in_data[i];
            #endif
                idx = i;
            }
        }
    #ifdef AUTOREG
        out_data.data[idx] = static_cast<typename res_arr_T::value_type>(1);
    #else
        out_data[idx] = static_cast<typename res_arr_T::value_type>(1);
    #endif
        res_pipe::write(out_data);
    }
}

template <class data_pipe, class res_pipe, typename CONFIG_T> void softmax_stream() {
    if constexpr (CONFIG_T::implementation == softmax_implementation::latency) {
        softmax_latency_stream<data_pipe, res_pipe, CONFIG_T>();
    } else if constexpr (CONFIG_T::implementation == softmax_implementation::argmax) {
        softmax_argmax_stream<data_pipe, res_pipe, CONFIG_T>();
    } else if constexpr (CONFIG_T::implementation == softmax_implementation::legacy) {
        softmax_argmax_stream<data_pipe, res_pipe, CONFIG_T>();
    } else { // Default to stable
        softmax_stable_stream<data_pipe, res_pipe, CONFIG_T>();
    }
}

// *************************************************
//       Multidimensional Softmax
// *************************************************
template <class data_pipe, class res_pipe, typename CONFIG_T> inline void softmax_multidim_stream() {
#ifdef AUTOREG
    using data_pipe_T = typename ExtractPipeType<data_pipe>::value_type;
    using data_arr_T = typename data_pipe_T::data_type;
    using data_T = typename data_arr_T::value_type;
    using res_pipe_T = typename ExtractPipeType<res_pipe>::value_type;
    using res_arr_T = typename res_pipe_T::data_type;
    using res_T = typename res_arr_T::value_type;

    [[intel::fpga_register]] res_pipe_T out_pack_pipe;
#else
    using data_arr_T = typename ExtractPipeType<data_pipe>::value_type;
    using data_T = typename data_arr_T::value_type;
    using res_arr_T = typename ExtractPipeType<res_pipe>::value_type;
    using res_T = typename res_arr_T::value_type;
#endif

    using in_slice_arr_T = nnet::array<data_T, CONFIG_T::n_slice>;
    using out_slice_arr_T = nnet::array<res_T, CONFIG_T::n_slice>;

    using slice_config = softmax_multidim_slice_config<CONFIG_T>;

    [[intel::fpga_register]] data_arr_T buffer_in;
    [[intel::fpga_register]] res_arr_T buffer_out;
    [[intel::fpga_register]] in_slice_arr_T smax_slice_in;
    [[intel::fpga_register]] out_slice_arr_T smax_slice_out;

#ifdef AUTOREG
    while (true) {
#else
    static constexpr unsigned TENSOR_SIZE = CONFIG_T::n_slice * CONFIG_T::n_inner * CONFIG_T::n_outer;
    for (unsigned t = 0; t < CONFIG_T::n_in / TENSOR_SIZE; t++){
#endif

    #ifdef AUTOREG
        [[intel::fpga_register]] data_pipe_T buffer_in_pipe = data_pipe::read();
        out_pack_pipe.feedback = buffer_in_pipe.feedback;
        if (buffer_in_pipe.exit_task){
            out_pack_pipe.exit_task = true;
            res_pipe::write(out_pack_pipe);
            return;
        }
        buffer_in = buffer_in_pipe.data;
    #else
        buffer_in = data_pipe::read();
    #endif   

        #pragma unroll
        for (unsigned i = 0; i < CONFIG_T::n_outer; i++) {
            unsigned outer_offset = i * CONFIG_T::n_slice * CONFIG_T::n_inner;
            #pragma unroll
            for (unsigned k = 0; k < CONFIG_T::n_inner; k++) {

                // TODO: Access might be inefficient consider data rearrangement
                #pragma unroll
                for (unsigned j = 0; j < CONFIG_T::n_slice; j++) {
                    unsigned idx = outer_offset + j * CONFIG_T::n_inner + k;
                    smax_slice_in[j] = buffer_in[idx];
                }

                nnet::softmax<in_slice_arr_T, out_slice_arr_T, slice_config>(smax_slice_in, smax_slice_out);

                #pragma unroll
                for (unsigned j = 0; j < CONFIG_T::n_slice; j++) {
                    unsigned idx = outer_offset + j * CONFIG_T::n_inner + k;
                    buffer_out[idx] = smax_slice_out[j];
                }
            }
        }
    #ifdef AUTOREG
        out_pack_pipe.data = buffer_out;
        out_pack_pipe.exit_task = false;
        res_pipe::write(out_pack_pipe);
    #else
        res_pipe::write(buffer_out);
    #endif 
    }
}

// *************************************************
//       TanH Activation
// *************************************************
template <class data_pipe, class res_pipe, typename CONFIG_T> void dense_tanh_stream() {
#include "activation_tables/tanh_table.tb"

#ifdef AUTOREG
    using data_pipe_T = typename ExtractPipeType<data_pipe>::value_type;
    using data_arr_T = typename data_pipe_T::data_type;
    using res_pipe_T = typename ExtractPipeType<res_pipe>::value_type;
    using res_arr_T = typename res_pipe_T::data_type;
#else
    using data_arr_T = typename ExtractPipeType<data_pipe>::value_type;
    using res_arr_T = typename ExtractPipeType<res_pipe>::value_type;
#endif

    static const int MAX_VALUE = 4;

    constexpr unsigned multiplier_limit =
        DIV_ROUNDUP(std::tuple_size<data_arr_T>{}, CONFIG_T::reuse_factor);
    constexpr unsigned pipeline = std::tuple_size<data_arr_T>{} / multiplier_limit;

TanHActLoop:
#ifdef AUTOREG
    while (true){
        data_pipe_T in_data_pipe = data_pipe::read();
        res_pipe_T out_data_pipe;
        out_data_pipe.feedback = in_data_pipe.feedback;
        if(in_data_pipe.exit_task){
            out_data_pipe.exit_task = true;
            res_pipe::write(out_data_pipe);
            break;
        }
        data_arr_T in_data = in_data_pipe.data;
#else
    [[intel::initiation_interval(pipeline)]] for (int i = 0;
                                                  i < CONFIG_T::n_in /
                                                          std::tuple_size<res_arr_T>{};
                                                  i++) {
        auto in_data = data_pipe::read();
#endif
        res_arr_T out_data;

    TanHPackLoop:
        #pragma unroll
        for (int j = 0; j < std::tuple_size<res_arr_T>{}; j++) {
            [[intel::fpga_register]] typename data_arr_T::value_type absoluteValue;

            if (in_data[j] < 0)
                absoluteValue = (-1) * in_data[j];
            else
                absoluteValue = in_data[j];

            [[intel::fpga_register]] int index;
            if (absoluteValue <= MAX_VALUE)
                index = (absoluteValue * (CONFIG_T::table_size / MAX_VALUE)).to_int();
            else
                index = CONFIG_T::table_size - 1;

            if (in_data[j] > 0)
                out_data[j] = tanh_table[index];
            else
                out_data[j] = -tanh_table[index];
        }

    #ifdef AUTOREG
        out_data_pipe.data = out_data;
        out_data_pipe.exit_task = false;
        res_pipe::write(out_data_pipe);
    #else
        res_pipe::write(out_data);
    #endif
    }
}

// *************************************************
//       Sigmoid Activation
// *************************************************
template <class data_pipe, class res_pipe, typename CONFIG_T> void sigmoid_stream() {
#include "activation_tables/sigmoid_table.tb"

#ifdef AUTOREG
    using data_pipe_T = typename ExtractPipeType<data_pipe>::value_type;
    using data_arr_T = typename data_pipe_T::data_type;
    using res_pipe_T = typename ExtractPipeType<res_pipe>::value_type;
    using res_arr_T = typename res_pipe_T::data_type;
#else
    using data_arr_T = typename ExtractPipeType<data_pipe>::value_type;
    using res_arr_T = typename ExtractPipeType<res_pipe>::value_type;
#endif

    static const int MAX_VALUE = 8;

    constexpr unsigned multiplier_limit =
        DIV_ROUNDUP(std::tuple_size<data_arr_T>{}, CONFIG_T::reuse_factor);
    constexpr unsigned pipeline = std::tuple_size<data_arr_T>{} / multiplier_limit;

SigmoidActLoop:
#ifdef AUTOREG
    while (true){
        data_pipe_T in_data_pipe = data_pipe::read();
        res_pipe_T out_data_pipe;
        out_data_pipe.feedback = in_data_pipe.feedback;
        if(in_data_pipe.exit_task){
            out_data_pipe.exit_task = true;
            res_pipe::write(out_data_pipe);
            break;
        }
        data_arr_T in_data = in_data_pipe.data;
#else
    [[intel::initiation_interval(pipeline)]] for (int i = 0;
                                                  i < CONFIG_T::n_in /
                                                          std::tuple_size<res_arr_T>{};
                                                  i++) {
        auto in_data = data_pipe::read();
#endif
        res_arr_T out_data;

    SigmoidPackLoop:
        #pragma unroll
        for (int j = 0; j < std::tuple_size<res_arr_T>{}; j++) {
            [[intel::fpga_register]] typename data_arr_T::value_type absoluteValue;

            if (in_data[j] < 0)
                absoluteValue = (-1) * in_data[j];
            else
                absoluteValue = in_data[j];

            [[intel::fpga_register]] int index;
            if (absoluteValue <= MAX_VALUE)
                index = (absoluteValue * (CONFIG_T::table_size / MAX_VALUE)).to_int();
            else
                index = CONFIG_T::table_size - 1;

            if (in_data[j] > 0)
                out_data[j] = sigmoid_table[index];
            else
                out_data[j] = 1 - sigmoid_table[index];
        }

    #ifdef AUTOREG
        out_data_pipe.data = out_data;
        out_data_pipe.exit_task = false;
        res_pipe::write(out_data_pipe);
    #else
        res_pipe::write(out_data);
    #endif
    }
}

// *************************************************
//       Hard sigmoid Activation
// *************************************************
// Note - Theano and Tensorflow might have different definitions for hard sigmoid; could provide two implementations
template <class data_pipe, class res_pipe, typename CONFIG_T> void hard_sigmoid_stream() {

#ifdef AUTOREG
    using data_pipe_T = typename ExtractPipeType<data_pipe>::value_type;
    using data_arr_T = typename data_pipe_T::data_type;
    using res_pipe_T = typename ExtractPipeType<res_pipe>::value_type;
    using res_arr_T = typename res_pipe_T::data_type;
#else
    using data_arr_T = typename ExtractPipeType<data_pipe>::value_type;
    using res_arr_T = typename ExtractPipeType<res_pipe>::value_type;
#endif

    constexpr unsigned multiplier_limit =
        DIV_ROUNDUP(std::tuple_size<data_arr_T>{}, CONFIG_T::reuse_factor);
    constexpr unsigned pipeline = std::tuple_size<data_arr_T>{} / multiplier_limit;

HardSigmoidActLoop:
#ifdef AUTOREG
    while (true){
        data_pipe_T in_data_pipe = data_pipe::read();
        res_pipe_T out_data_pipe;
        out_data_pipe.feedback = in_data_pipe.feedback;
        if(in_data_pipe.exit_task){
            out_data_pipe.exit_task = true;
            res_pipe::write(out_data_pipe);
            break;
        }
        data_arr_T in_data = in_data_pipe.data;
#else
    [[intel::initiation_interval(pipeline)]] for (int i = 0;
                                                  i < CONFIG_T::n_in /
                                                          std::tuple_size<res_arr_T>{};
                                                  i++) {

        auto in_data = data_pipe::read();
#endif
        res_arr_T out_data;

    HardSigmoidPackLoop:
        #pragma unroll
        for (int j = 0; j < std::tuple_size<res_arr_T>{}; j++) {
            [[intel::fpga_register]] auto datareg = CONFIG_T::slope * in_data[j] + CONFIG_T::shift;
            if (datareg > 1)
                datareg = 1;
            else if (datareg < 0)
                datareg = 0;
            out_data[j] = datareg;
        }

    #ifdef AUTOREG
        out_data_pipe.data = out_data;
        out_data_pipe.exit_task = false;
        res_pipe::write(out_data_pipe);
    #else
        res_pipe::write(out_data);
    #endif
    }
}

template <class data_pipe, class res_pipe, typename CONFIG_T> void hard_tanh_stream() {

#ifdef AUTOREG
    using data_pipe_T = typename ExtractPipeType<data_pipe>::value_type;
    using data_arr_T = typename data_pipe_T::data_type;
    using res_pipe_T = typename ExtractPipeType<res_pipe>::value_type;
    using res_arr_T = typename res_pipe_T::data_type;
#else
    using data_arr_T = typename ExtractPipeType<data_pipe>::value_type;
    using res_arr_T = typename ExtractPipeType<res_pipe>::value_type;
#endif

    constexpr unsigned multiplier_limit =
        DIV_ROUNDUP(std::tuple_size<data_arr_T>{}, CONFIG_T::reuse_factor);
    constexpr unsigned pipeline = std::tuple_size<data_arr_T>{} / multiplier_limit;

HardSigmoidActLoop:
#ifdef AUTOREG
    while (true){
        data_pipe_T in_data_pipe = data_pipe::read();
        res_pipe_T out_data_pipe;
        out_data_pipe.feedback = in_data_pipe.feedback;
        if(in_data_pipe.exit_task){
            out_data_pipe.exit_task = true;
            res_pipe::write(out_data_pipe);
            break;
        }
        data_arr_T in_data = in_data_pipe.data;
#else
    [[intel::initiation_interval(pipeline)]] for (int i = 0;
                                                  i < CONFIG_T::n_in /
                                                          std::tuple_size<res_arr_T>{};
                                                  i++) {

        auto in_data = data_pipe::read();
#endif
        res_arr_T out_data;

    HardSigmoidPackLoop:
        #pragma unroll
        for (int j = 0; j < std::tuple_size<res_arr_T>{}; j++) {
            auto sigmoid = CONFIG_T::slope * in_data[j] + CONFIG_T::shift;
            if (sigmoid > 1)
                sigmoid = 1;
            else if (sigmoid < 0)
                sigmoid = 0;
            out_data[j] = 2 * sigmoid - 1;
        }

    #ifdef AUTOREG
        out_data_pipe.data = out_data;
        out_data_pipe.exit_task = false;
        res_pipe::write(out_data_pipe);
    #else
        res_pipe::write(out_data);
    #endif
    }
}

// *************************************************
//       Binary TanH Activation
// *************************************************
template <class data_pipe, class res_pipe, typename CONFIG_T> void binary_tanh_stream() {

#ifdef AUTOREG
    using data_pipe_T = typename ExtractPipeType<data_pipe>::value_type;
    using data_arr_T = typename data_pipe_T::data_type;
    using res_pipe_T = typename ExtractPipeType<res_pipe>::value_type;
    using res_arr_T = typename res_pipe_T::data_type;
#else
    using data_arr_T = typename ExtractPipeType<data_pipe>::value_type;
    using res_arr_T = typename ExtractPipeType<res_pipe>::value_type;
#endif

    using cache_T = ac_int<2, true>;

BinaryTanHActLoop:
#ifdef AUTOREG
    while (true){
        data_pipe_T in_data_pipe = data_pipe::read();
        res_pipe_T out_data_pipe;
        out_data_pipe.feedback = in_data_pipe.feedback;
        if(in_data_pipe.exit_task){
            out_data_pipe.exit_task = true;
            res_pipe::write(out_data_pipe);
            break;
        }
        data_arr_T in_data = in_data_pipe.data;
#else
    [[intel::initiation_interval(
        1)]] for (int i = 0; i < CONFIG_T::n_in / std::tuple_size<res_arr_T>{}; i++) {

        [[intel::fpga_register]] auto in_data = data_pipe::read();
#endif
        [[intel::fpga_register]] res_arr_T out_data;

    BinaryTanHPackLoop:
        #pragma unroll
        for (int j = 0; j < std::tuple_size<res_arr_T>{}; j++) {
            cache_T cache;

            if (in_data[j] >= 0)
                cache = 1;
            else
                cache = -1;

            out_data[j] = binary_cast<cache_T, typename res_arr_T::value_type>(cache);
        }

    #ifdef AUTOREG
        out_data_pipe.data = out_data;
        out_data_pipe.exit_task = false;
        res_pipe::write(out_data_pipe);
    #else
        res_pipe::write(out_data);
    #endif
    }
}

// *************************************************
//       Ternary TanH Activation
// *************************************************
template <class data_pipe, class res_pipe, typename CONFIG_T> void ternary_tanh_stream() {

#ifdef AUTOREG
    using data_pipe_T = typename ExtractPipeType<data_pipe>::value_type;
    using data_arr_T = typename data_pipe_T::data_type;
    using res_pipe_T = typename ExtractPipeType<res_pipe>::value_type;
    using res_arr_T = typename res_pipe_T::data_type;
#else
    using data_arr_T = typename ExtractPipeType<data_pipe>::value_type;
    using res_arr_T = typename ExtractPipeType<res_pipe>::value_type;
#endif    

TernaryTanHActLoop:
#ifdef AUTOREG
    while (true){
        data_pipe_T in_data_pipe = data_pipe::read();
        res_pipe_T out_data_pipe;
        out_data_pipe.feedback = in_data_pipe.feedback;
        if(in_data_pipe.exit_task){
            out_data_pipe.exit_task = true;
            res_pipe::write(out_data_pipe);
            break;
        }
        data_arr_T in_data = in_data_pipe.data;
#else
    [[intel::initiation_interval(
        1)]] for (int i = 0; i < CONFIG_T::n_in / std::tuple_size<res_arr_T>{}; i++) {

        [[intel::fpga_register]] auto in_data = data_pipe::read();
#endif
        [[intel::fpga_register]] res_arr_T out_data;

    TernaryTanHPackLoop:
        #pragma unroll
        for (int j = 0; j < std::tuple_size<res_arr_T>{}; j++) {
            if (in_data[j] > 1)
                out_data[j] = static_cast<typename res_arr_T::value_type>(1);
            else if (in_data[j] <= -1)
                out_data[j] = static_cast<typename res_arr_T::value_type>(-1);
            else
                out_data[j] = static_cast<typename res_arr_T::value_type>(0);
        }

    #ifdef AUTOREG
        out_data_pipe.data = out_data;
        out_data_pipe.exit_task = false;
        res_pipe::write(out_data_pipe);
    #else
        res_pipe::write(out_data);
    #endif
    }
}

} // namespace nnet

#endif
