#ifndef NNET_BATCHNORM_STREAM_H_
#define NNET_BATCHNORM_STREAM_H_

#include "nnet_common.h"
#include "nnet_helpers.h"
#include "nnet_mult.h"
#include "nnet_types.h"

namespace nnet {

    
// ****************************************************
//       Streaming Batch Normalization
// ****************************************************
template <class data_pipe, class res_pipe, typename CONFIG_T>
void normalize_stream(typename CONFIG_T::scale_t scale, typename CONFIG_T::bias_t bias) {

    constexpr unsigned multiplier_limit = DIV_ROUNDUP(CONFIG_T::n_in, CONFIG_T::reuse_factor);
    constexpr unsigned pipeline = CONFIG_T::n_in / multiplier_limit;
#ifdef AUTOREG
    using data_pipe_T = typename ExtractPipeType<data_pipe>::value_type;
    using data_arr_T = typename data_pipe_T::data_type;

    using res_pipe_T = typename ExtractPipeType<res_pipe>::value_type;
    using res_arr_T = typename res_pipe_T::data_type;
#else
    using data_arr_T = typename ExtractPipeType<data_pipe>::value_type;
    using res_arr_T = typename ExtractPipeType<res_pipe>::value_type;
#endif

    constexpr auto datasize = std::tuple_size<data_arr_T>{};
    constexpr unsigned packets_per_tensor = CONFIG_T::n_in / datasize;
    CONFIG_T::template product<typename data_arr_T::value_type,
                               typename CONFIG_T::scale_t::value_type>::limit(multiplier_limit);

BatchNormLoop:

#ifdef AUTOREG
    unsigned i = 0;
    while (true){
        auto in_data = data_pipe::read();
        
        res_pipe_T out_data;

        if(in_data.exit_task){
            out_data.exit_task = true;
            res_pipe::write(out_data);
            break;
        }
#else
    //[[intel::initiation_interval(pipeline)]]
    for (int i = 0; i < packets_per_tensor; i++) {
        auto in_data = data_pipe::read();
        res_arr_T out_data;
#endif

    BatchNormpack:
        int norm_index = 0;
        #pragma unroll 4 // TODO: Introduce a config parameter for unroll factor here
        for (int j = 0; j < datasize; j++) {
            if constexpr (CONFIG_T::n_filt == -1)
                norm_index = i * datasize + j;
            else{
                //norm_index = j % CONFIG_T::n_filt;
                if (norm_index == CONFIG_T::n_filt) norm_index = 0;
            }
                
        
        #ifdef AUTOREG
            out_data.data[j] =
                CONFIG_T::template product<typename data_arr_T::value_type,
                                           typename CONFIG_T::scale_t::value_type>::product(in_data.data[j], scale[norm_index]) +
                bias[norm_index];  
        #else
            out_data[j] =
                CONFIG_T::template product<typename data_arr_T::value_type,
                                           typename CONFIG_T::scale_t::value_type>::product(in_data[j], scale[norm_index]) +
                bias[norm_index];
        #endif
            norm_index++;
        }

    #ifdef AUTOREG
        out_data.exit_task = false;
        // In autoreg mode we expect data to arrive in correct tensor-packet order
        i++;
        if (i == packets_per_tensor) i = 0;
    #endif
        res_pipe::write(out_data);
    }
}

// ****************************************************
//       Merged Batch Normalization and Quantized Tanh
// ****************************************************
template <class data_pipe, class res_pipe, typename CONFIG_T>
void normalize_binary_tanh_stream(typename CONFIG_T::threshold_t threshold) {

#ifdef AUTOREG
    using data_pipe_T = typename ExtractPipeType<data_pipe>::value_type;
    using data_arr_T = typename data_pipe_T::data_type;
    using res_pipe_T = typename ExtractPipeType<res_pipe>::value_type;
    using res_arr_T = typename res_pipe_T::data_type;
#else
    using data_arr_T = typename ExtractPipeType<data_pipe>::value_type;
    using res_arr_T = typename ExtractPipeType<res_pipe>::value_type;
#endif

    constexpr auto datasize = std::tuple_size<data_arr_T>{};
    constexpr unsigned packets_per_tensor = CONFIG_T::n_in / datasize;

BinaryNormLoop:
#ifdef AUTOREG
    unsigned i = 0;
    while (true) {
        auto in_data = data_pipe::read();
        res_pipe_T out_pipe;
        if (in_data.exit_task){
            out_pipe.exit_task = true;
            res_pipe::write(out_pipe);
            break;
        }
#else
    [[intel::initiation_interval(1)]] for (int i = 0; i < CONFIG_T::n_in / datasize; i++) {
        auto in_data = data_pipe::read();
#endif
        nnet::array<ac_int<1, false>, CONFIG_T::n_scale_bias> out_data;

    BatchNormPack:
        int norm_index = 0;
        #pragma unroll
        for (int j = 0; j < datasize; j++) {
            if constexpr (CONFIG_T::n_filt == -1)
                norm_index = i * datasize + j;
            else{            
                //norm_index = j % CONFIG_T::n_filt;
                if (norm_index == CONFIG_T::n_filt) norm_index = 0;
            }
        
        #ifdef AUTOREG
            out_data[j] = (in_data.data[j] >= threshold[norm_index]) ? 1 : 0;
        #else
            out_data[j] = (in_data[j] >= threshold[norm_index]) ? 1 : 0;
        #endif
            norm_index++;
        }
    #ifdef AUTOREG
        i++;
        if (i == packets_per_tensor) i = 0;
        out_pipe.data = out_data;
        out_pipe.exit_task = false;
        res_pipe::write(out_pipe);
    #else
        res_pipe::write(out_data);
    #endif
    }
}

template <class data_pipe, class res_pipe, typename CONFIG_T>
void normalize_ternary_tanh_stream(typename CONFIG_T::threshold_hi_t threshold_hi,
                                   typename CONFIG_T::threshold_lo_t threshold_lo) {

#ifdef AUTOREG
    using data_pipe_T = typename ExtractPipeType<data_pipe>::value_type;
    using data_arr_T = typename data_pipe_T::data_type;
    using res_pipe_T = typename ExtractPipeType<res_pipe>::value_type;
    using res_arr_T = typename res_pipe_T::data_type;
#else
    using data_arr_T = typename ExtractPipeType<data_pipe>::value_type;
    using res_arr_T = typename ExtractPipeType<res_pipe>::value_type;
#endif

    constexpr auto datasize = std::tuple_size<data_arr_T>{};
    constexpr unsigned packets_per_tensor = CONFIG_T::n_in / datasize;

TernaryNormLoop:
#ifdef AUTOREG
    unsigned i = 0;
    while (true) {
        auto in_data = data_pipe::read();
        res_pipe_T out_pipe;
        if (in_data.exit_task){
            out_pipe.exit_task = true;
            res_pipe::write(out_pipe);
            break;
        }
#else
    [[intel::initiation_interval(1)]] for (int i = 0; i < CONFIG_T::n_in / datasize; i++) {
        auto in_data = data_pipe::read();
#endif
        nnet::array<ac_int<2, true>, CONFIG_T::n_scale_bias> out_data;

    BatchNormPack:
        int norm_index = 0;
        #pragma unroll
        for (int j = 0; j < datasize; j++) {
            if constexpr (CONFIG_T::n_filt == -1)
                norm_index = i * datasize + j;
            else{            
                //norm_index = j % CONFIG_T::n_filt;
                if (norm_index == CONFIG_T::n_filt) norm_index = 0;
            }

        #ifdef AUTOREG
            if (in_data.data[j] > threshold_hi[norm_index])
                out_data[j] = 1;
            else if (in_data.data[j] <= threshold_lo[norm_index])
                out_data[j] = -1;
        #else
            if (in_data[j] > threshold_hi[norm_index])
                out_data[j] = 1;
            else if (in_data[j] <= threshold_lo[norm_index])
                out_data[j] = -1;
        #endif
            else
                out_data[j] = 0;
            norm_index++;
        }
    #ifdef AUTOREG
        i++;
        if (i == packets_per_tensor) i = 0;
        out_pipe.data = out_data;
        out_pipe.exit_task = false;
        res_pipe::write(out_pipe);
    #else
        res_pipe::write(out_data);
    #endif
    }
}

} // namespace nnet

#endif
