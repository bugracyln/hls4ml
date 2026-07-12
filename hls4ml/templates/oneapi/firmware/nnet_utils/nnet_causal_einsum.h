#ifndef NNET_EINSUM_STREAMED_H_
#define NNET_EINSUM_STREAMED_H_

#include "nnet_common.h"
#include "nnet_dense.h"
#include "nnet_helpers.h"
#include "nnet_mult.h"
#include "nnet_transpose.h"
#include <limits>
#include <sycl/ext/oneapi/device_global/device_global.hpp>
#include <type_traits>

namespace nnet {

struct config_causal_einsum {
    typedef void tpose_inp0_config;
    typedef void tpose_inp1_config;
    typedef void tpose_out_conf;

    // Layer Sizes
    static const unsigned n_free0;
    static const unsigned n_free1;
    static const unsigned n_contract;
    static const unsigned n_inplace;

    // Resource reuse info
    static const unsigned io_type;
    static const unsigned reuse_factor;

    template <class x_T, class y_T> using product = nnet::product::mult<x_T, y_T>;
};

#define PADDING_TYPES 1
#if PADDING_TYPES
// Padding Helper


// ac_fixed<res_T::width, res_T::i_width, res_T::sign, res_T::q_mode, AC_SAT> min_fx =
//  value<AC_VAL_MIN>(ac_fixed<res_T::width, res_T::i_width, res_T::sign, res_T::q_mode, AC_SAT>());
// auto int_part = my_var.slc<res_T::i_width>(res_T::width - res_T::i_width);
// auto frac_part = my_var.slc<res_T::width - res_T::i_width>(0);
#endif

using namespace sycl::ext::oneapi::experimental;

template <typename data_T, typename CONFIG_T> struct CausalState {
    static constexpr unsigned BRAM_SIZE = CONFIG_T::n_ctx * CONFIG_T::n_inplace * CONFIG_T::n_contract * CONFIG_T::n_free1;

    struct State {
        //[[intel::numbanks(CONFIG_T::n_ctx), intel::bankwidth(sizeof(data_T))]]
        [[intel::singlepump]] data_T causal_buff[BRAM_SIZE];
        unsigned write_ptrs[CONFIG_T::n_inplace];
        unsigned ctx_cts[CONFIG_T::n_inplace];
    };
};

template <typename data_T, typename CONFIG_T>
device_global<typename CausalState<data_T, CONFIG_T>::State, decltype(properties(device_image_scope, host_access_none))>
    causal_state;

// reads a contraction length unit from stream used for datas of dimension > 2
// only works with per-I streamed data though, careful about what you stream
template <class data_T, class data_buf_T, typename CONFIG_T>
#ifdef AUTOREG
bool
#else
void 
#endif
read_causal_pipe(data_buf_T buff, unsigned i, unsigned *write_ptrs, unsigned *ctx_cts, data_T *causal_buffer) {

    constexpr std::size_t CAUSAL_PIPE_SIZE = std::tuple_size<data_buf_T>::value;
    constexpr unsigned C = CONFIG_T::n_contract;
    constexpr unsigned I = CONFIG_T::n_inplace;
    constexpr unsigned L1 = CONFIG_T::n_free1;
    constexpr unsigned CTX = CONFIG_T::n_ctx;

    if (!CONFIG_T::contract_dim) {
        if (CAUSAL_PIPE_SIZE == C && L1 == 1) { // case where we have a dot product so stream is not 1xL1 but 1xC instead
            //#pragma unroll
            for (unsigned l1 = 0; l1 < L1; l1++) {
                // assumes stream is vector by vector (1xC each time)
                for (unsigned c = 0; c < C; c++) {
                    #ifdef AUTOREG
                    causal_buffer[L1 * C * I * write_ptrs[i] + C * L1 * i + C * l1 + c] = buff.data[c];
                    #else
                    causal_buffer[L1 * C * I * write_ptrs[i] + C * L1 * i + C * l1 + c] = buff[c];
                    #endif
                }
            }
        } else {
            for (unsigned c = 0; c < C; c++) {
                // assumes stream is vector by vector (1xL1 each time)
                //#pragma unroll
                for (unsigned l1 = 0; l1 < L1; l1++) {
                    #ifdef AUTOREG
                    causal_buffer[L1 * C * I * write_ptrs[i] + C * L1 * i + C * l1 + c] = buff.data[l1];
                    #else
                    causal_buffer[L1 * C * I * write_ptrs[i] + C * L1 * i + C * l1 + c] = buff[l1];
                    #endif
                }
            }
        }
    } else {
        //#pragma unroll
        for (unsigned l1 = 0; l1 < L1; l1++) {
            #ifdef AUTOREG
            causal_buffer[L1 * I * write_ptrs[i] + L1 * i + l1] = buff.data[l1];
            #else
            causal_buffer[L1 * I * write_ptrs[i] + L1 * i + l1] = buff[l1];
            #endif
        }
    }
    ctx_cts[i] = (ctx_cts[i] + 1 < CTX) ? (ctx_cts[i] + 1) : CTX;
    write_ptrs[i] = (write_ptrs[i] + 1 >= CTX) ? (write_ptrs[i] + 1 - CTX) : (write_ptrs[i] + 1);
    
    #ifdef AUTOREG
    return buff.exit_task;
    #endif
}

// read specific to contraction along the context
template <class data_T, class data_buf_T, typename CONFIG_T>
void read_causal_pipe_ctx(data_buf_T &data_buff, unsigned i, unsigned *write_ptrs, unsigned *ctx_cts,
                          data_T *causal_buffer) {

    constexpr std::size_t CAUSAL_PIPE_SIZE = std::tuple_size<data_buf_T>::value;

    constexpr unsigned C = CONFIG_T::n_contract;
    constexpr unsigned I = CONFIG_T::n_inplace;
    constexpr unsigned L1 = CONFIG_T::n_free1;
    constexpr unsigned CTX = CONFIG_T::n_ctx;

    unsigned wptr = (write_ptrs[i] + CTX - 1) % CTX;
    unsigned offset = L1 * I * wptr + L1 * i;
    #pragma unroll
    for (unsigned l1 = 0; l1 < L1; l1++) {
        causal_buffer[offset + l1] = data_buff[l1];
    }

    // ctx_cts[i] = std::min(ctx_cts[i] + 1, CTX);
    // write_ptrs[i] = (write_ptrs[i] + 1) % CTX;
}

// reads a contraction length unit from stream used for datas of dimension > 2
// only works with per-I streamed data though, careful about what you stream
template <class data_T, class data_arr_T, typename CONFIG_T>
#ifdef AUTOREG
bool
#else
void 
#endif
read_stateless_pipe(data_arr_T buff, data_T data_vect_buffer[(CONFIG_T::contract_dim ? CONFIG_T::n_free0 : CONFIG_T::n_contract)]) {

    constexpr unsigned C = CONFIG_T::n_contract;
    constexpr unsigned L0 = CONFIG_T::n_free0;

    // assumes stream is vector by vector (1xC each time)

    if (!CONFIG_T::contract_dim) {
        #pragma unroll
        for (unsigned c = 0; c < C; c++) {
#ifdef AUTOREG
            data_vect_buffer[c] = buff.data[c];
#else
            data_vect_buffer[c] = buff[c];
#endif
        }
    } else {
        #pragma unroll
        for (unsigned l0 = 0; l0 < L0; l0++) {
#ifdef AUTOREG
            data_vect_buffer[l0] = buff.data[l0];
#else
            data_vect_buffer[l0] = buff[l0];
#endif
        }
    }
#ifdef AUTOREG
    return buff.exit_task;
#endif
}


// THIS ASSUMES DATA ARRIVES IN {STATELESS,CAUSAL} FASHION
template <class data0_pipe, class data1_pipe, class res_pipe, typename CONFIG_T> void causal_einsum() {

#ifdef AUTOREG
    using data0_pipe_T = typename ExtractPipeType<data0_pipe>::value_type;
    using data1_pipe_T = typename ExtractPipeType<data1_pipe>::value_type;
    using res_pipe_T = typename ExtractPipeType<res_pipe>::value_type;
    using data0_buf_T = typename data0_pipe_T::data_type;
    using data1_buf_T = typename data1_pipe_T::data_type;
    using res_buf_T = typename res_pipe_T::data_type;
#else
    using data0_buf_T = typename ExtractPipeType<data0_pipe>::value_type;
    using data1_buf_T = typename ExtractPipeType<data1_pipe>::value_type;
    using res_buf_T = typename ExtractPipeType<res_pipe>::value_type;
#endif    
    using data0_T = typename data0_buf_T::value_type;
    using data1_T = typename data1_buf_T::value_type;
    using res_T = typename res_buf_T::value_type;

    constexpr unsigned L0 = CONFIG_T::n_free0;
    constexpr unsigned L1 = CONFIG_T::n_free1;
    constexpr unsigned C = CONFIG_T::n_contract;
    constexpr unsigned I = CONFIG_T::n_inplace;
    constexpr unsigned CTX = CONFIG_T::n_ctx;

    constexpr unsigned accum_T_accum_bits = (CONFIG_T::contract_dim) ? ceil_log2(CTX) : ceil_log2(C);
    constexpr unsigned accum_T_width = data0_T::width + data1_T::width + accum_T_accum_bits;
    constexpr unsigned accum_T_integer = data0_T::i_width + data1_T::i_width + accum_T_accum_bits;
    using accum_T = ac_fixed<accum_T_width, accum_T_integer, res_T::sign>;

    // initialise the buffers to read into
    [[intel::fpga_register]] data0_T data_vect_buffer[CONFIG_T::contract_dim ? L0 : C];
    [[intel::fpga_register]] res_pipe_T res_pipe_buffer;
    [[intel::fpga_register]] res_buf_T res_buffer;

    //######## REQUIRED AS GLOBAL PER LAYER NOT PER FUNC CALL ########
    auto &state = causal_state<data1_T, CONFIG_T>.get();
    auto &causal_buff = state.causal_buff;
    auto &write_ptrs = state.write_ptrs;
    auto &ctx_cts = state.ctx_cts;
    //################################################################

    // Lambda to clean BRAMs before exit
    auto clean_brams = [&](){
        for (unsigned i = 0; i < I; i++) {
            ctx_cts[i] = 0;
            write_ptrs[i] = 0;
        }
    };

#ifdef AUTOREG
    while (true) {
#else
    for (unsigned loop = 0; loop < CTX; loop++) {
#endif

        // COMBINE THIS WITH TILED APPROACH FOR A SPEEDUP
        #pragma unroll 4
        for (unsigned i = 0; i < I; i++) {

            if constexpr (!CONFIG_T::contract_dim) { // CONTRACT ALONG THE C DIMENSION

                for (unsigned l0 = 0; l0 < L0; l0++) {

                #ifdef AUTOREG
                    if (read_stateless_pipe<data0_T, data0_buf_T, CONFIG_T>(data0_pipe::read(),data_vect_buffer)){
                        // Since both pipes work simultaneously, drain both before exiting.
                        data1_pipe::read();
                        res_pipe_buffer.exit_task = true;
                        res_pipe::write(res_pipe_buffer);
                        clean_brams();
                        return;
                    }
                #else
                    read_stateless_pipe<data0_T, data0_buf_T, CONFIG_T>(data0_pipe::read(),data_vect_buffer);
                #endif

                    if (l0 == 0){
                    #ifdef AUTOREG
                        if (read_causal_pipe<data1_T, data1_buf_T, CONFIG_T>(data1_pipe::read(), i, write_ptrs, ctx_cts, causal_buff)){
                            res_buffer.exit_task = true;
                            res_pipe::write(res_buffer);
                            clean_brams();
                            return;
                        }
                    #else
                        read_causal_pipe<data1_T, data1_pipe, CONFIG_T>(i, write_ptrs, ctx_cts, causal_buff);
                    #endif
                        
                    }
                      

                    unsigned offset_ctx = (ctx_cts[i] == CTX) ? write_ptrs[i] : 0;

                    for (unsigned ctx = 0; ctx < CTX; ctx++) {

                        const unsigned ctx_offset = L1 * C * I * ((offset_ctx + ctx) % CTX);
                        const unsigned ctx_buff_offset = L1 * ctx;

                        if (ctx < ctx_cts[i]) {
                            #pragma unroll 4
                            for (unsigned l1 = 0; l1 < L1; l1++) {
                                accum_T tmp = 0;

                                //#pragma unroll 4
                                for (unsigned c = 0; c < C; c++) {
                                    tmp += data_vect_buffer[c] * causal_buff[ctx_offset + C * L1 * i + C * l1 + c];
                                }

                                tmp /= CONFIG_T::sqrt_dk;
                            #ifdef AUTOREG
                                res_buffer.data[ctx_buff_offset + l1] = static_cast<res_T>(tmp);
                            #else
                                res_buffer[ctx_buff_offset + l1] = static_cast<res_T>(tmp);
                            #endif
                            }
                        } else {
                            #pragma unroll
                            for (unsigned l1 = 0; l1 < L1; l1++) {
                            #ifdef AUTOREG
                                res_buffer.data[ctx_buff_offset + l1] = minval<res_T>();
                            #else
                                res_buffer[ctx_buff_offset + l1] = minval<res_T>();
                            #endif  
                            }
                        }
                    }
                #ifdef AUTOREG
                    res_buffer.exit_task = false;
                #endif
                    res_pipe::write(res_buffer);
                }

            } else { // CONTRACT ALONG THE CONTEXT - In this mode L0 == CTX and C is irrelevant

            #ifdef AUTOREG
                if (read_stateless_pipe<data0_T, data0_buf_T, CONFIG_T>(data0_pipe::read(), data_vect_buffer)){
                    // Empty the other pipe before exiting
                    data1_pipe::read();
                    res_buffer.exit_task = true;
                    res_pipe::write(res_buffer);
                    clean_brams();
                    return;
                }
            #else
                read_stateless_pipe<data0_T, data0_buf_T, CONFIG_T>(data0_pipe::read(), data_vect_buffer);
            #endif
                
            #ifdef AUTOREG
                data1_pipe_T causal_pipe = data1_pipe::read();
                if (causal_pipe.exit_task){
                    res_buffer.exit_task = true;
                    res_pipe::write(res_buffer);
                    clean_brams();
                    return;
                }
                [[intel::fpga_register]] data1_buf_T causal_data = causal_pipe.data;
            #else
                [[intel::fpga_register]] data1_buf_T causal_data = data1_pipe::read();
            #endif
                [[intel::fpga_register]] data1_buf_T causal_data_operate = causal_data;
                
                // Pre-calculate pointers to prevent r/w memory dependency
                ctx_cts[i] = (ctx_cts[i] + 1 < CTX) ? (ctx_cts[i] + 1) : CTX;
                write_ptrs[i] = (write_ptrs[i] + 1 >= CTX) ? (write_ptrs[i] + 1 - CTX) : (write_ptrs[i] + 1);
                unsigned offset_ctx = (ctx_cts[i] == CTX) ? write_ptrs[i] : 0;

                #pragma unroll 4 //TODO - Tune unrolls 
                for (unsigned l1 = 0; l1 < L1; l1++) {
                    accum_T tmp = 0;
                    tmp = (CTX - 1 < ctx_cts[i]) ? (data_vect_buffer[CTX - 1] * causal_data_operate[l1])
                                                 : (data_vect_buffer[ctx_cts[i] - 1] * causal_data_operate[l1]);
                    for (unsigned ctx = 0; ctx < CTX - 1; ctx++) {
                        if (ctx < ctx_cts[i] - 1) {

                            unsigned idx = offset_ctx + ctx;
                            unsigned offset_to_buffer = (idx >= CTX) ? idx - CTX : idx;
                            tmp += data_vect_buffer[ctx] * causal_buff[L1 * I * offset_to_buffer + L1 * i + l1];
                        }
                    }
                    #ifdef AUTOREG
                        res_buffer.data[l1] = static_cast<res_T>(tmp);
                    #else
                        res_buffer[l1] = static_cast<res_T>(tmp);
                    #endif
                }
                #ifdef AUTOREG
                    res_buffer.exit_task = false;
                #endif
                res_pipe::write(res_buffer);
                // Pointers are updated externally so this func does not touch ctx_cts or write_ptrs
                read_causal_pipe_ctx<data1_T, data1_buf_T, CONFIG_T>(causal_data, i, write_ptrs, ctx_cts,
                                                                                 causal_buff);
            }
        }
    }

    // Clean BRAMs before exit
    clean_brams();
}

} // namespace nnet

#endif
