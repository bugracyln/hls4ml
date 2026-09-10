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

using namespace sycl::ext::oneapi::experimental;

template <typename data_T, typename CONFIG_T> struct CausalState {
    static constexpr unsigned BRAM_SIZE = CONFIG_T::n_ctx * CONFIG_T::n_inplace * CONFIG_T::n_contract * CONFIG_T::n_free1;
    static constexpr unsigned N_BANKS = CONFIG_T::contract_dim ? CONFIG_T::n_free1 : CONFIG_T::n_contract;
    static constexpr unsigned BANK_ELEMS = BRAM_SIZE / N_BANKS;

    // One banked row. The lowest dimension is the bank-select index:
    // c when contracting over C, l1 when contracting over the context.
    using row_T = data_T[N_BANKS];

    [[intel::singlepump]] data_T dg_causal_buff[BANK_ELEMS][N_BANKS];
};

/*
template <typename data_T, typename CONFIG_T> struct CausalWritePtrs {
    [[intel::singlepump]] unsigned dg_write_ptrs[CONFIG_T::n_inplace];
};

template <typename data_T, typename CONFIG_T> struct CausalCtxCts {
    [[intel::singlepump]] unsigned dg_ctx_cts[CONFIG_T::n_inplace];
};
*/

template <typename data_T, typename CONFIG_T>
device_global<CausalState<data_T, CONFIG_T>, decltype(properties(device_image_scope, host_access_none))> causal_state_;

/*
template <typename data_T, typename CONFIG_T>
device_global<CausalWritePtrs<data_T, CONFIG_T>,
    decltype(properties(device_image_scope, host_access_none))> write_ptrs_;

template <typename data_T, typename CONFIG_T>
device_global<CausalCtxCts<data_T, CONFIG_T>,
    decltype(properties(device_image_scope, host_access_none))> ctx_cts_;
*/

// reads a contraction length unit from stream used for datas of dimension > 2
// only works with per-I streamed data though, careful about what you stream
template <class data_T, class data_buf_T, typename CONFIG_T>
void read_causal_pipe(data_buf_T buff, unsigned i, unsigned write_ptrs,
                      typename CausalState<data_T, CONFIG_T>::row_T *causal_buffer) {

    constexpr unsigned CAUSAL_PIPE_SIZE = std::tuple_size<data_buf_T>::value;
    constexpr unsigned C = CONFIG_T::n_contract;
    constexpr unsigned I = CONFIG_T::n_inplace;
    constexpr unsigned L1 = CONFIG_T::n_free1;
    constexpr unsigned CTX = CONFIG_T::n_ctx;

    const unsigned wptr = (write_ptrs == 0) ? (CTX - 1) : (write_ptrs - 1);

    if constexpr (!CONFIG_T::contract_dim) {
        static_assert(CAUSAL_PIPE_SIZE == C && L1 == 1, "!contract_dim path assumes the 1xC dot-product stream layout");

        unsigned offset = L1 * I * wptr + L1 * i;

        // flat index was L1*C*I*wptr + C*L1*i + C*l1 + c, so row = that / C and bank = c
        if (CAUSAL_PIPE_SIZE == C && L1 == 1) { // case where we have a dot product so stream is not 1xL1 but 1xC instead

            for (unsigned l1 = 0; l1 < L1; l1++) {
                // assumes stream is vector by vector (1xC each time)
                #pragma unroll
                for (unsigned c = 0; c < C; c++) {
                    causal_buffer[offset + l1][c] = buff[c];
                }
            }
        } else {
            for (unsigned c = 0; c < C; c++) {
                // assumes stream is vector by vector (1xL1 each time)
                //#pragma unroll
                for (unsigned l1 = 0; l1 < L1; l1++) {
                    causal_buffer[offset + l1][c] = buff[l1];
                }
            }
        }
    } else { // DEAD PATH - DELETE
        // flat index was L1*I*wptr + L1*i + l1, so row = that / L1 and bank = l1
        #pragma unroll
        for (unsigned l1 = 0; l1 < L1; l1++) {
            causal_buffer[I * wptr + i][l1] = buff[l1];
        }
    }
    // ctx_cts[i] = (ctx_cts[i] + 1 < CTX) ? (ctx_cts[i] + 1) : CTX;
    // write_ptrs[i] = (write_ptrs[i] + 1 >= CTX) ? (write_ptrs[i] + 1 - CTX) : (write_ptrs[i] + 1);
}

// read specific to contraction along the context
template <class data_T, class data_buf_T, typename CONFIG_T>
void read_causal_pipe_ctx(data_buf_T &data_buff, unsigned i, unsigned write_ptrs,
                          typename CausalState<data_T, CONFIG_T>::row_T *causal_buffer) {

    constexpr unsigned I = CONFIG_T::n_inplace;
    constexpr unsigned L1 = CONFIG_T::n_free1;
    constexpr unsigned CTX = CONFIG_T::n_ctx;

    unsigned wptr = (write_ptrs == 0) ? (CTX - 1) : (write_ptrs - 1);
    unsigned offset = I * wptr + i;
    #pragma unroll
    for (unsigned l1 = 0; l1 < L1; l1++) {
        causal_buffer[offset][l1] = data_buff[l1];
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
read_stateless_pipe(data_arr_T buff, data_T data_vect_buffer[(CONFIG_T::contract_dim ? CONFIG_T::n_free0 : CONFIG_T::n_contract)]
#ifdef AUTOREG
    , bool& fb
#endif
) {

    constexpr unsigned C = CONFIG_T::n_contract;
    constexpr unsigned L0 = CONFIG_T::n_free0;

    // assumes stream is vector by vector (1xC each time)

    if constexpr (!CONFIG_T::contract_dim) {
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
    fb |= buff.feedback;
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
    constexpr unsigned TILE_SIZE = CONFIG_T::reuse_factor;

    constexpr unsigned accum_T_accum_bits = (CONFIG_T::contract_dim) ? ceil_log2(CTX) : ceil_log2(C);
    constexpr unsigned accum_T_width = data0_T::width + data1_T::width + accum_T_accum_bits;
    constexpr unsigned accum_T_integer = data0_T::i_width + data1_T::i_width + accum_T_accum_bits;
    using accum_T = ac_fixed<accum_T_width, accum_T_integer, res_T::sign>;

    // initialise the buffers to read into
    [[intel::fpga_register]] data0_T data_vect_buffer[CONFIG_T::contract_dim ? L0 : C];
#ifdef AUTOREG
    [[intel::fpga_register]] res_pipe_T res_buffer;
#else
    [[intel::fpga_register]] res_buf_T res_buffer;
#endif

    //######## REQUIRED AS GLOBAL PER LAYER NOT PER FUNC CALL ########
    auto &causal_buff = causal_state_<data1_T, CONFIG_T>.get().dg_causal_buff;
    // auto &write_ptrs = write_ptrs_<data1_T, CONFIG_T>.get().dg_write_ptrs;
    // auto &ctx_cts = ctx_cts_<data1_T, CONFIG_T>.get().dg_ctx_cts;
    //################################################################

    // Lambda to clean BRAMs before exit
    /*
    auto clean_brams = [&](){
        #pragma unroll
        for (unsigned i = 0; i < I; i++) {
            ctx_cts[i] = 0;
            write_ptrs[i] = 0;
        }
    };
    */

    // The pointers should only exist per invocation, not in between.
    unsigned ctx_cts = 0;
    unsigned write_ptrs = 0;

#ifdef AUTOREG
    bool exit_task = false;

    while (!exit_task) {
        bool fb = 0;
#else
    for (unsigned loop = 0; loop < CTX; loop++) {
#endif
        // Pre-calculate pointers to prevent r/w memory dependency
        ctx_cts = (ctx_cts + 1 < CTX) ? (ctx_cts + 1) : CTX;
        write_ptrs = (write_ptrs + 1 >= CTX) ? (write_ptrs + 1 - CTX) : (write_ptrs + 1);
        unsigned offset_ctx = (ctx_cts == CTX) ? write_ptrs : 0;

        // COMBINE THIS WITH TILED APPROACH FOR A SPEEDUP
        for (unsigned i = 0; i < I; i++) {

            if constexpr (!CONFIG_T::contract_dim) { // CONTRACT ALONG THE C DIMENSION

#ifdef AUTOREG
                [[intel::fpga_register]] data1_pipe_T causal_pipe = data1_pipe::read();
                [[intel::fpga_register]] data1_buf_T data_buff = causal_pipe.data;
                fb |= causal_pipe.feedback;
#else
                [[intel::fpga_register]] data1_buf_T data_buff = data1_pipe::read();
#endif

                for (unsigned l0 = 0; l0 < L0; l0++) {

#ifdef AUTOREG
                    res_buffer.exit_task =
                        read_stateless_pipe<data0_T, data0_pipe_T, CONFIG_T>(data0_pipe::read(), data_vect_buffer, fb);
                    res_buffer.exit_task |= causal_pipe.exit_task;
#else
                    read_stateless_pipe<data0_T, data0_buf_T, CONFIG_T>(data0_pipe::read(), data_vect_buffer);
#endif

                    for (unsigned ctx = 0; ctx < CTX; ctx++) {

                        // row offset only, the bank index is c
                        unsigned raw_ctx_offset = offset_ctx + ctx;
                        unsigned wrap_ctx_offset = (raw_ctx_offset >= CTX) ? (raw_ctx_offset - CTX) : raw_ctx_offset;
                        unsigned ctx_offset = L1 * I * wrap_ctx_offset;
                        unsigned ctx_buff_offset = L1 * ctx;

                        if (ctx < ctx_cts) {

                            bool last_wr = (ctx == ctx_cts - 1);

                            for (unsigned l1 = 0; l1 < L1; l1++) {

                                accum_T tmp = 0;

                                #pragma unroll
                                for (unsigned c = 0; c < C; c++) {
                                    tmp += data_vect_buffer[c] *
                                           (last_wr ? data_buff[c] : causal_buff[ctx_offset + L1 * i + l1][c]);
                                }
                                // tmp *= CONFIG_T::inv_sqrt_dk; Now softmax tables scale automatically

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
                    res_buffer.feedback = fb;
                    exit_task |= res_buffer.exit_task;
#endif
                    res_pipe::write(res_buffer);
                }
                read_causal_pipe<data1_T, data1_buf_T, CONFIG_T>(data_buff, i, write_ptrs, causal_buff);

            } else { // CONTRACT ALONG THE CONTEXT - In this mode L0 == CTX and C is irrelevant

#ifdef AUTOREG
                res_buffer.exit_task =
                    read_stateless_pipe<data0_T, data0_pipe_T, CONFIG_T>(data0_pipe::read(), data_vect_buffer, fb);
#else
                read_stateless_pipe<data0_T, data0_buf_T, CONFIG_T>(data0_pipe::read(), data_vect_buffer);
#endif

#ifdef AUTOREG
                data1_pipe_T causal_pipe = data1_pipe::read();
                fb |= causal_pipe.feedback;
                res_buffer.exit_task |= causal_pipe.exit_task;

                [[intel::fpga_register]] data1_buf_T causal_data = causal_pipe.data;
#else
                [[intel::fpga_register]] data1_buf_T causal_data = data1_pipe::read();
#endif
                [[intel::fpga_register]] data1_buf_T causal_data_operate = causal_data;

                [[intel::fpga_register]] accum_T acc[L1];

                unsigned vect_buffer_idx = (CTX - 1 < ctx_cts) ? (CTX - 1) : (ctx_cts - 1);
                #pragma unroll
                for (unsigned l1 = 0; l1 < L1; l1++)
                    acc[l1] = data_vect_buffer[vect_buffer_idx] * causal_data_operate[l1];

                for (unsigned ctx_tile = 0; ctx_tile < CTX; ctx_tile += TILE_SIZE) {

                    [[intel::fpga_register]] accum_T partial[L1][TILE_SIZE];

                    #pragma unroll
                    for (unsigned tile_offset = 0; tile_offset < TILE_SIZE; tile_offset++) {

                        unsigned ctx = ctx_tile + tile_offset;
                        unsigned causal_buff_ctx_offet = (ctx < CTX) ? (I * ctx) : 0;

                        unsigned offset_from_earliest_val =
                            (ctx < CTX && (ctx >= offset_ctx)) ? (ctx - offset_ctx) : (ctx + CTX - offset_ctx);
                        data0_T data_vect =
                            (offset_from_earliest_val < vect_buffer_idx) ? data_vect_buffer[offset_from_earliest_val] : 0;

                        #pragma unroll
                        for (unsigned l1 = 0; l1 < L1; l1++) {
                            // row offset only, the bank index is l1
                            partial[l1][tile_offset] = data_vect * causal_buff[causal_buff_ctx_offet + i][l1];
                        }
                    }

                    Op_add<accum_T> op_add;
                    #pragma unroll
                    for (unsigned l1 = 0; l1 < L1; l1++) {
                        acc[l1] += reduce<accum_T, TILE_SIZE, Op_add<accum_T>>(partial[l1], op_add);
                    }
                }

                #pragma unroll
                for (unsigned l1 = 0; l1 < L1; l1++) {
#ifdef AUTOREG
                    res_buffer.data[l1] = static_cast<res_T>(acc[l1]);
#else
                    res_buffer[l1] = static_cast<res_T>(acc[l1]);
#endif
                }

#ifdef AUTOREG
                res_buffer.feedback = fb;
                exit_task |= res_buffer.exit_task;
#endif
                res_pipe::write(res_buffer);

                // Pointers are updated externally so this func does not touch ctx_cts or write_ptrs
                read_causal_pipe_ctx<data1_T, data1_buf_T, CONFIG_T>(causal_data, i, write_ptrs, causal_buff);
            }
#ifdef AUTOREG
            if (exit_task)
                break;
#endif
        }
    }
#ifdef AUTOREG
    if constexpr (CONFIG_T::contract_dim) {
        #pragma unroll
        for (int h = 0; h < I - 1; h++)
            res_pipe::write(res_buffer);
    }
#endif
    // Clean BRAMs before exit
    // clean_brams();
}

} // namespace nnet

#endif
