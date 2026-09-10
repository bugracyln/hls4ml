#ifndef NNET_RECURRENT_STREAM_H_
#define NNET_RECURRENT_STREAM_H_

#include "nnet_common.h"
#include "nnet_dense.h"
#include "nnet_recurrent_activation.h"

namespace nnet {
template <class data_pipe, class res_pipe, typename CONFIG_T>
void gru_stream(typename CONFIG_T::weight_t weights, typename CONFIG_T::recurrent_weight_t recurrent_weights,
                typename CONFIG_T::bias_t bias, typename CONFIG_T::recurrent_bias_t recurrent_bias) {

#ifdef AUTOREG
    using data_pipe_T = typename ExtractPipeType<data_pipe>::value_type;
    using res_pipe_T = typename ExtractPipeType<res_pipe>::value_type;
    using data_T = typename data_pipe_T::data_type;
    using res_T = typename res_pipe_T::data_type;

    [[intel::fpga_register]] res_pipe_T res_pack_pipe;
#else
    using data_T = typename ExtractPipeType<data_pipe>::value_type;
    using res_T = typename ExtractPipeType<res_pipe>::value_type;
#endif

    using h_T = array<typename res_T::value_type, CONFIG_T::n_units>;

    constexpr auto datasize = std::tuple_size<data_T>{};
    constexpr auto ressize = std::tuple_size<res_T>{};

    [[intel::fpga_register]] h_T h;
    #pragma unroll
    for (int i = 0; i < CONFIG_T::n_units; i++) {
        h[i] = 0;
    }

    [[intel::fpga_register]] data_T x;

DataPropagation:
#ifdef AUTOREG
    while (true) {
        auto data_pack_pipe = data_pipe::read();
        if (data_pack_pipe.exit_task) {
            // If we are set to produce the final output, do so just before exiting. Note that
            // it is fine to write two elements since all units wait for EOS in autoreg so there
            // will not be a case where where we do 2 writes but only read one on the downstream
            // task_sequence.
            if (!CONFIG_T::return_sequences) {
                res_T res_pack;
                #pragma unroll
                for (int i_pack = 0; i_pack < ressize; i_pack++)
                    res_pack[i_pack] = h[i_pack];
                res_pack_pipe.data = res_pack;
                res_pack_pipe.exit_task = false;
                res_pipe::write(res_pack_pipe);
            }
            res_pack_pipe.exit_task = true;
            res_pipe::write(res_pack_pipe);
            return;
        }
        auto data_pack = data_pack_pipe.data;
        bool fb = data_pack_pipe.feedback;
#else
    for (int i_in = 0; i_in < CONFIG_T::n_timesteps * CONFIG_T::n_in / datasize; i_in++) {
        auto data_pack = data_pipe::read();
#endif

    DataPack:
        #pragma unroll
        for (int i_pack = 0; i_pack < datasize; i_pack++) {
            x[i_pack] = data_pack[i_pack];
        }

        nnet::gru_cell<data_T, h_T, CONFIG_T>(x, h, weights, recurrent_weights, bias, recurrent_bias);

        if (CONFIG_T::return_sequences) {
            res_T res_pack;

        ResPackRetSeq:
            #pragma unroll
            for (int i_pack = 0; i_pack < ressize; i_pack++) {
                res_pack[i_pack] = h[i_pack];
            }
#ifdef AUTOREG
            res_pack_pipe.data = res_pack;
            res_pack_pipe.exit_task = false;
            res_pack_pipe.feedback = data_pack_pipe.feedback;
            res_pipe::write(res_pack_pipe);
#else
            res_pipe::write(res_pack);
#endif
        }
    }

    // In autoreg we return when we detect EOS instead of here
    // so this path is only releavant to normal application
#ifndef AUTOREG
    if (!CONFIG_T::return_sequences) {
        res_T res_pack;

    ResPackNoRetSeq:
        #pragma unroll
        for (int i_pack = 0; i_pack < ressize; i_pack++) {
            res_pack[i_pack] = h[i_pack];
        }
#ifdef AUTOREG
        res_pack_pipe.data = res_pack;
        res_pack_pipe.exit_task = false;
        res_pack_pipe.feedback = false;
        res_pipe::write(res_pack_pipe);
#else
        res_pipe::write(res_pack);
#endif
    }
#endif
}

} // namespace nnet

#endif
