#ifndef NNET_TRANSPOSE_STREAM_H_
#define NNET_TRANSPOSE_STREAM_H_

namespace nnet {

template <class data_pipe, class res_pipe, typename CONFIG_T> void transpose_stream() {

#ifdef AUTOREG
    using data_pipe_T = typename ExtractPipeType<data_pipe>::value_type;
    using res_pipe_T = typename ExtractPipeType<res_pipe>::value_type;
    using data_T = typename data_pipe_T::data_type;
    using res_T = typename res_pipe_T::data_type;

    [[intel::fpga_register]] res_pipe_T out_data_pipe;
#else
    using data_T = typename ExtractPipeType<data_pipe>::value_type;
    using res_T = typename ExtractPipeType<res_pipe>::value_type;
#endif

    constexpr auto data_size = std::tuple_size<data_T>{};
    constexpr auto res_size = std::tuple_size<res_T>{};

#ifdef AUTOREG
    while (true) {
#endif

        [[intel::fpga_register]] typename data_T::value_type data_array[CONFIG_T::N];

        for (int i = 0; i < CONFIG_T::N / data_size; i++) {
#ifdef AUTOREG
            [[intel::fpga_register]] data_pipe_T in_data_pipe = data_pipe::read();
            bool fb = in_data_pipe.feedback;
            if (in_data_pipe.exit_task) {
                out_data_pipe.exit_task = true;
                res_pipe::write(out_data_pipe);
                return;
            }
            [[intel::fpga_register]] data_T in_data = in_data_pipe.data;
#else
        [[intel::fpga_register]] data_T in_data = data_pipe::read();
#endif

            #pragma unroll
            for (int j = 0; j < data_size; j++) {
                data_array[i * data_size + j] = typename data_T::value_type(in_data[j]);
            }
        }

        for (int i = 0; i < CONFIG_T::N / res_size; i++) {
            [[intel::fpga_register]] res_T out_data;

            #pragma unroll
            for (int j = 0; j < res_size; j++) {
                out_data[j] = typename res_T::value_type(data_array[transfer_idx<CONFIG_T>(i * res_size + j)]);
            }

#ifdef AUTOREG
            out_data_pipe.data = out_data;
            out_data_pipe.exit_task = false;
            out_data_pipe.feedback = fb;
            res_pipe::write(out_data_pipe);
#else
        res_pipe::write(out_data);
#endif
        }

#ifdef AUTOREG
    }
#endif
}

} // namespace nnet

#endif
