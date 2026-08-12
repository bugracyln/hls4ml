#ifndef NNET_PADDING_STREAM_H_
#define NNET_PADDING_STREAM_H_

namespace nnet {

template <class res_pipe, typename CONFIG_T> inline void fill_zero() {
#ifdef AUTOREG
    [[intel::fpga_register]] typename ExtractPipeType<res_pipe>::value_type res_part_pipe;
    [[intel::fpga_register]] typename ExtractPipeType<res_pipe>::value_type::data_type res_part;
#else
    [[intel::fpga_register]] typename ExtractPipeType<res_pipe>::value_type res_part;
#endif
    
    #pragma unroll
    for (int i = 0; i < CONFIG_T::n_chan; i++) {
        res_part[i] = 0;
    }
#ifdef AUTOREG
    res_part_pipe.data = res_part;
    res_part_pipe.exit_task = false;
    res_pipe::write(res_part_pipe);
#else
    res_pipe::write(res_part);
#endif
}

template <class data_pipe, class res_pipe, typename CONFIG_T> inline void fill_data(
    #ifdef AUTOREG
    bool &exit_task
    #endif
) {
    [[intel::fpga_register]] auto data_part = data_pipe::read();
    [[intel::fpga_register]] typename ExtractPipeType<res_pipe>::value_type res_part;

#ifdef AUTOREG
    if(data_part.exit_task){
        res_part.exit_task = true;
        exit_task = true;
        res_pipe::write(res_part);
        return;
    }
    res_part.feedback = data_part.feedback;
#endif

    #pragma unroll
    for (int i = 0; i < CONFIG_T::n_chan; i++) {
    #ifdef AUTOREG
        res_part.data[i] = data_part.data[i];
    #else
        res_part[i] = data_part[i];
    #endif
    }
    res_pipe::write(res_part);
}

template <class data_pipe, class res_pipe, typename CONFIG_T> void zeropad1d_cl_stream() {
#ifdef AUTOREG
 while (true){
    bool exit_task = 0;
#endif

    PadLeft:
        for (int i = 0; i < CONFIG_T::pad_left; i++) {
            fill_zero<res_pipe, CONFIG_T>();
        }

    CopyMain:
        for (int i = 0; i < CONFIG_T::in_width; i++) {
            fill_data<data_pipe, res_pipe, CONFIG_T>(
        #ifdef AUTOREG
            exit_task
        #endif
            );
        #ifdef AUTOREG
            if (exit_task) break;
        #endif
        }
    #ifdef AUTOREG
        if (exit_task) break;
    #endif

    PadRight:
        for (int i = 0; i < CONFIG_T::pad_right; i++) {
            fill_zero<res_pipe, CONFIG_T>();
        }
#ifdef AUTOREG
    }
#endif
}

template <class data_pipe, class res_pipe, typename CONFIG_T> void zeropad2d_cl_stream() {
#ifdef AUTOREG
    while (true) {
        bool exit_task = 0;
#endif

    PadTop:
        [[intel::loop_coalesce(2)]] for (int i = 0; i < CONFIG_T::pad_top; i++) {
        PadTopWidth:
            for (int j = 0; j < CONFIG_T::out_width; j++) {
                fill_zero<res_pipe, CONFIG_T>();
            }
        }

    PadMain:
        [[intel::loop_coalesce(2)]] for (int i = 0; i < CONFIG_T::in_height; i++) {

        PadLeft:
            for (int j = 0; j < CONFIG_T::pad_left; j++) {
                fill_zero<res_pipe, CONFIG_T>();
            }

        CopyMain:
            for (int j = 0; j < CONFIG_T::in_width; j++) {
                fill_data<data_pipe, res_pipe, CONFIG_T>(
            #ifdef AUTOREG
                exit_task
            #endif
                );
            #ifdef AUTOREG
                if (exit_task) break;
            #endif
            }
        #ifdef AUTOREG
            if (exit_task) break;
        #endif

        PadRight:
            for (int j = 0; j < CONFIG_T::pad_right; j++) {
                fill_zero<res_pipe, CONFIG_T>();
            }
        }
    #ifdef AUTOREG
        if (exit_task) break;
    #endif

    PadBottom:
        for (int i = 0; i < CONFIG_T::pad_bottom; i++) {
        PadBottomWidth:
            for (int j = 0; j < CONFIG_T::out_width; j++) {
                fill_zero<res_pipe, CONFIG_T>();
            }
        }
#ifdef AUTOREG
    }
#endif
}

} // namespace nnet

#endif
