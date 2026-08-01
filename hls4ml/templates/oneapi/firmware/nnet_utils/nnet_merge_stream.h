#ifndef NNET_MERGE_STREAM_H_
#define NNET_MERGE_STREAM_H_

namespace nnet {

template <class input1_pipe, class input2_pipe, class res_pipe, typename CONFIG_T> void add_stream() {

    // both inputs are the same size
#ifdef AUTOREG
    constexpr auto inputSize = std::tuple_size<typename ExtractPipeType<input1_pipe>::value_type::data_type>{};
    constexpr auto outputSize = std::tuple_size<typename ExtractPipeType<res_pipe>::value_type::data_type>{};
    using res_pipe_unit_T = typename ExtractPipeType<res_pipe>::value_type::data_type::value_type;
#else
    constexpr auto inputSize = std::tuple_size<typename ExtractPipeType<input1_pipe>::value_type>{};
    constexpr auto outputSize = std::tuple_size<typename ExtractPipeType<res_pipe>::value_type>{};
    using res_pipe_unit_T = typename ExtractPipeType<res_pipe>::value_type::value_type;
#endif

AddLoop:
    #ifdef AUTOREG
    while (true){
        [[intel::fpga_register]] auto in_data_pipe1 = input1_pipe::read();
        [[intel::fpga_register]] auto in_data_pipe2 = input2_pipe::read();
        [[intel::fpga_register]] auto in_data1 = in_data_pipe1.data;
        [[intel::fpga_register]] auto in_data2 = in_data_pipe2.data;
        [[intel::fpga_register]] typename ExtractPipeType<res_pipe>::value_type out_data_pipe;
        [[intel::fpga_register]] typename ExtractPipeType<res_pipe>::value_type::data_type out_data;

        if (in_data_pipe1.exit_task || in_data_pipe2.exit_task){
            out_data_pipe.exit_task = true;
            res_pipe::write(out_data_pipe);
            return;
        }
    #else
    [[intel::initiation_interval(1)]] for (int i = 0; i < CONFIG_T::n_elem / inputSize; i++) {

        [[intel::fpga_register]] auto in_data1 = input1_pipe::read();
        [[intel::fpga_register]] auto in_data2 = input2_pipe::read();
        [[intel::fpga_register]] typename ExtractPipeType<res_pipe>::value_type out_data;
    #endif

    AddPack:
        #pragma unroll
        for (int j = 0; j < outputSize; j++) {
            out_data[j] = static_cast<res_pipe_unit_T>(in_data1[j] + in_data2[j]);
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

template <class input1_pipe, class input2_pipe, class res_pipe, typename CONFIG_T> void subtract_stream() {
    // both inputs are the same size
#ifdef AUTOREG
    constexpr auto inputSize = std::tuple_size<typename ExtractPipeType<input1_pipe>::value_type::data_type>{};
    constexpr auto outputSize = std::tuple_size<typename ExtractPipeType<res_pipe>::value_type::data_type>{};
    using res_pipe_unit_T = typename ExtractPipeType<res_pipe>::value_type::data_type::value_type;
#else
    constexpr auto inputSize = std::tuple_size<typename ExtractPipeType<input1_pipe>::value_type>{};
    constexpr auto outputSize = std::tuple_size<typename ExtractPipeType<res_pipe>::value_type>{};
    using res_pipe_unit_T = typename ExtractPipeType<res_pipe>::value_type::value_type;
#endif

SubtractLoop:
#ifdef AUTOREG
    while (true){
        [[intel::fpga_register]] auto in_data_pipe1 = input1_pipe::read();
        [[intel::fpga_register]] auto in_data_pipe2 = input2_pipe::read();
        [[intel::fpga_register]] auto in_data1 = in_data_pipe1.data;
        [[intel::fpga_register]] auto in_data2 = in_data_pipe2.data;
        [[intel::fpga_register]] typename ExtractPipeType<res_pipe>::value_type out_data_pipe;
        [[intel::fpga_register]] typename ExtractPipeType<res_pipe>::value_type::data_type out_data;

        if (in_data_pipe1.exit_task ||in_data_pipe2.exit_task){
            out_data_pipe.exit_task = true;
            res_pipe::write(out_data_pipe);
            break;
        }
    #else
    [[intel::initiation_interval(1)]] for (int i = 0; i < CONFIG_T::n_elem / inputSize; i++) {

        [[intel::fpga_register]] auto in_data1 = input1_pipe::read();
        [[intel::fpga_register]] auto in_data2 = input2_pipe::read();
        [[intel::fpga_register]] typename ExtractPipeType<res_pipe>::value_type out_data;
    #endif

    SubtractPack:
        #pragma unroll
        for (int j = 0; j < outputSize; j++) {
            out_data[j] = static_cast<res_pipe_unit_T>(in_data1[j] - in_data2[j]);
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

template <class input1_pipe, class input2_pipe, class res_pipe, typename CONFIG_T> void multiply_stream() {
    // both inputs are the same size
#ifdef AUTOREG
    constexpr auto inputSize = std::tuple_size<typename ExtractPipeType<input1_pipe>::value_type::data_type>{};
    constexpr auto outputSize = std::tuple_size<typename ExtractPipeType<res_pipe>::value_type::data_type>{};
    using res_pipe_unit_T = typename ExtractPipeType<res_pipe>::value_type::data_type::value_type;
#else
    constexpr auto inputSize = std::tuple_size<typename ExtractPipeType<input1_pipe>::value_type>{};
    constexpr auto outputSize = std::tuple_size<typename ExtractPipeType<res_pipe>::value_type>{};
    using res_pipe_unit_T = typename ExtractPipeType<res_pipe>::value_type::value_type;
#endif

MultLoop:
#ifdef AUTOREG
    while (true){
        [[intel::fpga_register]] auto in_data_pipe1 = input1_pipe::read();
        [[intel::fpga_register]] auto in_data_pipe2 = input2_pipe::read();
        [[intel::fpga_register]] auto in_data1 = in_data_pipe1.data;
        [[intel::fpga_register]] auto in_data2 = in_data_pipe2.data;
        [[intel::fpga_register]] typename ExtractPipeType<res_pipe>::value_type out_data_pipe;
        [[intel::fpga_register]] typename ExtractPipeType<res_pipe>::value_type::data_type out_data;

        if (in_data_pipe1.exit_task ||in_data_pipe2.exit_task){
            out_data_pipe.exit_task = true;
            res_pipe::write(out_data_pipe);
            break;
        }
    #else
    [[intel::initiation_interval(1)]] for (int i = 0; i < CONFIG_T::n_elem / inputSize; i++) {

        [[intel::fpga_register]] auto in_data1 = input1_pipe::read();
        [[intel::fpga_register]] auto in_data2 = input2_pipe::read();
        [[intel::fpga_register]] typename ExtractPipeType<res_pipe>::value_type out_data;
    #endif

    MultPack:
        #pragma unroll
        for (int j = 0; j < outputSize; j++) {
            out_data[j] = static_cast<res_pipe_unit_T>(in_data1[j] * in_data2[j]);
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

template <class input1_pipe, class input2_pipe, class res_pipe, typename CONFIG_T> void average_stream() {
    // both inputs are the same size
#ifdef AUTOREG
    constexpr auto inputSize = std::tuple_size<typename ExtractPipeType<input1_pipe>::value_type::data_type>{};
    constexpr auto outputSize = std::tuple_size<typename ExtractPipeType<res_pipe>::value_type::data_type>{};
    using res_pipe_unit_T = typename ExtractPipeType<res_pipe>::value_type::data_type::value_type;
#else
    constexpr auto inputSize = std::tuple_size<typename ExtractPipeType<input1_pipe>::value_type>{};
    constexpr auto outputSize = std::tuple_size<typename ExtractPipeType<res_pipe>::value_type>{};
    using res_pipe_unit_T = typename ExtractPipeType<res_pipe>::value_type::value_type;
#endif

AvgLoop:
#ifdef AUTOREG
    while (true){
        [[intel::fpga_register]] auto in_data_pipe1 = input1_pipe::read();
        [[intel::fpga_register]] auto in_data_pipe2 = input2_pipe::read();
        [[intel::fpga_register]] auto in_data1 = in_data_pipe1.data;
        [[intel::fpga_register]] auto in_data2 = in_data_pipe2.data;
        [[intel::fpga_register]] typename ExtractPipeType<res_pipe>::value_type out_data_pipe;
        [[intel::fpga_register]] typename ExtractPipeType<res_pipe>::value_type::data_type out_data;

        if (in_data_pipe1.exit_task ||in_data_pipe2.exit_task){
            out_data_pipe.exit_task = true;
            res_pipe::write(out_data_pipe);
            break;
        }
    #else
    [[intel::initiation_interval(1)]] for (int i = 0; i < CONFIG_T::n_elem / inputSize; i++) {

        [[intel::fpga_register]] auto in_data1 = input1_pipe::read();
        [[intel::fpga_register]] auto in_data2 = input2_pipe::read();
        [[intel::fpga_register]] typename ExtractPipeType<res_pipe>::value_type out_data;
    #endif

    AvgPack:
        #pragma unroll
        for (int j = 0; j < outputSize; j++) {
            out_data[j] = static_cast<res_pipe_unit_T>(
                (in_data1[j] + in_data2[j]) * ac_fixed<2, 1, false>(0.5));
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

template <class input1_pipe, class input2_pipe, class res_pipe, typename CONFIG_T> void maximum_stream() {
    // both inputs are the same size
#ifdef AUTOREG
    constexpr auto inputSize = std::tuple_size<typename ExtractPipeType<input1_pipe>::value_type::data_type>{};
    constexpr auto outputSize = std::tuple_size<typename ExtractPipeType<res_pipe>::value_type::data_type>{};
    using res_pipe_unit_T = typename ExtractPipeType<res_pipe>::value_type::data_type::value_type;
#else
    constexpr auto inputSize = std::tuple_size<typename ExtractPipeType<input1_pipe>::value_type>{};
    constexpr auto outputSize = std::tuple_size<typename ExtractPipeType<res_pipe>::value_type>{};
    using res_pipe_unit_T = typename ExtractPipeType<res_pipe>::value_type::value_type;
#endif

MaxLoop:
#ifdef AUTOREG
    while (true){
        [[intel::fpga_register]] auto in_data_pipe1 = input1_pipe::read();
        [[intel::fpga_register]] auto in_data_pipe2 = input2_pipe::read();
        [[intel::fpga_register]] auto in_data1 = in_data_pipe1.data;
        [[intel::fpga_register]] auto in_data2 = in_data_pipe2.data;
        [[intel::fpga_register]] typename ExtractPipeType<res_pipe>::value_type out_data_pipe;
        [[intel::fpga_register]] typename ExtractPipeType<res_pipe>::value_type::data_type out_data;

        if (in_data_pipe1.exit_task ||in_data_pipe2.exit_task){
            out_data_pipe.exit_task = true;
            res_pipe::write(out_data_pipe);
            break;
        }
    #else
    [[intel::initiation_interval(1)]] for (int i = 0; i < CONFIG_T::n_elem / inputSize; i++) {

        [[intel::fpga_register]] auto in_data1 = input1_pipe::read();
        [[intel::fpga_register]] auto in_data2 = input2_pipe::read();
        [[intel::fpga_register]] typename ExtractPipeType<res_pipe>::value_type out_data;
    #endif

    MaxPack:
        #pragma unroll
        for (int j = 0; j < outputSize; j++) {
            out_data[j] = (in_data1[j] > in_data2[j])
                              ? static_cast<res_pipe_unit_T>(in_data1[j])
                              : static_cast<res_pipe_unit_T>(in_data2[j]);
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

template <class input1_pipe, class input2_pipe, class res_pipe, typename CONFIG_T> void minimum_stream() {
    // both inputs are the same size
#ifdef AUTOREG
    constexpr auto inputSize = std::tuple_size<typename ExtractPipeType<input1_pipe>::value_type::data_type>{};
    constexpr auto outputSize = std::tuple_size<typename ExtractPipeType<res_pipe>::value_type::data_type>{};
    using res_pipe_unit_T = typename ExtractPipeType<res_pipe>::value_type::data_type::value_type;
#else
    constexpr auto inputSize = std::tuple_size<typename ExtractPipeType<input1_pipe>::value_type>{};
    constexpr auto outputSize = std::tuple_size<typename ExtractPipeType<res_pipe>::value_type>{};
    using res_pipe_unit_T = typename ExtractPipeType<res_pipe>::value_type::value_type;
#endif

MinLoop:
#ifdef AUTOREG
    while (true){
        [[intel::fpga_register]] auto in_data_pipe1 = input1_pipe::read();
        [[intel::fpga_register]] auto in_data_pipe2 = input2_pipe::read();
        [[intel::fpga_register]] auto in_data1 = in_data_pipe1.data;
        [[intel::fpga_register]] auto in_data2 = in_data_pipe2.data;
        [[intel::fpga_register]] typename ExtractPipeType<res_pipe>::value_type out_data_pipe;
        [[intel::fpga_register]] typename ExtractPipeType<res_pipe>::value_type::data_type out_data;

        if (in_data_pipe1.exit_task ||in_data_pipe2.exit_task){
            out_data_pipe.exit_task = true;
            res_pipe::write(out_data_pipe);
            break;
        }
    #else
    [[intel::initiation_interval(1)]] for (int i = 0; i < CONFIG_T::n_elem / inputSize; i++) {

        [[intel::fpga_register]] auto in_data1 = input1_pipe::read();
        [[intel::fpga_register]] auto in_data2 = input2_pipe::read();
        [[intel::fpga_register]] typename ExtractPipeType<res_pipe>::value_type out_data;
    #endif

    MinPack:
        #pragma unroll
        for (int j = 0; j < outputSize; j++) {
            out_data[j] = (in_data1[j] < in_data2[j])
                              ? static_cast<res_pipe_unit_T>(in_data1[j])
                              : static_cast<res_pipe_unit_T>(in_data2[j]);
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

template <class input1_pipe, class input2_pipe, class res_pipe, typename CONFIG_T> void concatenate1d_stream() {
#ifdef AUTOREG
    constexpr auto input1Size = std::tuple_size<typename ExtractPipeType<input1_pipe>::value_type::data_type>{};
    constexpr auto input2Size = std::tuple_size<typename ExtractPipeType<input2_pipe>::value_type::data_type>{};
    using res_pipe_unit_T = typename ExtractPipeType<res_pipe>::value_type::data_type::value_type;
    [[intel::fpga_register]] typename ExtractPipeType<res_pipe>::value_type out_data_pipe;
    [[intel::fpga_register]] typename ExtractPipeType<res_pipe>::value_type::data_type out_data;
    bool exit_task = 0;
#else
    constexpr auto input1Size = std::tuple_size<typename ExtractPipeType<input1_pipe>::value_type>{};
    constexpr auto input2Size = std::tuple_size<typename ExtractPipeType<input2_pipe>::value_type>{};
    using res_pipe_unit_T = typename ExtractPipeType<res_pipe>::value_type::value_type;
    [[intel::fpga_register]] typename ExtractPipeType<res_pipe>::value_type out_data;
#endif

#ifdef AUTOREG
    while (true) { 
        exit_task = 0;
#endif
    ConcatLoop1:
        [[intel::initiation_interval(1)]] for (int i = 0; i < CONFIG_T::n_elem1_0 / input1Size; i++) {
        #ifdef AUTOREG
            [[intel::fpga_register]] auto in_data1_pipe = input1_pipe::read();
            [[intel::fpga_register]] auto in_data1 = in_data1_pipe.data;
            if (in_data1_pipe.exit_task){
                exit_task = true;
                break;
            }
        #else
            [[intel::fpga_register]] auto in_data1 = input1_pipe::read();
        #endif
            
        ConcatPack1:
            #pragma unroll
            for (int j = 0; j < input1Size; j++) {
                out_data[j + (i * input1Size)] =
                    static_cast<res_pipe_unit_T>(in_data1[j]);
            }
        }

    ConcatLoop2:
        [[intel::initiation_interval(1)]] for (int i = 0; i < CONFIG_T::n_elem2_0 / input2Size; i++) {
        #ifdef AUTOREG
            [[intel::fpga_register]] auto in_data2_pipe = input2_pipe::read();
            [[intel::fpga_register]] auto in_data2 = in_data2_pipe.data;
            if (in_data2_pipe.exit_task){
                exit_task = true;
                break;
            }
        #else
            [[intel::fpga_register]] auto in_data2 = input2_pipe::read();
        #endif

        ConcatPack2:
            #pragma unroll
            for (int j = 0; j < input2Size; j++) {
                out_data[j + (i * input2Size) + (CONFIG_T::n_elem1_0)] =
                    static_cast<res_pipe_unit_T>(in_data2[j]);
            }
        }

    #ifdef AUTOREG
        // We check if there is exit task before writing into pipe so if stop signal is received we write outside the while loop
        // We dont have if(exit_task) break; between input 1 and 2 to ensure both pipes drain
        if(exit_task) break;
        out_data_pipe.data = out_data;
        out_data_pipe.exit_task = false;
        res_pipe::write(out_data_pipe);
    #else
        res_pipe::write(out_data);
    #endif
        
#ifdef AUTOREG
    }
    // Just before exit, pass the stop signal, data contained inside the pipe is irrelevant
    out_data_pipe.exit_task = true;
    res_pipe::write(out_data_pipe);
#endif 
}

template <class input1_pipe, class input2_pipe, class res_pipe, typename CONFIG_T> void concatenate2d_0_stream() {
#ifdef AUTOREG
    constexpr auto input1Size = std::tuple_size<typename ExtractPipeType<input1_pipe>::value_type::data_type>{};
    constexpr auto input2Size = std::tuple_size<typename ExtractPipeType<input2_pipe>::value_type::data_type>{};
    using res_pipe_unit_T = typename ExtractPipeType<res_pipe>::value_type::data_type::value_type;
    [[intel::fpga_register]] typename ExtractPipeType<res_pipe>::value_type out_data_pipe;
#else
    constexpr auto input1Size = std::tuple_size<typename ExtractPipeType<input1_pipe>::value_type>{};
    constexpr auto input2Size = std::tuple_size<typename ExtractPipeType<input2_pipe>::value_type>{};
    using res_pipe_unit_T = typename ExtractPipeType<res_pipe>::value_type::value_type;
#endif

#ifdef AUTOREG
    while (true){
        bool exit_task = 0;
#endif

    ConcatLoopHeight1:
        [[intel::initiation_interval(1)]] for (int i = 0; i < CONFIG_T::n_elem1_0; i++) {
        #ifdef AUTOREG
            [[intel::fpga_register]] auto in_data1_pipe = input1_pipe::read();
            [[intel::fpga_register]] auto in_data1 = in_data1_pipe.data;
            [[intel::fpga_register]] typename ExtractPipeType<res_pipe>::value_type::data_type out_data;
            if (in_data1_pipe.exit_task) {
                exit_task = true;
                break;
            }
        #else
            [[intel::fpga_register]] auto in_data1 = input1_pipe::read();
            [[intel::fpga_register]] typename ExtractPipeType<res_pipe>::value_type out_data;
        #endif

        ConcatPackInput1:
            #pragma unroll
            for (int k = 0; k < input1Size; k++) {
                out_data[k] = static_cast<res_pipe_unit_T>(in_data1[k]);
            }
        #ifdef AUTOREG
            out_data_pipe.data = out_data;
            out_data_pipe.exit_task = false;
            res_pipe::write(out_data_pipe);
        #else
            res_pipe::write(out_data);
        #endif
        }


    ConcatLoopHeight2:
        [[intel::initiation_interval(1)]] for (int i = 0; i < CONFIG_T::n_elem2_0; i++) {
        #ifdef AUTOREG
            [[intel::fpga_register]] auto in_data2_pipe = input2_pipe::read();
            [[intel::fpga_register]] auto in_data2 = in_data2_pipe.data;
            [[intel::fpga_register]] typename ExtractPipeType<res_pipe>::value_type::data_type out_data;
            if (in_data2_pipe.exit_task) {
                exit_task = true;
                break;
            }
        #else
            [[intel::fpga_register]] auto in_data2 = input2_pipe::read();
            [[intel::fpga_register]] typename ExtractPipeType<res_pipe>::value_type out_data;
        #endif

        ConcatPackInput2:
            #pragma unroll
            for (int k = 0; k < input2Size; k++) {
                out_data[k] = static_cast<res_pipe_unit_T>(in_data2[k]);
            }

        #ifdef AUTOREG
            out_data_pipe.data = out_data;
            out_data_pipe.exit_task = false;
            res_pipe::write(out_data_pipe);
        #else
            res_pipe::write(out_data);
        #endif
        }
#ifdef AUTOREG
        if(exit_task) break;
    }
    out_data_pipe.exit_task = true;
    res_pipe::write(out_data_pipe);
#endif
}

template <class input1_pipe, class input2_pipe, class res_pipe, typename CONFIG_T> void concatenate2d_1_stream() {
#ifdef AUTOREG
    constexpr auto input1Size = std::tuple_size<typename ExtractPipeType<input1_pipe>::value_type::data_type>{};
    constexpr auto input2Size = std::tuple_size<typename ExtractPipeType<input2_pipe>::value_type::data_type>{};
    using res_pipe_unit_T = typename ExtractPipeType<res_pipe>::value_type::data_type::value_type;
    [[intel::fpga_register]] typename ExtractPipeType<res_pipe>::value_type out_data_pipe;
#else
    constexpr auto input1Size = std::tuple_size<typename ExtractPipeType<input1_pipe>::value_type>{};
    constexpr auto input2Size = std::tuple_size<typename ExtractPipeType<input2_pipe>::value_type>{};
    using res_pipe_unit_T = typename ExtractPipeType<res_pipe>::value_type::value_type;
#endif

#ifdef AUTOREG
    while (true){
        bool exit_task = 0;
#endif

    ConcatLoopHeight:
        [[intel::initiation_interval(1)]] for (int i = 0; i < CONFIG_T::n_elem1_0; i++) {
        #ifdef AUTOREG
            [[intel::fpga_register]] auto in_data1_pipe = input1_pipe::read();
            [[intel::fpga_register]] auto in_data2_pipe = input2_pipe::read();
            [[intel::fpga_register]] auto in_data1 = in_data1_pipe.data;
            [[intel::fpga_register]] auto in_data2 = in_data2_pipe.data;
            [[intel::fpga_register]] typename ExtractPipeType<res_pipe>::value_type::data_type out_data;
            if (in_data1_pipe.exit_task || in_data2_pipe.exit_task){
                exit_task = true; 
                break;
            }
        #else
            [[intel::fpga_register]] auto in_data1 = input1_pipe::read();
            [[intel::fpga_register]] auto in_data2 = input2_pipe::read();
            [[intel::fpga_register]] typename ExtractPipeType<res_pipe>::value_type out_data;
        #endif

        ConcatPackInput1:
            #pragma unroll
            for (int k = 0; k < input1Size; k++) {
                out_data[k] = static_cast<res_pipe_unit_T>(in_data1[k]);
            }

        ConcatPackInput2:
            #pragma unroll
            for (int k = 0; k < input2Size; k++) {
                out_data[input1Size + k] = static_cast<res_pipe_unit_T>(in_data2[k]);
            }
        #ifdef AUTOREG
            if(exit_task) break;
            out_data_pipe.data = out_data;
            out_data_pipe.exit_task = false;
            res_pipe::write(out_data_pipe);
        #else
            res_pipe::write(out_data);
        #endif
        }
#ifdef AUTOREG
    }
    out_data_pipe.exit_task = true;
    res_pipe::write(out_data_pipe);
#endif
}

template <class input1_pipe, class input2_pipe, class res_pipe, typename CONFIG_T> void concatenate2d_stream() {
    if (CONFIG_T::axis == 2 || CONFIG_T::axis == -1) {
        concatenate2d_1_stream<input1_pipe, input2_pipe, res_pipe, CONFIG_T>();
    } else {
        concatenate2d_0_stream<input1_pipe, input2_pipe, res_pipe, CONFIG_T>();
    }
}

template <class input1_pipe, class input2_pipe, class res_pipe, typename CONFIG_T> void concatenate3d_0_stream() {
#ifdef AUTOREG
    constexpr auto input1Size = std::tuple_size<typename ExtractPipeType<input1_pipe>::value_type::data_type>{};
    constexpr auto input2Size = std::tuple_size<typename ExtractPipeType<input2_pipe>::value_type::data_type>{};
    using res_pipe_unit_T = typename ExtractPipeType<res_pipe>::value_type::data_type::value_type;
    [[intel::fpga_register]] typename ExtractPipeType<res_pipe>::value_type out_data_pipe;
#else
    constexpr auto input1Size = std::tuple_size<typename ExtractPipeType<input1_pipe>::value_type>{};
    constexpr auto input2Size = std::tuple_size<typename ExtractPipeType<input2_pipe>::value_type>{};
    using res_pipe_unit_T = typename ExtractPipeType<res_pipe>::value_type::value_type;
#endif

#ifdef AUTOREG
    while (true){
        bool exit_task = 0;
#endif
    ConcatLoopHeight1:
        for (int i = 0; i < CONFIG_T::n_elem1_0; i++) {

        ConcatLoopWidth1:
            [[intel::initiation_interval(1)]] for (int j = 0; j < CONFIG_T::n_elem1_1; j++) {
            #ifdef AUTOREG
                [[intel::fpga_register]] auto in_data1_pipe = input1_pipe::read();
                [[intel::fpga_register]] auto in_data1 = in_data1_pipe.data;
                [[intel::fpga_register]] typename ExtractPipeType<res_pipe>::value_type::data_type out_data;
                if (in_data1_pipe.exit_task) {
                    exit_task = true;
                    break;
                }
            #else
                [[intel::fpga_register]] auto in_data1 = input1_pipe::read();
                [[intel::fpga_register]] typename ExtractPipeType<res_pipe>::value_type out_data;
            #endif

            ConcatPackInput1:
                #pragma unroll
                for (int k = 0; k < input1Size; k++) {
                    out_data[k] = static_cast<res_pipe_unit_T>(in_data1[k]);
                }

            #ifdef AUTOREG
                out_data_pipe.data = out_data;
                out_data_pipe.exit_task = false;
                res_pipe::write(out_data_pipe);
            #else
                res_pipe::write(out_data);
            #endif
            }
        #ifdef AUTOREG
            if(exit_task) break;
        #endif
        }

    ConcatLoopHeight2:
        for (int i = 0; i < CONFIG_T::n_elem2_0; i++) {

        ConcatLoopWidth2:
            [[intel::initiation_interval(1)]] for (int j = 0; j < CONFIG_T::n_elem2_1; j++) {
            #ifdef AUTOREG
                [[intel::fpga_register]] auto in_data2_pipe = input2_pipe::read();
                [[intel::fpga_register]] auto in_data2 = in_data2_pipe.data;
                [[intel::fpga_register]] typename ExtractPipeType<res_pipe>::value_type::data_type out_data;
                if (in_data2_pipe.exit_task) {
                    exit_task = true;
                    break;
                }
            #else
                [[intel::fpga_register]] auto in_data2 = input2_pipe::read();
                [[intel::fpga_register]] typename ExtractPipeType<res_pipe>::value_type out_data;
            #endif

            ConcatPackInput2:
                #pragma unroll
                for (int k = 0; k < input2Size; k++) {
                    out_data[k] = static_cast<res_pipe_unit_T>(in_data2[k]);
                }

            #ifdef AUTOREG
                out_data_pipe.data = out_data;
                out_data_pipe.exit_task = false;
                res_pipe::write(out_data_pipe);
            #else
                res_pipe::write(out_data);
            #endif
            }
        #ifdef AUTOREG
            if(exit_task) break;
        #endif
        }
#ifdef AUTOREG
        if(exit_task) break;
    }
    out_data_pipe.exit_task = true;
    res_pipe::write(out_data_pipe);
#endif
}

template <class input1_pipe, class input2_pipe, class res_pipe, typename CONFIG_T> void concatenate3d_1_stream() {
#ifdef AUTOREG
    constexpr auto input1Size = std::tuple_size<typename ExtractPipeType<input1_pipe>::value_type::data_type>{};
    constexpr auto input2Size = std::tuple_size<typename ExtractPipeType<input2_pipe>::value_type::data_type>{};
    using res_pipe_unit_T = typename ExtractPipeType<res_pipe>::value_type::data_type::value_type;
    [[intel::fpga_register]] typename ExtractPipeType<res_pipe>::value_type out_data_pipe;
#else
    constexpr auto input1Size = std::tuple_size<typename ExtractPipeType<input1_pipe>::value_type>{};
    constexpr auto input2Size = std::tuple_size<typename ExtractPipeType<input2_pipe>::value_type>{};
    using res_pipe_unit_T = typename ExtractPipeType<res_pipe>::value_type::value_type;
#endif

#ifdef AUTOREG
    while (true){
        bool exit_task = 0;
#endif
    ConcatLoopHeight:
        for (int i = 0; i < CONFIG_T::n_elem1_0; i++) {
        ConcatLoopWidth1:
            [[intel::initiation_interval(1)]] for (int j = 0; j < CONFIG_T::n_elem1_1; j++) {
            #ifdef AUTOREG
                [[intel::fpga_register]] auto in_data1_pipe = input1_pipe::read();
                [[intel::fpga_register]] auto in_data1 = in_data1_pipe.data;
                [[intel::fpga_register]] typename ExtractPipeType<res_pipe>::value_type::data_type out_data;
                if (in_data1_pipe.exit_task) {
                    exit_task = true;
                    break;
                }
            #else
                [[intel::fpga_register]] auto in_data1 = input1_pipe::read();
                [[intel::fpga_register]] typename ExtractPipeType<res_pipe>::value_type out_data;
            #endif


            ConcatPackInput1:
                #pragma unroll
                for (int k = 0; k < input1Size; k++) {
                    out_data[k] = static_cast<res_pipe_unit_T>(in_data1[k]);
                }
            #ifdef AUTOREG
                out_data_pipe.data = out_data;
                out_data_pipe.exit_task = false;
                res_pipe::write(out_data_pipe);
            #else
                res_pipe::write(out_data);
            #endif
            }

        ConcatLoopWidth2:
            [[intel::initiation_interval(1)]] for (int j = 0; j < CONFIG_T::n_elem2_1; j++) {
            #ifdef AUTOREG
                [[intel::fpga_register]] auto in_data2_pipe = input2_pipe::read();
                [[intel::fpga_register]] auto in_data2 = in_data2_pipe.data;
                [[intel::fpga_register]] typename ExtractPipeType<res_pipe>::value_type::data_type out_data;
                if (in_data2_pipe.exit_task) {
                    exit_task = true;
                    break;
                }
            #else
                [[intel::fpga_register]] auto in_data2 = input2_pipe::read();
                [[intel::fpga_register]] typename ExtractPipeType<res_pipe>::value_type out_data;
            #endif

            ConcatPackInput2:
                #pragma unroll
                for (int k = 0; k < input2Size; k++) {
                    out_data[k] = static_cast<res_pipe_unit_T>(in_data2[k]);
                }

            #ifdef AUTOREG
                out_data_pipe.data = out_data;
                out_data_pipe.exit_task = false;
                res_pipe::write(out_data_pipe);
            #else
                res_pipe::write(out_data);
            #endif
            }
        #ifdef AUTOREG
            if (exit_task) break;
        #endif
        }
#ifdef AUTOREG
        if(exit_task) break;
    }
    out_data_pipe.exit_task = true;
    res_pipe::write(out_data_pipe);
#endif
    
}

template <class input1_pipe, class input2_pipe, class res_pipe, typename CONFIG_T> void concatenate3d_2_stream() {
#ifdef AUTOREG
    constexpr auto input1Size = std::tuple_size<typename ExtractPipeType<input1_pipe>::value_type::data_type>{};
    constexpr auto input2Size = std::tuple_size<typename ExtractPipeType<input2_pipe>::value_type::data_type>{};
    using res_pipe_unit_T = typename ExtractPipeType<res_pipe>::value_type::data_type::value_type;
    [[intel::fpga_register]] typename ExtractPipeType<res_pipe>::value_type out_data_pipe;
#else
    constexpr auto input1Size = std::tuple_size<typename ExtractPipeType<input1_pipe>::value_type>{};
    constexpr auto input2Size = std::tuple_size<typename ExtractPipeType<input2_pipe>::value_type>{};
    using res_pipe_unit_T = typename ExtractPipeType<res_pipe>::value_type::value_type;
#endif

#ifdef AUTOREG
    while (true){
        bool exit_task = 0;
#endif
    ConcatLoopHeight:
        for (int i = 0; i < CONFIG_T::n_elem1_0; i++) {
        ConcatLoopWidth:
            [[intel::initiation_interval(1)]] for (int j = 0; j < CONFIG_T::n_elem1_1; j++) {
            #ifdef AUTOREG
                [[intel::fpga_register]] auto in_data1_pipe = input1_pipe::read();
                [[intel::fpga_register]] auto in_data2_pipe = input2_pipe::read();
                [[intel::fpga_register]] auto in_data1 = in_data1_pipe.data;
                [[intel::fpga_register]] auto in_data2 = in_data2_pipe.data;
                [[intel::fpga_register]] typename ExtractPipeType<res_pipe>::value_type::data_type out_data;
                if (in_data1_pipe.exit_task || in_data2_pipe.exit_task){
                    exit_task = true; 
                    break;
                }
            #else
                [[intel::fpga_register]] auto in_data1 = input1_pipe::read();
                [[intel::fpga_register]] auto in_data2 = input2_pipe::read();
                [[intel::fpga_register]] typename ExtractPipeType<res_pipe>::value_type out_data;
            #endif

            ConcatPackInput1:
                #pragma unroll
                for (int k = 0; k < input1Size; k++) {
                    out_data[k] = static_cast<res_pipe_unit_T>(in_data1[k]);
                }

            ConcatPackInput2:
                #pragma unroll
                for (int k = 0; k < input2Size; k++) {
                    out_data[input1Size + k] =
                        static_cast<res_pipe_unit_T>(in_data2[k]);
                }
            #ifdef AUTOREG
                out_data_pipe.data = out_data;
                out_data_pipe.exit_task = false;
                res_pipe::write(out_data_pipe);
            #else
                res_pipe::write(out_data);
            #endif
            }
        #ifdef AUTOREG
            if (exit_task) break;
        #endif
        }
#ifdef AUTOREG
        if (exit_task) break;
    }
    out_data_pipe.exit_task = true;
    res_pipe::write(out_data_pipe);
#endif
}

template <class input1_pipe, class input2_pipe, class res_pipe, typename CONFIG_T> void concatenate3d_stream() {
    if (CONFIG_T::axis == 3 || CONFIG_T::axis == -1) {
        concatenate3d_2_stream<input1_pipe, input2_pipe, res_pipe, CONFIG_T>();
    } else if (CONFIG_T::axis == 2 || CONFIG_T::axis == -2) {
        concatenate3d_1_stream<input1_pipe, input2_pipe, res_pipe, CONFIG_T>();
    } else {
        concatenate3d_0_stream<input1_pipe, input2_pipe, res_pipe, CONFIG_T>();
    }
}

} // namespace nnet

#endif
