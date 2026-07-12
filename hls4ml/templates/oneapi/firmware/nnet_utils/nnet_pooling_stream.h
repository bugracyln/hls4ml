#ifndef NNET_POOLING_STREAM_H_
#define NNET_POOLING_STREAM_H_

#include "nnet_conv1d_stream.h"
#include "nnet_conv2d_stream.h"
#include "nnet_pooling.h"
#include "nnet_types.h"

namespace nnet {

/*
 * void compute_pool_buffer_1d(in_element, res_stream, line_buffer, kernel_window)
 *
 * Args:
 *   in_element - current elements from input image, data_T type is usually nnet::array, size of array corresponds to number
 * of channels res_stream - output stream, passed by reference to allow direct writing line_buffer - chained array of shift
 * registers, one for each row of the pool and channel kernel_window - array of values from the input curently being pooled
 *
 * Function executes 4 steps:
 *   (1) Shift line buffer - updates the contents of the chained shift registers, inserting the new inputs and removing last
 * elements (2) Kernel shift - updates the elements of the kernel window, by storing the new inputs and popped elements from
 * the line buffer (3) Pooling - performs dense matrix multiplication between the current input window and kernel weights (4)
 * Counter housekeeping - performs the required pooling operation
 *
 */
template <class data_arr_T, class data_window_T, class res_pipe, typename CONFIG_T>
void compute_pool_buffer_1d(const data_arr_T &in_elem,
                            #ifdef AUTOREG
                            nnet::shift_reg<typename data_arr_T::data_type::value_type, CONFIG_T::in_width> 
                            #else
                            nnet::shift_reg<typename data_arr_T::value_type, CONFIG_T::in_width> 
                            #endif
                            line_buffer[CONFIG_T::n_filt],
                            data_window_T &kernel_window, int &pX, int &sX 
                        #ifdef AUTOREG
                            , bool &exit_task
                        #endif
                            ) {

    #ifdef AUTOREG
        using data_T = typename data_arr_T::data_type;
        using res_T = typename ExtractPipeType<res_pipe>::data_type::value_type;
        [[intel::fpga_register]]  typename ExtractPipeType<res_pipe>::value_type res_pack_pipe;

        if (in_elem.exit_task){
            exit_task = true;
            res_pack_pipe.exit_task = true;
            res_pipe::write(res_pack_pipe);
            return;
        }
    #else
        using data_T = data_arr_T;            
        using res_T = typename ExtractPipeType<res_pipe>::value_type;
    #endif

    // Thresholds
    constexpr int lShiftX = CONFIG_T::pool_width - 1;

    // Step 1 - Shift line buffer
    [[intel::fpga_register]] typename data_T::value_type shift_buffer[CONFIG_T::n_filt];
#ifdef AUTOREG
    nnet::shift_line_buffer_1d<data_T, CONFIG_T>(in_elem.data, line_buffer, shift_buffer);
#else
    nnet::shift_line_buffer_1d<data_T, CONFIG_T>(in_elem, line_buffer, shift_buffer);
#endif

    // Step 2 - Kernel shift
    nnet::kernel_shift_1d<data_T, data_window_T, CONFIG_T>(shift_buffer, kernel_window);

    // Check to see if we have a full pool window
    if ((sX - lShiftX) == 0 && pX > (lShiftX - 1)) {
        [[intel::fpga_register]] res_T res_pack;

    FiltLoop:
        #pragma unroll
        for (int filter = 0; filter < CONFIG_T::n_filt; filter++) {
            [[intel::fpga_register]] typename data_T::value_type pool_window[CONFIG_T::pool_width];

        // Retrieve data for current channel
        PoolLoop:
            #pragma unroll
            for (int i = 0; i < CONFIG_T::pool_width; i++) {
                pool_window[i] = kernel_window[i * CONFIG_T::n_filt + filter];
            }

            // Step 3 - Pooling
            res_pack[filter] = static_cast<typename res_T::value_type>(
                pool_op<typename data_T::value_type, CONFIG_T::pool_width, CONFIG_T::pool_op, typename CONFIG_T::accum_t>(
                    pool_window));
        }

        // Write result to output stream
    #ifdef AUTOREG
        res_pack_pipe.data = res_pack;
        res_pack_pipe.exit_task = false;
        res_pipe::write(res_pack_pipe);
    #else
        res_pipe::write(res_pack);
    #endif
    }

    // Reached end of image
    if ((pX + 1) == (CONFIG_T::in_width + CONFIG_T::pad_left + CONFIG_T::pad_right)) {
        pX = 0;
        sX = 0;
        // Move to the right
    } else {
        pX++;
        sX = ((sX - lShiftX) == 0) ? (sX - CONFIG_T::stride_width + 1) : (sX + 1);
    }
}

template <class data_pipe, class res_pipe, typename CONFIG_T> void pooling1d_cl_stream() {
    assert(CONFIG_T::pool_width == CONFIG_T::stride_width);
    assert(CONFIG_T::pad_left == 0 && CONFIG_T::pad_right == 0);

    using data_arr_T = typename ExtractPipeType<data_pipe>::value_type;
#ifdef AUTOREG
    using data_element_T = typename data_arr_T::data_type::value_type;
#else
    using data_element_T = typename data_arr_T::value_type;
#endif
    
    using data_window_T = array<data_element_T, CONFIG_T::pool_width * CONFIG_T::n_filt>;

#ifdef AUTOREG
    while (true){
        bool exit_task = 0;
#endif

        // Line buffer and kernel window
        [[intel::fpga_register]] nnet::shift_reg<data_element_T, CONFIG_T::in_width> line_buffer[CONFIG_T::n_filt];
        [[intel::fpga_register]] data_window_T kernel_window;

        // move former static variables outside the function calls
        // X position pixel
        int pX = 0;
        // X strides
        int sX = 0;

    // Read input image
    ReadInputWidth:
        for (int col = 0; col < CONFIG_T::in_width; col++) {
            compute_pool_buffer_1d<data_arr_T, data_window_T, res_pipe, CONFIG_T>(data_pipe::read(), line_buffer, kernel_window,
                                                                                pX, sX 
                                                                                #ifdef AUTOREG
                                                                                , exit_task
                                                                                #endif
                                                                                );
        #ifdef AUTOREG
            if (exit_task) break;
        #endif                                                          
        }
#ifdef AUTOREG
        if (exit_task) break;
    }
#endif
}

/*
 * void compute_pool_buffer_2d(in_element, res_stream, line_buffer, kernel_window)
 *
 * Args:
 *   in_element - current elements from input image, data_T type is usually nnet::array, size of array corresponds to number
 * of channels res_stream - output stream, passed by reference to allow direct writing line_buffer - chained array of shift
 * registers, one for each row of the pool and channel kernel_window - array of values from the input curently being pooled
 *
 * Function executes 4 steps:
 *   (1) Shift line buffer - updates the contents of the chained shift registers, inserting the new inputs and removing last
 * elements (2) Kernel shift - updates the elements of the kernel window, by storing the new inputs and popped elements from
 * the line buffer (3) Pooling - performs dense matrix multiplication between the current input window and kernel weights (4)
 * Counter housekeeping - performs the required pooling operation
 *
 */
template <class data_arr_T, class data_window_T, class res_pipe, typename CONFIG_T>
void compute_pool_buffer_2d(const data_arr_T &in_elem,
                            #ifdef AUTOREG
                            nnet::shift_reg<typename data_arr_T::data_type::value_type, CONFIG_T::in_width>
                            #else
                            nnet::shift_reg<typename data_arr_T::value_type, CONFIG_T::in_width>
                            #endif
                            line_buffer[CONFIG_T::pool_height - 1][CONFIG_T::n_filt],
                            data_window_T &kernel_window, int &pX, int &pY, int &sX, int &sY                        
                            #ifdef AUTOREG
                            , bool &exit_task
                            #endif
                            ) {

    #ifdef AUTOREG
        using data_T = typename data_arr_T::data_type;
        using res_T = typename ExtractPipeType<res_pipe>::data_type::value_type;
        [[intel::fpga_register]]  typename ExtractPipeType<res_pipe>::value_type res_pack_pipe;

        if (in_elem.exit_task){
            exit_task = true;
            res_pack_pipe.exit_task = true;
            res_pipe::write(res_pack_pipe);
            return;
        }
    #else
        using data_T = data_arr_T;            
        using res_T = typename ExtractPipeType<res_pipe>::value_type;
    #endif

    // Thresholds
    static constexpr int lShiftX = CONFIG_T::pool_width - 1;
    static constexpr int lShiftY = CONFIG_T::pool_height - 1;

    // Step 1 - Shift line buffer
    [[intel::fpga_register]] typename data_T::value_type shift_buffer[CONFIG_T::pool_height][CONFIG_T::n_filt];
#ifdef AUTOREG
    nnet::shift_line_buffer_2d<data_T, CONFIG_T>(in_elem.data, line_buffer, shift_buffer);
#else
    nnet::shift_line_buffer_2d<data_T, CONFIG_T>(in_elem, line_buffer, shift_buffer);
#endif


    // Step 2 - Kernel shift
    nnet::kernel_shift_2d<data_T, data_window_T, CONFIG_T>(shift_buffer, kernel_window);

    // Check to see if we have a full pool window
    if ((sX - lShiftX) == 0 && (sY - lShiftY) == 0 && pY > (lShiftY - 1) && pX > (lShiftX - 1)) {
        [[intel::fpga_register]] res_T res_pack;

    FiltLoop:
        #pragma unroll
        for (int filter = 0; filter < CONFIG_T::n_filt; filter++) {
            [[intel::fpga_register]] typename data_T::value_type pool_window[CONFIG_T::pool_height * CONFIG_T::pool_width];

        // Retrieve data for current channel
        PoolLoop:
            #pragma unroll
            for (int i = 0; i < CONFIG_T::pool_height * CONFIG_T::pool_width; i++) {
                pool_window[i] = kernel_window[i * CONFIG_T::n_filt + filter];
            }

            // Step 3 - Pooling
            res_pack[filter] = static_cast<typename res_T::value_type>(
                pool_op<typename data_T::value_type, CONFIG_T::pool_height * CONFIG_T::pool_width, CONFIG_T::pool_op,
                        typename CONFIG_T::accum_t>(pool_window));
        }

        // Write result to output stream
    #ifdef AUTOREG
        res_pack_pipe.data = res_pack;
        res_pack_pipe.exit_task = false;
        res_pipe::write(res_pack_pipe);
    #else
        res_pipe::write(res_pack);
    #endif
    }

    // Reached end of image
    if ((pX + 1) == (CONFIG_T::in_width + CONFIG_T::pad_left + CONFIG_T::pad_right) &&
        (pY + 1) == (CONFIG_T::in_height + CONFIG_T::pad_top + CONFIG_T::pad_bottom)) {
        pX = 0;
        sX = 0;
        pY = 0;
        sY = 0;
        // Reached end of row
    } else if ((pX + 1) == (CONFIG_T::in_width + CONFIG_T::pad_left + CONFIG_T::pad_right)) {
        pX = 0;
        sX = 0;
        pY++;
        sY = ((sY - lShiftY) == 0) ? (sY - CONFIG_T::stride_height + 1) : (sY + 1);
        // Same row, same colum, therefore, move to the right
    } else {
        pX++;
        sX = ((sX - lShiftX) == 0) ? (sX - CONFIG_T::stride_width + 1) : (sX + 1);
    }
}

template <class data_pipe, class res_pipe, typename CONFIG_T> void pooling2d_cl_stream() {
    assert(CONFIG_T::pool_height == CONFIG_T::stride_height && CONFIG_T::pool_width == CONFIG_T::stride_width);
    assert(CONFIG_T::pad_left == 0 && CONFIG_T::pad_right == 0);
    assert(CONFIG_T::pad_top == 0 && CONFIG_T::pad_bottom == 0);

    using data_arr_T = typename ExtractPipeType<data_pipe>::value_type;
    using data_element_T = typename data_arr_T::value_type;
    using data_window_T = array<data_element_T, CONFIG_T::pool_height * CONFIG_T::pool_width * CONFIG_T::n_filt>;

#ifdef AUTOREG
    while (true){
        bool exit_task = 0;
#endif

        // Line buffer and kernel window
        [[intel::fpga_register]] nnet::shift_reg<data_element_T, CONFIG_T::in_width>
            line_buffer[MAX(CONFIG_T::pool_height - 1, 1)][CONFIG_T::n_filt];
        [[intel::fpga_register]] data_window_T kernel_window;

        // former static variables
        // X, Y position pixels
        int pX = 0;
        int pY = 0;

        // X, Y strides
        int sX = 0;
        int sY = 0;

    ReadInputHeight:
        [[intel::loop_coalesce(2)]] for (int row = 0; row < CONFIG_T::in_height; row++) {
        // Read input image
        ReadInputWidth:
            for (int col = 0; col < CONFIG_T::in_width; col++) {
                compute_pool_buffer_2d<data_arr_T, data_window_T, res_pipe, CONFIG_T>(data_pipe::read(), line_buffer,
                                                                                    kernel_window, pX, pY, sX, sY 
                                                                                #ifdef AUTOREG
                                                                                    , exit_task
                                                                                #endif
                                                                                    );
            #ifdef AUTOREG
                if (exit_task) break;
            #endif
            }
        #ifdef AUTOREG
            if (exit_task) break;
        #endif
        }
#ifdef AUTOREG
        if (exit_task) break;
    }
#endif
}

/*
 * A function used with Global Pooling
 * Updates the output pooling value
 * Max : Return the maximum between the previous maximum and current input
 * Avg : Returns the cumulative sum
 */
template <class T_y, class T_x, Pool_Op op> inline T_y reduce_global_pool(T_y y, T_x x) {
    if (op == Max) {
        return (x > y) ? (T_y)x : y;
    } else {
        return (T_y)(x + y);
    }
}

/*
 * A function used with Global Pooling
 * For every filter, it updates the value by summing the current input (Average) or updating the maximum value (Max)
 */
template <class data_T, class res_T, typename CONFIG_T> void compute_global_pool(const data_T &in_elem, res_T &data_input) {
    #pragma unroll
    for (unsigned i = 0; i < CONFIG_T::n_filt; i++) {
        data_input[i] = reduce_global_pool<typename CONFIG_T::accum_t, typename data_T::value_type, CONFIG_T::pool_op>(
            data_input[i], in_elem[i]);
    }
}

template <class data_pipe, class res_pipe, typename CONFIG_T> void global_pooling1d_cl_stream() {
    assert(CONFIG_T::pad_left == 0 && CONFIG_T::pad_right == 0);

#ifdef AUTOREG
    using data_pipe_T = typename ExtractPipeType<data_pipe>::value_type;
    using res_pipe_T = typename ExtractPipeType<res_pipe>::value_type;
    using data_T = typename ExtractPipeType<data_pipe>::data_type::value_type;
    using res_T = typename ExtractPipeType<res_pipe>::data_type::value_type;

    [[intel::fpga_register]] res_pipe_T out_data_pipe;
#else
    using data_T = typename ExtractPipeType<data_pipe>::value_type;
    using res_T = typename ExtractPipeType<res_pipe>::value_type;
#endif

    using accum_arr_t = array<typename CONFIG_T::accum_t, CONFIG_T::n_filt>;

#ifdef AUTOREG
    while (true){
#endif
        [[intel::fpga_register]] accum_arr_t data_input;

        #pragma unroll
        for (int i = 0; i < CONFIG_T::n_filt; i++) {
            data_input[i] = pad_val<typename CONFIG_T::accum_t, CONFIG_T::pool_op>();
        }

        for (int i = 0; i < CONFIG_T::n_in; i++) {
        #ifdef AUTOREG
            data_pipe_T in_data_pipe = data_pipe::read();
            if (in_data_pipe.exit_task){
                out_data_pipe.exit_task = true; 
                res_pipe::write(out_data_pipe);
                return;
            }
            data_T in_data = in_data_pipe.data;
        #else
            data_T in_data = data_pipe::read();
        #endif
            compute_global_pool<data_T, accum_arr_t, CONFIG_T>(in_data, data_input);
        }

        [[intel::fpga_register]] res_T res_pack;
        if (CONFIG_T::pool_op == Average) {
            #pragma unroll
            for (int i = 0; i < CONFIG_T::n_filt; i++) {
                res_pack[i] = static_cast<typename res_T::value_type>(data_input[i] / CONFIG_T::n_in);
            }
        } else {
            #pragma unroll
            for (int i = 0; i < CONFIG_T::n_filt; i++) {
                res_pack[i] = static_cast<typename res_T::value_type>(data_input[i]);
            }
        }
    #ifdef AUTOREG
        out_data_pipe.data = res_pack;
        out_data_pipe.exit_task = false;
        res_pipe::write(out_data_pipe);
    #else
        res_pipe::write(res_pack);
    #endif
#ifdef AUTOREG
    }
#endif
}

template <class data_pipe, class res_pipe, typename CONFIG_T> void global_pooling2d_cl_stream() {
    assert(CONFIG_T::pad_left == 0 && CONFIG_T::pad_right == 0);
    assert(CONFIG_T::pad_top == 0 && CONFIG_T::pad_bottom == 0);

#ifdef AUTOREG
    using data_pipe_T = typename ExtractPipeType<data_pipe>::value_type;
    using res_pipe_T = typename ExtractPipeType<res_pipe>::value_type;
    using data_T = typename ExtractPipeType<data_pipe>::data_type::value_type;
    using res_T = typename ExtractPipeType<res_pipe>::data_type::value_type;

    [[intel::fpga_register]] res_pipe_T out_data_pipe;
#else
    using data_T = typename ExtractPipeType<data_pipe>::value_type;
    using res_T = typename ExtractPipeType<res_pipe>::value_type;
#endif

    using accum_arr_t = array<typename CONFIG_T::accum_t, CONFIG_T::n_filt>;

#ifdef AUTOREG
    while (true){
#endif

        [[intel::fpga_register]] accum_arr_t data_input;

        #pragma unroll
        for (int i = 0; i < CONFIG_T::n_filt; i++) {
            data_input[i] = pad_val<typename CONFIG_T::accum_t, CONFIG_T::pool_op>();
        }

        for (int i = 0; i < CONFIG_T::in_height; i++) {
            for (int j = 0; j < CONFIG_T::in_width; j++) {
            #ifdef AUTOREG
                data_pipe_T in_data_pipe = data_pipe::read();
                if (in_data_pipe.exit_task){
                    out_data_pipe.exit_task = true; 
                    res_pipe::write(out_data_pipe);
                    return;
                }
                data_T in_data = in_data_pipe.data;
            #else
                data_T in_data = data_pipe::read();
            #endif
                compute_global_pool<data_T, accum_arr_t, CONFIG_T>(in_data, data_input);
            }
        }

        [[intel::fpga_register]] res_T res_pack;
        if (CONFIG_T::pool_op == Average) {
            #pragma unroll
            for (int i = 0; i < CONFIG_T::n_filt; i++) {
                res_pack[i] =
                    static_cast<typename res_T::value_type>(data_input[i] / (CONFIG_T::in_width * CONFIG_T::in_height));
            }
        } else {
            #pragma unroll
            for (int i = 0; i < CONFIG_T::n_filt; i++) {
                res_pack[i] = static_cast<typename res_T::value_type>(data_input[i]);
            }
        }
    #ifdef AUTOREG
        out_data_pipe.data = res_pack;
        out_data_pipe.exit_task = false;
        res_pipe::write(out_data_pipe);
    #else
        res_pipe::write(res_pack);
    #endif
#ifdef AUTOREG
    }
#endif
}

} // namespace nnet

#endif
