#ifndef AUTOREGRESSIVE_HELPER_H_
#define AUTOREGRESSIVE_HELPER_H_

/*
Global switch to keep track of where we read from,
use seperate switches to avoid pipeline stall issues
*/
#include "nnet_common.h"
#include "nnet_helpers.h"
#include "nnet_printf.h"
#include "nnet_types.h"
#include <tuple>
#include <type_traits>
namespace nnet {

template <class host_pipe, class fb_pipe, class sw_pipe> struct SwitchSet {
    using host = host_pipe;
    using feedback = fb_pipe;
    using sw = sw_pipe;
};

// We expect nnet::array<arr_T, 1> so a size 1 array for token feedback.
// This unit also converts a vector to token id stream.
// THIS VERSION ASSUMES HOST PHASE RUNS ONLY ONCE
template <class host_pipe, class feedback_pipe, class switch_pipe, typename CONFIG_T> void input_switch() {

    using host_pipe_T = typename ExtractPipeType<host_pipe>::value_type;
    using fb_pipe_T = typename ExtractPipeType<feedback_pipe>::value_type;
    using data_T = typename host_pipe_T::value_type;
    using sw_pipe_T = typename ExtractPipeType<switch_pipe>::value_type;
    using sw_arr_T = typename sw_pipe_T::data_type;

    // TODO - CHECK IF THESE QUANTISE TO THE SAME AS SW_DATA_T##############################
    static constexpr data_T SWITCH_ID = CONFIG_T::switch_signal;
    static constexpr data_T STOP_ID = CONFIG_T::stop_signal;
    //#####################################################################################

    static constexpr unsigned HOST_VECT_SIZE = std::tuple_size<host_pipe_T>{};
    using host_vect_bits_t = ac_int<HOST_VECT_SIZE, false>;

    static constexpr unsigned MAX_ITERATIONS = CONFIG_T::max_iterations;
    static constexpr unsigned MAX_ITER_BITS = ceil_log2(CONFIG_T::max_iterations) + 1;
    static constexpr bool INP_EXCEEDS_MAX_ITER = (HOST_VECT_SIZE >= MAX_ITERATIONS);

    [[intel::fpga_register]] host_vect_bits_t exit_task_bits = 0;
    [[intel::fpga_register]] host_vect_bits_t fb_bits = 0;

    // Initial host transmission
    [[intel::fpga_register]] host_pipe_T in_data = host_pipe::read();

    #pragma unroll
    for (unsigned i = 0; i < HOST_VECT_SIZE; i++) {
        exit_task_bits[i] = in_data[i] == STOP_ID;
        fb_bits[i] = in_data[i] == SWITCH_ID;
    }

    if (exit_task_bits[0] || fb_bits[0]) {
        switch_pipe::write(sw_pipe_T{sw_arr_T{}, true, false});
        return;
    }

    [[intel::fpga_register]] data_T prev_token = in_data[0];
    [[intel::fpga_register]] bool exit_task = false;
    [[intel::fpga_register]] bool switch_to_fb = false;

    exit_task_bits >>= 1;
    fb_bits >>= 1;

    [[intel::fpga_register]] unsigned host_it = 1;
    for (; host_it < HOST_VECT_SIZE; host_it++) {
        bool switch_fb = fb_bits[0];
        fb_bits >>= 1;
        if (switch_fb)
            switch_to_fb = true;
        switch_pipe::write(sw_pipe_T{sw_arr_T{prev_token}, exit_task, switch_fb});
        if (exit_task)
            return;
        data_T el = in_data[host_it];
        exit_task = (exit_task_bits[0] || host_it + 1 >= MAX_ITERATIONS);
        exit_task_bits >>= 1;
        prev_token = el;
        if (switch_to_fb)
            break;
    }

    //[[intel::fpga_register]] unsigned iter_ct = host_it;//[[intel::fpga_register]] iterct_t iter_ct = host_it;

    if (!switch_to_fb) {
        bool final_el_exit = exit_task || INP_EXCEEDS_MAX_ITER;
        switch_pipe::write(sw_pipe_T{sw_arr_T{prev_token}, final_el_exit, !final_el_exit});
        if (final_el_exit)
            return;
        // iter_ct = HOST_VECT_SIZE;
        switch_to_fb = true;
    }

    using iter_T = ac_int<MAX_ITER_BITS + 1, true>;
    [[intel::fpga_register]] iter_T remaining_iters = static_cast<iter_T>((MAX_ITERATIONS - 1 - host_it) - 1);

    while (true) {
        // Feedback path is a per-token stream so we expect nnet::array<type,1>
        fb_pipe_T inp_id = feedback_pipe::read();
        // Look for sign flip instead of remaining_iters == 0 check
        bool max_iter_reached = remaining_iters[MAX_ITER_BITS];
        remaining_iters--;
        bool exit_task = (inp_id[0] == STOP_ID | max_iter_reached);
        switch_pipe::write(sw_pipe_T{inp_id, exit_task, !exit_task});
        if (exit_task)
            return;
    }
}

// Returns the maximum value from an array of size N - assumes data is array and output size is always 1 (not an array)
template <typename data_arr_T, unsigned N, typename res_T> inline res_T argmax_stream(data_arr_T data) {

    constexpr unsigned loops = ceil_log2(N);
    constexpr unsigned padded_len = 1 << loops;

    // Single element requires no comparisons so return straight away
    if constexpr (loops == 0)
        return static_cast<res_T>(data[0]);
    else {
        using idx_arr_T = typename nnet::array<ac_fixed<loops, loops, false>, padded_len>;
        using data_in_T = typename data_arr_T::value_type;

        [[intel::fpga_register]] idx_arr_T maxval_idxs;

        #pragma unroll
        for (unsigned it = 0; it < padded_len; it++) {
            maxval_idxs[it] = it;
        }

        #pragma unroll
        for (unsigned st = 0; st < loops; st++) {
            unsigned curr_len = padded_len >> (st + 1);
            #pragma unroll loops
            for (unsigned l = 0; l < curr_len; l++) {
                data_in_T data_first = ((2 * l) < N) ? data[2 * l] : minval<data_in_T>();
                data_in_T data_second = ((2 * l + 1) < N) ? data[2 * l + 1] : minval<data_in_T>();
                data[l] = (data_first > data_second) ? data_first : data_second;
                maxval_idxs[l] = (data_first > data_second) ? maxval_idxs[2 * l] : maxval_idxs[2 * l + 1];
            }
        }
        return static_cast<res_T>(maxval_idxs[0]);
    }
}

template <class host_pipe, class feedback_pipe, class switch_pipe, typename CONFIG_T> void output_switch() {

    using data_in_pipe_T = typename ExtractPipeType<switch_pipe>::value_type;
    using data_in_arr_T = typename data_in_pipe_T::data_type;
    using fb_arr_T = typename ExtractPipeType<feedback_pipe>::value_type;
    static constexpr unsigned data_in_arr_size = std::tuple_size<data_in_arr_T>{};
    using res_pipe_T = typename ExtractPipeType<host_pipe>::value_type;
    using res_arr_T = typename res_pipe_T::data_type;
    using res_T = typename res_arr_T::value_type;

    if constexpr (!CONFIG_T::argmax) {
        static_assert(std::is_same_v<data_in_pipe_T, res_pipe_T>,
                      "output_switch: switch_pipe packet type and host_pipe packet type differ");
    }

    while (true) {
        data_in_pipe_T data_pack = switch_pipe::read();
        bool exit_task = data_pack.exit_task;
        bool fb = data_pack.feedback;

        if constexpr (CONFIG_T::argmax) {
            res_arr_T maxval_idx{};
            if (!exit_task) {
                maxval_idx[0] = argmax_stream<data_in_arr_T, data_in_arr_size, res_T>(data_pack.data);

                if (fb) {
                    fb_arr_T out;
                    #pragma unroll
                    for (int i = 0; i < std::tuple_size<fb_arr_T>{}; i++) {
                        out[i] = maxval_idx[0];
                    }
                    feedback_pipe::write(out);
                }
            }
            host_pipe::write(res_pipe_T{maxval_idx, exit_task, false});
            if (exit_task)
                return;
        } else {
            if (!exit_task) {
                if (fb) {
                    fb_arr_T out;
                    #pragma unroll
                    for (int i = 0; i < std::tuple_size<fb_arr_T>{}; i++) {
                        out[i] = data_pack.data[i];
                    }
                    feedback_pipe::write(out);
                }
            }
            host_pipe::write(res_pipe_T{data_pack.data, exit_task, false});
            if (exit_task)
                return;
        }
    }
}
} // namespace nnet
#endif
