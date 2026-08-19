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

template<class host_pipe, class fb_pipe, class sw_pipe>
struct SwitchSet {
    using host = host_pipe; 
    using feedback = fb_pipe;
    using sw = sw_pipe;
};

enum class PipeSignal {
    no_feedback, feedback, stop
};

struct SignalPack {
    PipeSignal signal = PipeSignal::no_feedback;
    unsigned writect = 0;
};

/*
// We expect nnet::array<arr_T, 1> so a size 1 array for token feedback.
// This unit also converts a vector to token id stream.
template <class host_pipe, class feedback_pipe, class switch_pipe, typename CONFIG_T> void input_switch() {
    
    using host_pipe_T = typename ExtractPipeType<host_pipe>::value_type;
    using fb_pipe_T = typename ExtractPipeType<feedback_pipe>::value_type;
    using data_T = typename host_pipe_T::value_type;
    using sw_pipe_T = typename ExtractPipeType<switch_pipe>::value_type;
    using sw_arr_T = typename sw_pipe_T::data_type;

    //TODO - CHECK IF THESE QUANTISE TO THE SAME AS SW_DATA_T##############################
    static constexpr data_T SWITCH_ID = CONFIG_T::switch_signal;
    static constexpr data_T STOP_ID = CONFIG_T::stop_signal;
    //#####################################################################################

    static constexpr unsigned MAX_ITER_BITS = ceil_log2(CONFIG_T::max_iterations) + 1;
    using iterct_t = ac_fixed<MAX_ITER_BITS,MAX_ITER_BITS,false>;
    static constexpr iterct_t MAX_ITERATIONS =  CONFIG_T::max_iterations;

    [[intel::fpga_register]] host_pipe_T in_data;
    [[intel::fpga_register]] bool switch_to_fb = false;
    [[intel::fpga_register]] iterct_t iter_ct = 0;

    while(true){
        if(switch_to_fb){ // Feedback path is a per-token stream so we expect nnet::array<type,1>
            fb_pipe_T inp_id = feedback_pipe::read();
            bool exit_task = (inp_id[0] == STOP_ID || iter_ct + 1 >= MAX_ITERATIONS);
            switch_pipe::write(sw_pipe_T{inp_id,exit_task,!exit_task});
            if (exit_task) return;
            iter_ct++;
        }
        else{ // This branch expects a single vector DMA transfer of switch IDs, serialises them
            in_data = host_pipe::read();
            for(unsigned host_it = 0; host_it < std::tuple_size<host_pipe_T>{}; host_it++){
                data_T el = in_data[host_it];
                bool exit_task = (el == STOP_ID || iter_ct + 1 >= MAX_ITERATIONS);
                if (el == SWITCH_ID) switch_to_fb = true;
                switch_pipe::write(sw_pipe_T{sw_arr_T{el},exit_task,el == SWITCH_ID});
                if (exit_task) return;
                iter_ct++;
            }
        }
    }
}
*/
// We expect nnet::array<arr_T, 1> so a size 1 array for token feedback.
// This unit also converts a vector to token id stream.
template <class host_pipe, class feedback_pipe, class switch_pipe, typename CONFIG_T> void input_switch() {
    
    using host_pipe_T = typename ExtractPipeType<host_pipe>::value_type;
    using fb_pipe_T = typename ExtractPipeType<feedback_pipe>::value_type;
    using data_T = typename host_pipe_T::value_type;
    using sw_pipe_T = typename ExtractPipeType<switch_pipe>::value_type;
    using sw_arr_T = typename sw_pipe_T::data_type;

    //TODO - CHECK IF THESE QUANTISE TO THE SAME AS SW_DATA_T##############################
    static constexpr data_T SWITCH_ID = CONFIG_T::switch_signal;
    static constexpr data_T STOP_ID = CONFIG_T::stop_signal;
    //#####################################################################################

    static constexpr unsigned MAX_ITER_BITS = ceil_log2(CONFIG_T::max_iterations) + 1;
    using iterct_t = ac_fixed<MAX_ITER_BITS,MAX_ITER_BITS,false>;
    static constexpr iterct_t MAX_ITERATIONS =  CONFIG_T::max_iterations;

    [[intel::fpga_register]] host_pipe_T in_data;
    [[intel::fpga_register]] bool switch_to_fb = false;
    [[intel::fpga_register]] iterct_t iter_ct = 0;

    while(true){
        if(switch_to_fb){ // Feedback path is a per-token stream so we expect nnet::array<type,1>
            fb_pipe_T inp_id = feedback_pipe::read();
            bool exit_task = (inp_id[0] == STOP_ID || iter_ct + 1 >= MAX_ITERATIONS);
            switch_pipe::write(sw_pipe_T{inp_id,exit_task,!exit_task});
            if (exit_task) return;
            iter_ct++;
        }
        else{ // This branch expects a single vector DMA transfer of switch IDs, serialises them
            in_data = host_pipe::read();
            if(in_data[0] == SWITCH_ID || in_data[0] == STOP_ID){ 
                switch_pipe::write(sw_pipe_T{sw_arr_T{},true,false});
                return;
            }
            data_T prev_token = in_data[0];
            bool exit_task = false;
    
            for(unsigned host_it = 1; host_it < std::tuple_size<host_pipe_T>{}; host_it++){
                data_T el = in_data[host_it];
                if (el == SWITCH_ID) switch_to_fb = true;
                switch_pipe::write(sw_pipe_T{sw_arr_T{prev_token},exit_task,el == SWITCH_ID});
                if (exit_task) return;
                iter_ct++;
                exit_task = (el == STOP_ID || iter_ct + 1 >= MAX_ITERATIONS);
                prev_token = el;
                if (switch_to_fb) break;
            }
            if(!switch_to_fb){
                bool final_el_exit = exit_task || iter_ct + 1 >= MAX_ITERATIONS;
                switch_pipe::write(sw_pipe_T{sw_arr_T{prev_token},final_el_exit,!final_el_exit});
                if (final_el_exit) return;
                iter_ct++;
                switch_to_fb = true;
            }
        }
    }
}

// Overloaded with the position index, required by token streaming models
/*
template <class host_pipe, class feedback_pipe, class switch_pipe, class switch_pos_pipe, class control_pipe, typename CONFIG_T> void input_switch() {

    using host_pipe_T = typename ExtractPipeType<host_pipe>::value_type;
    using fb_pipe_T = typename ExtractPipeType<feedback_pipe>::value_type;
    using unit_pipe_T = typename ExtractPipeType<switch_pipe>::value_type;
    using data_T = typename host_pipe_T::value_type;
    using sw_pipe_T = typename ExtractPipeType<switch_pipe>::value_type;
    using sw_pos_pipe_T = typename ExtractPipeType<switch_pos_pipe>::value_type;
    static constexpr unsigned SWITCH_ID = CONFIG_T::switch_signal;
    static constexpr unsigned STOP_ID = CONFIG_T::stop_signal;
    static constexpr size_t MAX_ITERATIONS =  CONFIG_T::max_iterations;

    [[intel::fpga_register]] host_pipe_T in_data;
    [[intel::fpga_register]] bool switch_to_fb = false;
    [[intel::fpga_register]] SignalPack signal_pack; // TODO - Check memory attribute
    [[intel::fpga_register]] sw_pipe_T out_data_pack;
    [[intel::fpga_register]] sw_pos_pipe_T out_pos_data_pack;
    [[intel::fpga_register]] unsigned writect = 0;
    [[intel::fpga_register]] size_t iter_ct = 0;

    while(true){
        
        if(switch_to_fb){ // Feedback path is a per-token stream so we expect nnet::array<type,1>
            fb_pipe_T inp_id = feedback_pipe::read();
            if (inp_id[0] == STOP_ID || iter_ct + 1 >= MAX_ITERATIONS){
                signal_pack.signal = PipeSignal::stop;
                out_data_pack.exit_task = true;
                out_pos_data_pack.exit_task = true;
                control_pipe::write(signal_pack);
                switch_pipe::write(out_data_pack);
                //switch_pos_pipe::write(out_pos_data_pack);
                return;
            } 
            else if (inp_id[0] == SWITCH_ID){
                switch_to_fb = false;
                signal_pack.signal = PipeSignal::no_feedback;
                control_pipe::write(signal_pack);
            } 
            
            else {
                signal_pack.signal = PipeSignal::feedback;
                control_pipe::write(signal_pack);
            }
            out_data_pack.data = inp_id;
            out_data_pack.exit_task = signal_pack.signal == PipeSignal::stop;
            out_pos_data_pack.data[0] = writect; // Expect nnet::array of size 1
            out_pos_data_pack.exit_task = signal_pack.signal == PipeSignal::stop;
            switch_pipe::write(out_data_pack);
            //switch_pos_pipe::write(out_pos_data_pack);
            writect++;
            iter_ct++;
        }
        else{ // This branch expects a single vector DMA transfer of switch IDs
            in_data = host_pipe::read();
            for(unsigned host_it = 0; host_it < std::tuple_size<host_pipe_T>{}; host_it++){
                data_T el = in_data[host_it];
                if (el == STOP_ID || iter_ct + 1 >= MAX_ITERATIONS){
                    signal_pack.signal = PipeSignal::stop;
                    out_data_pack.exit_task = true;
                    out_pos_data_pack.exit_task = true;
                    control_pipe::write(signal_pack);
                    switch_pipe::write(out_data_pack);
                    //switch_pos_pipe::write(out_pos_data_pack);
                    return;
                } 
                else if (el == SWITCH_ID){
                    switch_to_fb = true;
                    signal_pack.signal = PipeSignal::feedback;
                    control_pipe::write(signal_pack);
                } 
                else {
                    signal_pack.signal = PipeSignal::no_feedback;
                    control_pipe::write(signal_pack);
                }
                out_data_pack.data[0] = el;
                out_data_pack.exit_task = signal_pack.signal == PipeSignal::stop;
                out_pos_data_pack.data[0] = writect; // Expect nnet::array of size 1
                out_pos_data_pack.exit_task = signal_pack.signal == PipeSignal::stop;
                switch_pipe::write(out_data_pack);
                //switch_pos_pipe::write(out_pos_data_pack);
                writect++;
                iter_ct++;
            }
        }
    }
    signal_pack.signal = PipeSignal::stop;
    out_data_pack.exit_task = true;
    out_pos_data_pack.exit_task = true;
    switch_pipe::write(out_data_pack);
    //switch_pos_pipe::write(out_pos_data_pack);
    control_pipe::write(signal_pack);
}
*/

// Returns the maximum value from an array of size N - assumes data is array and output size is always 1 (not an array)
template <typename data_arr_T, unsigned N, typename res_T>
inline res_T argmax_stream(data_arr_T data) {

    constexpr unsigned loops = ceil_log2(N);
    constexpr unsigned padded_len = 1 << loops;

    // Single element requires no comparisons so return straight away
    if constexpr (loops == 0)
        return static_cast<res_T>(data[0]);
    else{
        using idx_arr_T = typename nnet::array<ac_fixed<loops, loops, false>, padded_len>;
        using data_in_T = typename data_arr_T::value_type;

        [[intel::fpga_register]] idx_arr_T maxval_idxs;

        #pragma unroll
        for (unsigned it = 0; it < padded_len; it++) {
            maxval_idxs[it] = it;
        }

        #pragma unroll
        for (unsigned st = 0; st < loops; st++){
            unsigned curr_len = padded_len >> (st+1);
            #pragma unroll loops // TODO - TUNE THIS #######################################################################################################################################################################
            for (unsigned l = 0; l < curr_len; l++) {
                data_in_T data_first = ((2 * l) < N) ?  data[2 * l] : minval<data_in_T>();
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
        static_assert(
        std::is_same_v<data_in_pipe_T, res_pipe_T>, 
        "output_switch: switch_pipe packet type and host_pipe packet type differ"
        );
    }

    while(true){
        data_in_pipe_T data_pack = switch_pipe::read();
        bool exit_task = data_pack.exit_task;
        bool fb = data_pack.feedback;

        if constexpr (CONFIG_T::argmax) {
            res_arr_T maxval_idx{};
            if (!exit_task) {
                maxval_idx[0] = argmax_stream<data_in_arr_T, data_in_arr_size, res_T>(data_pack.data);
                
                if (fb){
                    fb_arr_T out;
                    #pragma unroll
                    for(int i = 0; i < std::tuple_size<fb_arr_T>{}; i++){
                        out[i] = maxval_idx[0];
                    }
                    feedback_pipe::write(out);
                }
            }
            host_pipe::write(res_pipe_T{maxval_idx, exit_task,false});
            if (exit_task) return;
        }
        else{
            if (!exit_task) {
                if (fb){
                    fb_arr_T out;
                    #pragma unroll
                    for(int i = 0; i < std::tuple_size<fb_arr_T>{}; i++){
                        out[i] = data_pack.data[i];
                    }
                    feedback_pipe::write(out);
                }
            }
            host_pipe::write(res_pipe_T{data_pack.data,exit_task,false}); //TODO - We enforce same var type but should i add a static cast just in case?
            if (exit_task) return;
        }
    }
}
} // namespace nnet
#endif
