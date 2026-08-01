#ifndef AUTOREGRESSIVE_HELPER_H_
#define AUTOREGRESSIVE_HELPER_H_

/*
Global switch to keep track of where we read from,
use seperate switches to avoid pipeline stall issues
*/
#include "nnet_common.h"
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


// We expect nnet::array<arr_T, 1> so a size 1 array for token feedback.
// This unit also converts a vector to token id stream.
template <class host_pipe, class feedback_pipe, class switch_pipe, class control_pipe, typename CONFIG_T> void input_switch() {
    
    using host_pipe_T = typename ExtractPipeType<host_pipe>::value_type;
    using fb_pipe_T = typename ExtractPipeType<feedback_pipe>::value_type;
    using data_T = typename host_pipe_T::value_type;
    using sw_pipe_T = typename ExtractPipeType<switch_pipe>::value_type;
    using sw_arr_T = typename sw_pipe_T::data_type;
    static constexpr unsigned SWITCH_ID = CONFIG_T::switch_signal;
    static constexpr unsigned STOP_ID = CONFIG_T::stop_signal;
    static constexpr size_t MAX_ITERATIONS =  CONFIG_T::max_iterations;

    [[intel::fpga_register]] host_pipe_T in_data;
    [[intel::fpga_register]] bool switch_to_fb = false;
    [[intel::fpga_register]] SignalPack signal_pack; // TODO - Check memory attribute
    [[intel::fpga_register]] sw_pipe_T out_data_pack;
    [[intel::fpga_register]] size_t iter_ct = 0;

    while(true){

        if(switch_to_fb){ // Feedback path is a per-token stream so we expect nnet::array<type,1>
            fb_pipe_T inp_id = feedback_pipe::read();
            if (inp_id[0] == STOP_ID || iter_ct + 1 >= MAX_ITERATIONS){
                signal_pack.signal = PipeSignal::stop;
                out_data_pack.exit_task = true;
                control_pipe::write(signal_pack);
                switch_pipe::write(out_data_pack);
                return;
            } 
            /*
            else if (inp_id[0] == SWITCH_ID){
                switch_to_fb = false;
                signal_pack.signal = PipeSignal::no_feedback;
                control_pipe::write(signal_pack);
            }
            */
            else {
                signal_pack.signal = PipeSignal::feedback;
                control_pipe::write(signal_pack);
            }
            out_data_pack.data = inp_id;
            out_data_pack.exit_task = signal_pack.signal == PipeSignal::stop;
            switch_pipe::write(out_data_pack);
            iter_ct++;
        }
        else{ // This branch expects a single vector DMA transfer of switch IDs, serialises them
            in_data = host_pipe::read();
            for(unsigned host_it = 0; host_it < std::tuple_size<host_pipe_T>{}; host_it++){
                data_T el = in_data[host_it];
                if (el == STOP_ID || iter_ct + 1 >= MAX_ITERATIONS){
                    signal_pack.signal = PipeSignal::stop;
                    out_data_pack.exit_task = true;
                    control_pipe::write(signal_pack);
                    switch_pipe::write(out_data_pack);
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
                switch_pipe::write(out_data_pack);
                iter_ct++;
            }
        }
    }
    signal_pack.signal = PipeSignal::stop;
    out_data_pack.exit_task = true;
    switch_pipe::write(out_data_pack);
    control_pipe::write(signal_pack);
}

// Overloaded with the position index, required by token streaming models
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
            /*
            else if (inp_id[0] == SWITCH_ID){
                switch_to_fb = false;
                signal_pack.signal = PipeSignal::no_feedback;
                control_pipe::write(signal_pack);
            } 
            */
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


// Returns the maximum value from an array of size N - assumes data is array and output size is always 1 (not an array)
template <typename data_arr_T, unsigned N, typename res_arr_T> res_arr_T argmax_stream(data_arr_T data) {

    constexpr unsigned loops = ceil_log2(N);
    constexpr unsigned padded_len = 1 << loops;

    using res_T = typename res_arr_T::value_type;

    res_arr_T out;

    // Single element requires no comparisons so return straight away
    if constexpr (loops <= 1)
        out[0] = static_cast<res_T>(data[0]);
    else{
        using idx_arr_T = typename nnet::array<ac_fixed<loops, loops, false>, padded_len>;
        using data_in_T = typename data_arr_T::value_type;

        idx_arr_T maxval_idxs;
        res_T maxval_idx = 0;

        #pragma unroll
        for (unsigned it = 0; it < padded_len; it++) {
            maxval_idxs[it] = it;
        }

        #pragma unroll
        for (unsigned arr_len = padded_len; arr_len > 1; arr_len /= 2) {
            unsigned curr_len = arr_len / 2;
            #pragma unroll
            for (unsigned l = 0; l < curr_len; l++) {
                data_in_T data_first = ((2 * l) < N) ?  data[2 * l] : minval<data_in_T>();
                data_in_T data_second = ((2 * l + 1) < N) ? data[2 * l + 1] : minval<data_in_T>();
                data[l] = (data_first > data_second) ? data_first : data_second;
                maxval_idxs[l] = (data_first > data_second) ? maxval_idxs[2 * l] : maxval_idxs[2 * l + 1];
            }
        }
        out[0] = maxval_idxs[0];
    }
    return out;
}


template <class host_pipe, class feedback_pipe, class switch_pipe, class control_pipe, typename CONFIG_T> void output_switch() {

    static constexpr size_t MAX_ITERATIONS = CONFIG_T::max_iterations;

    using control_T = typename ExtractPipeType<control_pipe>::value_type;
    using data_in_pipe_T = typename ExtractPipeType<switch_pipe>::value_type;
    using data_in_arr_T = typename data_in_pipe_T::data_type;
    using fb_arr_T = typename ExtractPipeType<feedback_pipe>::value_type;
    static constexpr unsigned data_in_arr_size = std::tuple_size<data_in_arr_T>{};
    using res_pipe_T = typename ExtractPipeType<host_pipe>::value_type;
    using res_arr_T = typename res_pipe_T::data_type;

    if constexpr (!CONFIG_T::argmax) {
        static_assert(
        std::is_same_v<data_in_pipe_T, res_pipe_T>, 
        "output_switch: switch_pipe packet type and host_pipe packet type differ"
        );
    }

    res_pipe_T out_data_packet;
    res_arr_T maxval_idx;
    
    [[intel::fpga_register]] size_t iter_ct = 0;

    while(true){
        // Read the control signal first
        control_T control_pack = control_pipe::read();
        PipeSignal control_signal = control_pack.signal;

        data_in_pipe_T data_pack = switch_pipe::read();

        if (control_signal == PipeSignal::stop || iter_ct >= MAX_ITERATIONS) {
            //switch_pipe::read(); // Wait for EOS packet (stale discard to empty the pipe)
            out_data_packet.exit_task = true;
            host_pipe::write(out_data_packet);
            return;
        }
            
        if constexpr (CONFIG_T::argmax) {
            maxval_idx = argmax_stream<data_in_arr_T, data_in_arr_size, res_arr_T>(data_pack.data);
            
            if (control_signal == PipeSignal::feedback){
                fb_arr_T out;
                #pragma unroll
                for(int i = 0; i < std::tuple_size<fb_arr_T>{}; i++){
                    out[i] = maxval_idx[0];
                }
                feedback_pipe::write(out);
            }
    
            out_data_packet.data = maxval_idx;
            out_data_packet.exit_task = false;
            host_pipe::write(out_data_packet);
        }
        else{

            if (control_signal == PipeSignal::feedback){
                fb_arr_T out;
                #pragma unroll
                for(int i = 0; i < std::tuple_size<fb_arr_T>{}; i++){
                    out[i] = data_pack.data[i];
                }
                feedback_pipe::write(out);
            }

            host_pipe::write(data_pack); //TODO - We enforce same var type but should i add a static cast just in case?
        }
        iter_ct++;
    }
}


/*
// Maxval helper - TODO: Find if there is alr implemented equivalent
template<class data_pipe, class res_pipe, unsigned n_in> void findmax_stream(){

    using data_in_arr_T = typename ExtractPipeType<data_pipe>::value_type;
    using res_T = typename ExtractPipeType<res_pipe>::value_type; // Assumes pipe is not an array but a single element

    constexpr unsigned n_data_in = std::tuple_size<data_in_arr_T>{};
    constexpr unsigned loops = ceil_log2(n_data_in);
    using idx_arr_T = typename nnet::array<ac_fixed<loops, loops, false>, n_data_in>;

    idx_arr_T maxval_idxs;
    #pragma unroll
    for (unsigned it = 0; it < n_data_in; it++) {
        maxval_idxs[it] = it;
    }
    
    data_in_arr_T data = data_pipe::read();

    #pragma unroll
    for (unsigned arr_len = n_data_in; arr_len > 1; arr_len /= 2) {
        unsigned curr_len = arr_len / 2;
        #pragma unroll
        for (unsigned l = 0; l < curr_len; l++) {
            data_in_T data_first = data[2 * l];
            data_in_T data_second = data[2 * l + 1];
            data[l] = (data_first > data_second) ? data_first : data_second;
            maxval_idxs[l] = (data_first > data_second) ? maxval_idxs[2 * l] : maxval_idxs[2 * l + 1];
        }
    }
    maxval_idx = maxval_idxs[0];
    res_pipe::write(static_cast<res_T>(maxval_idx));
}
*/

/*
template <class PipeSet, typename CONFIG_T> void inp_switch_single_pipe() {

    constexpr unsigned N_PREFILL = CONFIG_T::n_prefill;
    constexpr unsigned N_GENERATE = CONFIG_T::n_generate;
    constexpr unsigned N_TOTAL = N_PREFILL + N_GENERATE;

    using data_T = typename ExtractPipeType<typename PipeSet::host>::value_type;
    [[intel::fpga_register]] data_T in_data;

    for (unsigned i = 0; i < N_TOTAL; i++) {
        if (i >= N_PREFILL)
            in_data = PipeSet::feedback::read();
        else
            in_data = PipeSet::host::read();
        PipeSet::sw::write(in_data);
    }
}
*/

/*
template <class data_in_pipe, class loopback_emb_pipe, class loopback_pos_pipe, class output_pipe, typename CONFIG_T>
void output_switch() {

    constexpr unsigned N_PREFILL = CONFIG_T::n_prefill;
    constexpr unsigned N_GENERATE = CONFIG_T::n_generate;
    constexpr unsigned N_TOTAL = N_PREFILL + N_GENERATE;

    using data_in_arr_T = typename ExtractPipeType<data_in_pipe>::value_type;
    using data_in_T = typename data_in_arr_T::value_type;

    constexpr unsigned n_data_in = std::tuple_size<data_in_arr_T>{};
    constexpr unsigned loops = ceil_log2(n_data_in);
    using idx_T = ac_fixed<loops, loops, false>;
    using idx_arr_T = typename nnet::array<ac_fixed<loops, loops, false>, n_data_in>;

    // this is fine since we feed a single token
    using emb_arr_T = typename ExtractPipeType<loopback_emb_pipe>::value_type;
    using pos_arr_T = typename ExtractPipeType<loopback_pos_pipe>::value_type;
    using emb_T = typename emb_arr_T::value_type;
    using pos_T = typename pos_arr_T::value_type;

    [[intel::fpga_register]] emb_arr_T emb_arr;
    [[intel::fpga_register]] pos_arr_T pos_arr;

    idx_arr_T maxval_idxs;
    #pragma unroll
    for (unsigned it = 0; it < n_data_in; it++) {
        maxval_idxs[it] = it;
    }

    idx_T maxval_idx = 0;

    for (unsigned i = 0; i < N_TOTAL; i++) {

        data_in_arr_T data = data_in_pipe::read();

        #pragma unroll
        for (unsigned arr_len = n_data_in; arr_len > 1; arr_len /= 2) {
            unsigned curr_len = arr_len / 2;
            #pragma unroll
            for (unsigned l = 0; l < curr_len; l++) {
                data_in_T data_first = data[2 * l];
                data_in_T data_second = data[2 * l + 1];
                data[l] = (data_first > data_second) ? data_first : data_second;
                maxval_idxs[l] = (data_first > data_second) ? maxval_idxs[2 * l] : maxval_idxs[2 * l + 1];
            }
        }

        maxval_idx = maxval_idxs[0];
        for (unsigned j = 0; j < std::tuple_size<emb_arr_T>{}; j++) {
            emb_arr[j] = static_cast<emb_T>(maxval_idx);
            pos_arr[j] = static_cast<pos_T>(i + 1);
        }

        // Write the next token ID instead
        output_pipe::write(emb_arr);

        if ((i >= N_PREFILL - 1) && (i < N_TOTAL - 1)) {
            // guaranteed to be a single item; arrays are for formatting
            loopback_emb_pipe::write(emb_arr);
            loopback_pos_pipe::write(pos_arr);
        }
    }
}
*/

} // namespace nnet
#endif
