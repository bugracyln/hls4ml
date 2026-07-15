#ifndef AUTOREGRESSIVE_HELPER_H_
#define AUTOREGRESSIVE_HELPER_H_

/*
Global switch to keep track of where we read from,
use seperate switches to avoid pipeline stall issues
*/
#include "nnet_common.h"
#include "nnet_types.h"
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
    PipeSignal signal;
    unsigned writect;
};


/*
//TODO: Check if HLS parallelises each switch_single_pipe instance on its own else look for compiler command
// SwitchSet is a struct with data,feedback,sw pipes
template <class control_pipe, typename CONFIG_T, class... PipeSe> void input_switch() {
    (inp_switch_single_pipe<control_pipe, CONFIG_T, PipeSet>(), ...)
}

template <class control_pipe, class... PipeSet> void output_switch() {
    (out_switch_single_pipe<control_pipe, PipeSet>(), ...)
}*/

//##############################################################################################################################
// SHARE A DEVICE GLOBAL WITH writes for input and reads for output and once we receive the stop or switch signal
// To empty the pipeline simply read (writes - reads) elements on the output switch side
// Create a contol signal line so input can pass the signals to output directly 
//##############################################################################################################################

// Returns the maximum value from an array of size N - assumes data is array and output size is always 1 (not an array)
template <typename data_arr_T, int N, typename res_arr_T> res_arr_T argmax_stream(data_arr_T data) {

    constexpr unsigned loops = ceil_log2(N);
    constexpr unsigned padded_len = 1 << loops;

    using data_in_T = typename data_arr_T::value_type;
    using idx_arr_T = typename nnet::array<ac_fixed<loops, loops, false>, padded_len>;
    using res_T = typename res_arr_T::value_type;

    // Single element requires no pooling so return straight away
    if constexpr (loops == 0) return static_cast<res_T>(data[0]);

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

    maxval_idx = maxval_idxs[0];
    return maxval_idx;
}

// We expect nnet::array<arr_T, 1> so a size 1 array for token feedback.
template <class host_pipe, class feedback_pipe, class switch_pipe, class control_pipe, typename CONFIG_T> void inp_switch() {
    
    using pipe_T = typename ExtractPipeType<host_pipe>::value_type;
    using data_T = typename pipe_T::value_type;
    using sw_pipe_T = typename ExtractPipeType<switch_pipe>::value_type;
    constexpr data_T SWITCH_ID = CONFIG_T::switch_signal;
    constexpr data_T STOP_ID = CONFIG_T::stop_signal;

    [[intel::fpga_register]] pipe_T in_data;
    [[intel::fpga_register]] bool switch_to_fb = false;
    [[intel::fpga_memory]] SignalPack signal_pack; // TODO - Check memory attribute

    while(true){

        if(switch_to_fb){
            in_data = feedback_pipe::read();
            if (in_data[0] == STOP_ID){
                signal_pack.signal = PipeSignal::stop;
                control_pipe::write(signal_pack);
                break;
            } 
            else if (in_data[0] == SWITCH_ID){
                switch_to_fb = false;
                signal_pack.signal = PipeSignal::no_feedback;
                control_pipe::write(signal_pack);
            } 
            else {
                signal_pack.signal = PipeSignal::feedback;
                control_pipe::write(signal_pack);
            }
        }
        else{   
            in_data = host_pipe::read();
            if (in_data[0] == STOP_ID){
                signal_pack.signal = PipeSignal::stop;
                control_pipe::write(signal_pack);
                break;
            } 
            else if (in_data[0] == SWITCH_ID){
                switch_to_fb = true;
                signal_pack.signal = PipeSignal::feedback;
                control_pipe::write(signal_pack);
            } 
            else {
                signal_pack.signal = PipeSignal::no_feedback;
                control_pipe::write(signal_pack);
            }
        }

        sw_pipe_T out_data_pack;
        out_data_pack.data = in_data;
        out_data_pack.exit_task = signal_pack.signal == PipeSignal::stop;

        switch_pipe::write(out_data_pack);
        signal_pack.writect++;
    }
}

// Overloaded with the position index, required by token streaming models
template <class host_pipe, class feedback_pipe, class switch_pipe, class switch_pos_pipe, class control_pipe, typename CONFIG_T> void input_switch() {

    using pipe_T = typename ExtractPipeType<host_pipe>::value_type;
    using data_T = typename pipe_T::value_type;
    using sw_pipe_T = typename ExtractPipeType<switch_pipe>::value_type;
    using sw_pos_pipe_T = typename ExtractPipeType<switch_pos_pipe>::value_type;
    static constexpr data_T SWITCH_ID = CONFIG_T::switch_signal;
    static constexpr data_T STOP_ID = CONFIG_T::stop_signal;

    [[intel::fpga_register]] pipe_T in_data;
    [[intel::fpga_register]] bool switch_to_fb = false;
    [[intel::fpga_memory]] SignalPack signal_pack; // TODO - Check memory attribute

    while(true){
        if(switch_to_fb){
            in_data = feedback_pipe::read();
            signal_pack.writect++;
            
            if (in_data[0] == STOP_ID){
                signal_pack.signal = PipeSignal::stop;
                control_pipe::write(signal_pack);
                break;
            } 
            else if (in_data[0] == SWITCH_ID){
                switch_to_fb = false;
                signal_pack.signal = PipeSignal::no_feedback;
                control_pipe::write(signal_pack);
            } 
            else {
                signal_pack.signal = PipeSignal::feedback;
                control_pipe::write(signal_pack);
            }
        }
        else{   
            in_data = host_pipe::read();
            signal_pack.writect++;

            if (in_data[0] == STOP_ID){
                signal_pack.signal = PipeSignal::stop;
                control_pipe::write(signal_pack);
                break;
            } 
            else if (in_data[0] == SWITCH_ID){
                switch_to_fb = true;
                signal_pack.signal = PipeSignal::feedback;
                control_pipe::write(signal_pack);
            } 
            else {
                signal_pack.signal = PipeSignal::no_feedback;
                control_pipe::write(signal_pack);
            }
        }
        sw_pipe_T out_data_pack;
        out_data_pack.data = in_data;
        out_data_pack.exit_task = signal_pack.signal == PipeSignal::stop;

        sw_pos_pipe_T out_pos_data_pack;
        out_pos_data_pack.data[0] = signal_pack.writect; // Expect nnet::array of size 1
        out_pos_data_pack.exit_task = signal_pack.signal == PipeSignal::stop;

        switch_pipe::write(out_data_pack);
        switch_pos_pipe::write(out_pos_data_pack);
    }
}

template <class host_pipe, class feedback_pipe, class switch_pipe, class control_pipe, typename CONFIG_T> void output_switch() {

    using control_T = typename ExtractPipeType<control_pipe>::value_type;
    using data_in_pipe_T = typename ExtractPipeType<switch_pipe>::value_type;
    using data_in_arr_T = typename data_in_pipe_T::data_type;
    using data_in_T = typename data_in_arr_T::value_type;
    static constexpr unsigned data_in_arr_size = std::tuple_size<data_in_arr_T>{};
    using res_arr_T = typename ExtractPipeType<host_pipe>::value_type;
    static constexpr bool ARGMAX = CONFIG_T::argmax;
    
    PipeSignal prev_signal;
    unsigned readct = 0;
    res_arr_T maxval_idx;
    
    
    while (true) {

        // Read the control signal first
        control_T control_pack = control_pipe::read();
        PipeSignal control_signal = control_pack.signal;

        // Empty pipeline and exit 
        if (control_signal == PipeSignal::stop){
            for(unsigned i = 0; i < (control_pack.writect - readct); i++){
                data_in_pipe_T data_pack = switch_pipe::read();
                if constexpr (ARGMAX){
                    maxval_idx[0] = argmax_stream<data_in_arr_T, data_in_arr_size, res_arr_T>(data_pack.data);
                    host_pipe::write(maxval_idx);
                }
                else{
                    host_pipe::write(data_pack.data);
                } 
            }
            break;
        }
        else {
            if (prev_signal != control_signal){
                for(; readct < control_pack.writect; readct++){

                    data_in_pipe_T data_pack = switch_pipe::read();

                    if constexpr (ARGMAX) {
                        maxval_idx[0] = argmax_stream<data_in_arr_T, data_in_arr_size, res_arr_T>(data_pack.data);
                        host_pipe::write(maxval_idx);
                        if (prev_signal == PipeSignal::feedback){
                            feedback_pipe::write(maxval_idx);
                        }      
                    }
                    else{
                        host_pipe::write(data_pack.data);
                        if (prev_signal == PipeSignal::feedback){
                            feedback_pipe::write(data_pack.data);
                        }                       
                    }  
                                   
                }
            }

            // TODO - Check readct and writect logic dont make off by one mistake ensure we empty the pipe first and read the last one
            // so you might need for(; readct < control_pack.writect - 1; readct++) loop instead DOUBLE CHECK THIS
            data_in_pipe_T data_pack = switch_pipe::read();
            
            if constexpr (ARGMAX) {
                maxval_idx[0] = argmax_stream<data_in_arr_T, data_in_arr_size, res_arr_T>(data_pack.data);
                host_pipe::write(maxval_idx);
                if (control_signal == PipeSignal::feedback){
                    feedback_pipe::write(maxval_idx);
                }
            }
            else{
                host_pipe::write(data_pack.data);

                if (control_signal == PipeSignal::feedback){
                    feedback_pipe::write(data_pack.data);
                }
            }
            readct++;
        }

        prev_signal = control_signal;
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
