#ifndef AUTOREGRESSIVE_HELPER_H_
#define AUTOREGRESSIVE_HELPER_H_

/*
Global switch to keep track of where we read from,
use seperate switches to avoid pipeline stall issues
*/

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

struct SwitchCounters {
    unsigned write;
    unsigned read;
};

struct StopSignal {
    static constexpr unsigned stop = PipeSignal::stop;
    unsigned writect;
};

template<class switch_pair>
device_global<SwitchCounters, decltype(properties(device_image_scope, host_access_none))>
    switch_counters;

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

template <class host_pipe, class feedback_pipe, class switch_pipe, class control_pipe, typename CONFIG_T> void inp_switch() {

    using pipe_T = typename ExtractPipeType<host_pipe>::value_type;
    using data_T = typename pipe_T::value_type;
    constexpr data_T SWITCH_ID = CONFIG_T::switch_signal;
    constexpr data_T STOP_ID = CONFIG_T::stop_signal;

    [[intel::fpga_register]] data_T in_data;
    [[intel::fpga_register]] bool switch_to_fb = false;

    // Device global to share counters between switches
    auto &counters = switch_counters<switch_pipe>.get();
    auto &writect = counters.write;

    while(true){

        if(switch_to_fb){
            in_data = feedback_pipe::read();
            if (in_data == STOP_ID){
                control_pipe::write(PipeSignal::stop);
                break;
            } 
            else if (in_data == SWITCH_ID){
                switch_to_fb = false;
                control_pipe::write(PipeSignal::no_feedback);
            } 
            else control_pipe::write(PipeSignal::feedback);
        }
        else{   
            in_data = host_pipe::read();
            if (in_data == STOP_ID){
                control_pipe::write(PipeSignal::stop);
                break;
            } 
            else if (in_data == SWITCH_ID){
                switch_to_fb = true;
                control_pipe::write(PipeSignal::feedback);
            } 
            else control_pipe::write(PipeSignal::no_feedback);
        }
        switch_pipe::write(in_data);
        writect++;
    }
}

template <class host_pipe, class feedback_pipe, class switch_pipe, class control_pipe> void out_switch() {

    using control_T = typename ExtractPipeType<control_pipe>::value_type;
    using data_in_arr_T = typename ExtractPipeType<typename switch_pipe>::value_type;
    using data_in_T = typename data_in_arr_T::value_type;

    // Device global to share counters between switches
    auto &counters = switch_counters<switch_pipe>.get();
    auto &writect = counters.write;
    auto &readct = counters.read;
    
    while (true) {

        // Read the control signal first
        control_T control_signal = control_pipe::read();

        // Empty pipeline and exit 
        if (control_signal == PipeSignal::stop){
            for(unsigned i = 0; i < (writect - readct); i++){
                data_in_arr_T data = switch_pipe::read();
                host_pipe::write(data);
            }
            break;
        }
        else {
            data_in_arr_T data = switch_pipe::read();
            readct++;
            host_pipe::write(data);

            if (control_signal == PipeSignal::feedback){
                feedback_pipe::write(data);
            }
        }
    }
}

/*
template <class control_pipe, typename CONFIG_T, class PipeSet> void inp_switch() {

    using pipe_T = typename ExtractPipeType<typename PipeSet::host>::value_type;
    using data_T = typename pipe_T::value_type;
    constexpr data_T SWITCH_ID = CONFIG_T::switch_signal;
    constexpr data_T STOP_ID = CONFIG_T::stop_signal;

    [[intel::fpga_register]] data_T in_data;
    [[intel::fpga_register]] bool switch_to_fb = false;


    // Device global to share counters between switches
    auto &counters = switch_counters.get();
    auto &writect = counters.write;

    while(true){
        if(switch_to_fb){
            in_data = PipeSet::feedback::read();
            if (in_data == STOP_ID){
                control_pipe::write(PipeSignal::stop)
                break;
            } 
            else if (in_data == SWITCH_ID){
                switch_to_fb = false;
                control_pipe::write(PipeSignal::no_feedback);
            } 
            else control_pipe::write(PipeSignal::feedback);
        }
        else{   
            in_data = PipeSet::host::read();
            if (in_data == STOP_ID){
                control_pipe::write(PipeSignal::stop);
                break;
            } 
            else if (in_data == SWITCH_ID){
                switch_to_fb = true;
                control_pipe::write(PipeSignal::feedback);
            } 
            else control_pipe::write(PipeSignal::no_feedback);
        }
        PipeSet::sw::write(in_data);
        writect++;
    }
}

template <class control_pipe, class PipeSet> void out_switch() {

    using control_T = typename ExtractPipeType<control_pipe>::value_type;
    using data_in_arr_T = typename ExtractPipeType<typename PipeSet::sw>::value_type;
    using data_in_T = typename data_in_arr_T::value_type;

    // Device global to share counters between switches
    auto &counters = switch_counters.get();
    auto &writect = counters.write;
    auto &readct = counters.read;
    
    while (true) {

        // Read the control signal first
        control_T control_signal = control_pipe::read();

        // Empty pipeline and exit 
        if (control_signal == PipeSignal::stop){
            for(unsigned i = 0; i < (writect - readct); i++){
                data_in_arr_T data = PipeSet::sw::read();
                PipeSet::host::write(data);
            }
            break;
        }
        else {
            data_in_arr_T data = PipeSet::sw::read();
            readct++;
            PipeSet::host::write(data);

            if (control_signal == PipeSignal::feedback){
                PipeSet::feedback::write(data);
            }
        }
    }
}
*/


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
    res_pipe::write(static_cast<res_T>(maxval_idx));
}

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
