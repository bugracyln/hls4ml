#ifndef NNET_EMBED_STREAM_H_
#define NNET_EMBED_STREAM_H_

namespace nnet {

template <class data_pipe, class res_pipe, typename CONFIG_T> void embedding_stream() {

#ifdef AUTOREG
    using data_pipe_T = typename ExtractPipeType<data_pipe>::value_type;
    using data_arr_T = typename data_pipe_T::data_type;
    using res_pipe_T = typename ExtractPipeType<res_pipe>::value_type;
    using res_arr_T = typename res_pipe_T::data_type;
    res_pipe_T out_data_pipe;
#else
    using res_arr_T = typename ExtractPipeType<res_pipe>::value_type;
#endif

    constexpr auto datasize = std::tuple_size<data_arr_T>{};
    constexpr unsigned loopnum = ((CONFIG_T::n_in / datasize) > 0) ? (CONFIG_T::n_in / datasize) : 1;

    if constexpr ((datasize == 1) && (loopnum > 1)) {
    InputSequence2:
    #ifdef AUTOREG
        while (true) {
            auto in_data_pipe = data_pipe::read();
            if(in_data_pipe.exit_task){
                out_data_pipe.exit_task = true;
                res_pipe::write(out_data_pipe);
                return;
            }
            auto in_data = in_data_pipe.data;
    #else
        [[intel::initiation_interval(CONFIG_T::reuse_factor)]] for (int j = 0; j < loopnum; j++) {

            auto in_data = data_pipe::read();
    #endif
            res_arr_T res_pack;

        DenseEmbedding2:
            #pragma unroll CONFIG_T::num_banks
            for (int i = 0; i < CONFIG_T::n_out; i++) {
                res_pack[i] = CONFIG_T::embeddings[(in_data[0] * CONFIG_T::n_out + i).to_uint()];
            }
        
        #ifdef AUTOREG
            out_data_pipe.data = res_pack;
            out_data_pipe.exit_task = false;
            res_pipe::write(out_data_pipe);
        #else
            res_pipe::write(res_pack);
        #endif
                    
        }

    } else {
    
    #ifdef AUTOREG
        while (true) {
            auto in_data_pipe = data_pipe::read();
            if(in_data_pipe.exit_task){
                out_data_pipe.exit_task = true;
                res_pipe::write(out_data_pipe);
                return;
            }
            auto in_data = in_data_pipe.data;
    #else
            auto in_data = data_pipe::read();
    #endif

        InputSequence:
            [[intel::initiation_interval(CONFIG_T::reuse_factor)]] for (int j = 0; j < datasize; j++) {

                res_arr_T res_pack;

            DenseEmbedding:
                #pragma unroll CONFIG_T::num_banks
                for (int i = 0; i < CONFIG_T::n_out; i++) {
                    res_pack[i] = CONFIG_T::embeddings[(in_data[j] * CONFIG_T::n_out + i).to_uint()];
                }

            #ifdef AUTOREG
                out_data_pipe.data = res_pack;
                out_data_pipe.exit_task = false;
                res_pipe::write(out_data_pipe);
            #else
                res_pipe::write(res_pack);
            #endif
            }
    #ifdef AUTOREG
        }
    #endif  
    }
}

} // namespace nnet

#endif
