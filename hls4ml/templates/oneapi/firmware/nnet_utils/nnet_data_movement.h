#ifndef NNET_DATA_MOVEMENT_H
#define NNET_DATA_MOVEMENT_H

#ifdef AHLS
#include <sycl/ext/altera/fpga_extensions.hpp>
#else
#include <sycl/ext/intel/fpga_extensions.hpp>
#endif
#include <sycl/sycl.hpp>
#include "nnet_utils/nnet_printf.h"

// This file defines the methods to transfer the data to the kernel. In the HLS flow,
// these are really part of the testbench. However, in the accelerator (BSP) flow, they are
// actual kernels that are deployed in hardware.

namespace nnet {
//////////////////////////////////////////////////////////////////////////////
// These are the simple, testbench and bridge versions for the HLS flow
//////////////////////////////////////////////////////////////////////////////
/*
template <class srcType, class dest_pipe, size_t SIZE> void convert_data(sycl::queue &q, srcType *src) {
    constexpr auto dstTypeSize = std::tuple_size<typename ExtractPipeType<dest_pipe>::value_type>{};
    for (size_t i = 0; i < SIZE / dstTypeSize; i++) {
        typename ExtractPipeType<dest_pipe>::value_type ctype;
        for (size_t j = 0; j < dstTypeSize; j++) {
            ctype[j] = src[i * dstTypeSize + j];
        }
        dest_pipe::write(q, ctype);
    }
}

template <class src_pipe, class dstType, size_t SIZE> void convert_data_back(sycl::queue &q, dstType *dst) {
    constexpr auto srcTypeSize = std::tuple_size<typename ExtractPipeType<src_pipe>::value_type>{};
    for (size_t i = 0; i < SIZE / srcTypeSize; i++) {
        auto ctype = src_pipe::read(q);
        for (size_t j = 0; j < srcTypeSize; j++) {
            dst[i * srcTypeSize + j] = ctype[j].to_double();
        }
    }
}
*/

//////////////////////////////////////////////////////////////////////////////
// The ones below can be used both in testbenches and in the accelerator flow
//////////////////////////////////////////////////////////////////////////////
#if !defined(IS_BSP)
// Definition for buffer locations for Avalon MM host.
inline constexpr unsigned kInputBufferLocation = 0;
inline constexpr unsigned kOutputBufferLocation = 1;
#endif

template <class src_T, class Pipe> struct SrcPipePair {

    using source_type = src_T;
    using pipe_type = Pipe;

#if !defined(IS_BSP)
    sycl::ext::oneapi::experimental::annotated_arg<
        src_T *,
        decltype(sycl::ext::oneapi::experimental::properties{
        #ifdef AHLS
            sycl::ext::altera::experimental::latency<0>, sycl::ext::altera::experimental::dwidth<16>,
            sycl::ext::altera::experimental::buffer_location<kInputBufferLocation>,
            sycl::ext::altera::experimental::read_write_mode_read, sycl::ext::altera::experimental::wait_request_requested})>
        #else
            sycl::ext::intel::experimental::latency<0>, sycl::ext::intel::experimental::dwidth<16>,
            sycl::ext::intel::experimental::buffer_location<kInputBufferLocation>,
            sycl::ext::intel::experimental::read_write_mode_read, sycl::ext::intel::experimental::wait_request_requested})>
        #endif
        src;
#else
    src_T *const src;
#endif
};

// For any number of inputs
template <class... SrcPipePairs> struct DMA_convert_data : SrcPipePairs... {

    size_t total_inp_size;

    [[intel::kernel_args_restrict]] void operator()() const { (process_pipe<SrcPipePairs>(), ...); }

  private:
    template <class SrcPipePair> void process_pipe() const {

        using src_T = typename SrcPipePair::source_type;
        using dest_pipe = typename SrcPipePair::pipe_type;
        using PipeDataType = typename nnet::ExtractPipeType<dest_pipe>::value_type;
        constexpr size_t PacketSize = std::tuple_size<PipeDataType>{};
        auto src = static_cast<SrcPipePair const &>(*this).src;

        size_t num_packets = total_inp_size / PacketSize;
        num_packets = (num_packets >= 1) ? num_packets : 1;

#if defined(IS_BSP)
        sycl::ext::altera::host_ptr<src_T> src_ptr(src);
#else
        src_T *src_ptr = src;
#endif

        [[intel::fpga_register]] PipeDataType packet;

        for (size_t i = 0; i < num_packets; ++i) {
            #pragma unroll
            for (size_t j = 0; j < PacketSize; j++) {
                packet[j] = src_ptr[i * PacketSize + j];
            }
            dest_pipe::write(packet);
        }
    }
};

template <class src_T, class dest_pipe> struct DMA_convert_data_single {
#if !defined(IS_BSP)
    // When targeting a device family, we instantiate an Avalon Memory Mapped Host for
    // data transaction between host and the DMA kernel during emulation and simulation.
    sycl::ext::oneapi::experimental::annotated_arg<
        src_T *,
        decltype(sycl::ext::oneapi::experimental::properties{
        #ifdef AHLS
            sycl::ext::altera::experimental::latency<0>, sycl::ext::altera::experimental::dwidth<16>,
            sycl::ext::altera::experimental::buffer_location<kInputBufferLocation>,
            sycl::ext::altera::experimental::read_write_mode_read, sycl::ext::altera::experimental::wait_request_requested
        #else
            sycl::ext::intel::experimental::latency<0>, sycl::ext::intel::experimental::dwidth<16>,
            sycl::ext::intel::experimental::buffer_location<kInputBufferLocation>,
            sycl::ext::intel::experimental::read_write_mode_read, sycl::ext::intel::experimental::wait_request_requested
        #endif
        })>
#else
    // When targeting oneAPI BSP, we can use USM pointer to access host memory.
    src_T *const
#endif
    src;

    size_t num_iteration;

    [[intel::kernel_args_restrict]] void operator()() const {

#if defined(IS_BSP)
        // Access data using host pointer.
        sycl::ext::altera::host_ptr<src_T> src_ptr(src);
#else
        // Host allocation is not supported when targeting an FPGA family or part number.
        src_T *src_ptr(src);
#endif
        // First, extract the PipeDataT from the pipe
        using PipeDataType = typename nnet::ExtractPipeType<dest_pipe>::value_type;
        // By definition, both must have the same size
	    constexpr auto dstTypeSize = std::tuple_size<PipeDataType>{};

        [[intel::fpga_register]] PipeDataType packet;

        // Keep sending data to the input layer and keep the kernels running.
        for (size_t i = 0; i < num_iteration; i++) {
            
	        #pragma unroll
            for (size_t j = 0; j < dstTypeSize; j++) {
                packet[j] = src_ptr[i * dstTypeSize + j];
            }
            dest_pipe::write(packet);
        }
    }
};


// Symmetrical to the DMA_convert_data above, this DMA drains the output pipe and
// writes result to memory.
template <class src_pipe, class dst_T, class ttft_flag_T, class txct_T> struct DMA_convert_data_back {
#if !defined(IS_BSP)
    // Without BSP, instantiate an Avalon Memory Mapped Host to write to host.
    sycl::ext::oneapi::experimental::annotated_arg<
        dst_T *, decltype(sycl::ext::oneapi::experimental::properties{
                #ifdef AHLS
                     sycl::ext::altera::experimental::latency<0>, sycl::ext::altera::experimental::dwidth<16>,
                     sycl::ext::altera::experimental::buffer_location<kOutputBufferLocation>,
                     sycl::ext::altera::experimental::read_write_mode_write,
                     sycl::ext::altera::experimental::wait_request_requested})>
                #else
                     sycl::ext::intel::experimental::latency<0>, sycl::ext::intel::experimental::dwidth<16>,
                     sycl::ext::intel::experimental::buffer_location<kOutputBufferLocation>,
                     sycl::ext::intel::experimental::read_write_mode_write,
                     sycl::ext::intel::experimental::wait_request_requested})>
                #endif
#else
    // USM pointer, otherwise.
    dst_T *const
#endif
        dst;

    volatile ttft_flag_T *const ttft_flag;

    volatile txct_T *const tx_counter;

    size_t num_packs;

    [[intel::kernel_args_restrict]] void operator()() const {
#if defined(IS_BSP)
        sycl::ext::altera::host_ptr<dst_T> dst_ptr(dst);
#else
        dst_T *dst_ptr(dst);
#endif
        // First, extract the PipeDataT from the pipe
        using PipeDataType = typename nnet::ExtractPipeType<src_pipe>::value_type;
        // Then, extract the DataT from StreamingBeat
    #ifdef AUTOREG
        constexpr auto srcTypeSize = std::tuple_size<typename PipeDataType::data_type>{};
    #else
        constexpr auto srcTypeSize = std::tuple_size<PipeDataType>{};
    #endif

        [[intel::fpga_register]] PipeDataType packet;
        [[intel::fpga_register]] txct_T txct = 0;

        auto write_to_host = [&](size_t i, const PipeDataType& pack){
            #pragma unroll 4
            for (size_t j = 0; j < srcTypeSize; j++) {
            #ifdef AUTOREG
                dst_ptr[i * srcTypeSize + j] = static_cast<dst_T>(pack.data[j].to_double());
            #else
                dst_ptr[i * srcTypeSize + j] = static_cast<dst_T>(pack[j].to_double());
            #endif  
            }
        };

    #ifdef AUTOREG
        packet = src_pipe::read();  
        if (packet.exit_task) {
            *ttft_flag = 1;
            *tx_counter = txct + 1;
            return;
        }
        write_to_host(0, packet);
        *ttft_flag = 1;
        txct = 1;
        size_t i = (num_packs > 1) ? 1 : 0;

        while (true){
            packet = src_pipe::read();  
            if (packet.exit_task) break;
            txct++;
            write_to_host(i, packet);
            i = ((i + 1) >= num_packs) ? (i + 1 - num_packs) : (i + 1);
        }
        *tx_counter = txct + 1;
    #else
        if (num_packs > 0){
            packet = src_pipe::read();  
            write_to_host(0, packet);
            *ttft_flag = 1;
            txct = 1;
        }
        // Drain the output pipe and write result to memory.
        for (size_t i = 1; i < num_packs; i++) {
            packet = src_pipe::read();  
            txct++;
            write_to_host(i, packet);
        }
        *tx_counter = txct + 1;
    #endif    
    }  
};

template <class src_pipe, class dst_T> struct DMA_convert_data_back_bridge_ver {
#if !defined(IS_BSP)
    // Without BSP, instantiate an Avalon Memory Mapped Host to write to host.
    sycl::ext::oneapi::experimental::annotated_arg<
        dst_T *, decltype(sycl::ext::oneapi::experimental::properties{
                #ifdef AHLS
                     sycl::ext::altera::experimental::latency<0>, sycl::ext::altera::experimental::dwidth<16>,
                     sycl::ext::altera::experimental::buffer_location<kOutputBufferLocation>,
                     sycl::ext::altera::experimental::read_write_mode_write,
                     sycl::ext::altera::experimental::wait_request_requested})>
                #else
                     sycl::ext::intel::experimental::latency<0>, sycl::ext::intel::experimental::dwidth<16>,
                     sycl::ext::intel::experimental::buffer_location<kOutputBufferLocation>,
                     sycl::ext::intel::experimental::read_write_mode_write,
                     sycl::ext::intel::experimental::wait_request_requested})>
                #endif
#else
    // USM pointer, otherwise.
    dst_T *const
#endif
        dst;

    size_t num_packs;

    [[intel::kernel_args_restrict]] void operator()() const {
#if defined(IS_BSP)
        sycl::ext::altera::host_ptr<dst_T> dst_ptr(dst);
#else
        dst_T *dst_ptr(dst);
#endif

    // First, extract the PipeDataT from the pipe
    #ifdef AUTOREG
        using PipeDataType = typename nnet::ExtractPipeType<src_pipe>::value_type;
        using PipeArrayType = typename PipeDataType::data_type;
    #else
        using PipeDataType = typename nnet::ExtractPipeType<src_pipe>::value_type;
        using PipeArrayType = PipeDataType;
    #endif

        // Then, extract the DataT from StreamingBeat
        constexpr auto srcTypeSize = std::tuple_size<PipeArrayType>{};

        [[intel::fpga_register]] PipeDataType packet;

        // Drain the output pipe and write result to memory.
#ifdef AUTOREG
        size_t i = 0;
        while (true) {
            packet = src_pipe::read();
            if (packet.exit_task) return;
#else
        for (size_t i = 0; i < num_packs; i++) {
            packet = src_pipe::read();
#endif
            #pragma unroll 4
            for (size_t j = 0; j < srcTypeSize; j++) {
            #ifdef AUTOREG
                dst_ptr[i * srcTypeSize + j] = static_cast<dst_T>(packet.data[j].to_double());
            #else
                dst_ptr[i * srcTypeSize + j] = static_cast<dst_T>(packet[j].to_double());
            #endif
            }

        #ifdef AUTOREG
            i = ((i + 1) >= num_packs) ? (i + 1 - num_packs) : (i + 1);
        #endif
        }
    }
};

//////////////////////////////////////////////////////////////////////////////
// These are versions to convert data for the accelerator bridge (using BSP)
//////////////////////////////////////////////////////////////////////////////
/*
template <class srcType, class dest_pipe, size_t SIZE> void DMA_bridge_convert_data(sycl::queue &q, srcType *src) {
    // First, extract the PipeDataT from the pipe
    using PipeDataType = typename nnet::ExtractPipeType<dest_pipe>::value_type;
    // Then, extract the DataT from StreamingBeat
    using DstDataType = typename nnet::ExtractDataType<PipeDataType>::value_type;
    constexpr auto dstTypeSize = std::tuple_size<DstDataType>{};

    constexpr size_t num_iterations = SIZE / dstTypeSize;

    // Allocate host memory
    srcType *vals = sycl::malloc_host<srcType>(SIZE, q);
    if (vals == nullptr) {
        std::cerr << "ERROR: host allocation failed for input\n";
        return;
    }
    // copy to host memory
    for (size_t i = 0; i < SIZE; i++) {
        vals[i] = src[i];
    }
    q.single_task(DMA_convert_data<srcType, dest_pipe>{vals, num_iterations});
}

template <class src_pipe, class dstType, size_t SIZE> void DMA_bridge_convert_data_back(sycl::queue &q, dstType *dst) {
    // First, extract the PipeDataT from the pipe
    using PipeDataType = typename nnet::ExtractPipeType<src_pipe>::value_type;
    // Then, extract the DataT from StreamingBeat
    using SrcDataType = typename nnet::ExtractDataType<PipeDataType>::value_type;
    constexpr auto srcTypeSize = std::tuple_size<SrcDataType>{};

    constexpr size_t num_iterations = SIZE / srcTypeSize;

    // Allocate host memory
    dstType *outputs = sycl::malloc_host<dstType>(SIZE, q);
    if (outputs == nullptr) {
        std::cerr << "ERROR: host allocation failed for output\n";
        return;
    }

    q.single_task(DMA_convert_data_back<src_pipe, dstType>{outputs, num_iterations}).wait();

    // copy the data back
    for (size_t j = 0; j < SIZE; j++) {
        dst[j] = outputs[j];
    }
}
*/

} // namespace nnet

#endif
