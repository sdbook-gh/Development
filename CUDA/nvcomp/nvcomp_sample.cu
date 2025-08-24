#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <memory>
#include <cuda_runtime.h>
#include <nvcomp/lz4.h>
#include <nvcomp/nvcompManagerFactory.hpp>

// CUDA错误检查宏
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << std::endl; \
            exit(1); \
        } \
    } while(0)

#define NVCOMP_CHECK(call) \
    do { \
        nvcompStatus_t err = call; \
        if (err != nvcompSuccess) { \
            std::cerr << "nvcomp error at " << __FILE__ << ":" << __LINE__ << " - " << err << std::endl; \
            exit(1); \
        } \
    } while(0)

class BatchedLZ4Compressor {
private:
    cudaStream_t stream;
    void* temp_buffer;
    size_t temp_buffer_size;
    
public:
    BatchedLZ4Compressor() : temp_buffer(nullptr), temp_buffer_size(0) {
        CUDA_CHECK(cudaStreamCreate(&stream));
    }
    
    ~BatchedLZ4Compressor() {
        if (temp_buffer) {
            cudaFree(temp_buffer);
        }
        cudaStreamDestroy(stream);
    }
    
    // 批量压缩函数
    void compressBatched(const std::vector<void*>& input_ptrs,
                        const std::vector<size_t>& input_sizes,
                        std::vector<void*>& output_ptrs,
                        std::vector<size_t>& output_sizes) {
        
        const size_t batch_size = input_ptrs.size();
        if (batch_size == 0) return;
        
        // // 配置LZ4压缩选项
        // nvcompBatchedLZ4Opts_t opts = {0}; // 使用默认选项
        const size_t chunk_size = 1 << 16;
        // static_assert(chunk_size <= nvcompLZ4CompressionMaxAllowedChunkSize, "Chunk size must be less than the constant specified in the nvCOMP library");

        // 获取临时缓冲区大小
        size_t required_temp_size;
        NVCOMP_CHECK(nvcompBatchedLZ4CompressGetTempSize(
            batch_size,
            chunk_size,
            nvcompBatchedLZ4DefaultOpts,
            &required_temp_size));
        
        // 分配临时缓冲区（如果需要更大的）
        if (required_temp_size > temp_buffer_size) {
            if (temp_buffer) {
                cudaFree(temp_buffer);
            }
            CUDA_CHECK(cudaMalloc(&temp_buffer, required_temp_size));
            temp_buffer_size = required_temp_size;
        }
        
        // 获取最大压缩输出大小
        std::vector<size_t> max_output_sizes(batch_size);
        for (size_t i = 0; i < batch_size; ++i) {
            NVCOMP_CHECK(nvcompBatchedLZ4CompressGetMaxOutputChunkSize(
                input_sizes[i],
                nvcompBatchedLZ4DefaultOpts,
                &max_output_sizes[i]));
        }
        
        // 分配输出缓冲区
        output_ptrs.resize(batch_size);
        output_sizes.resize(batch_size);
        
        for (size_t i = 0; i < batch_size; ++i) {
            CUDA_CHECK(cudaMalloc(&output_ptrs[i], max_output_sizes[i]));
        }
        
        // 执行批量压缩
        NVCOMP_CHECK(nvcompBatchedLZ4CompressAsync(
            input_ptrs.data(),      // 输入数据指针数组
            input_sizes.data(),     // 输入数据大小数组
            chunk_size,  // 最大输入块大小
            batch_size,             // 批次大小
            temp_buffer,            // 临时缓冲区
            temp_buffer_size,       // 临时缓冲区大小
            output_ptrs.data(),     // 输出缓冲区指针数组
            output_sizes.data(),    // 输出大小数组（将被填充实际压缩大小）
            nvcompBatchedLZ4DefaultOpts, // 压缩选项
            stream));               // CUDA流
        
        // 等待压缩完成
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
};

// 示例用法
int main() {
    try {
        // 初始化CUDA设备
        int device = 0;
        CUDA_CHECK(cudaSetDevice(device));
        
        std::cout << "使用nvcomp批量LZ4压缩示例\n";
        
        // 创建测试数据
        const size_t num_chunks = 4;           // 批次大小
        const size_t chunk_size = 1024 * 1024; // 每个块1MB
        
        std::vector<std::vector<char>> host_data(num_chunks);
        std::vector<void*> device_input_ptrs(num_chunks);
        std::vector<size_t> input_sizes(num_chunks);
        
        // 生成测试数据并复制到GPU
        for (size_t i = 0; i < num_chunks; ++i) {
            // 生成测试数据（重复模式，容易压缩）
            host_data[i].resize(chunk_size);
            for (size_t j = 0; j < chunk_size; ++j) {
                host_data[i][j] = (char)(j % 256);
            }
            
            // 分配GPU内存并复制数据
            CUDA_CHECK(cudaMalloc(&device_input_ptrs[i], chunk_size));
            CUDA_CHECK(cudaMemcpy(device_input_ptrs[i], 
                                 host_data[i].data(), 
                                 chunk_size, 
                                 cudaMemcpyHostToDevice));
            input_sizes[i] = chunk_size;
        }
        
        // 创建压缩器并执行压缩
        BatchedLZ4Compressor compressor;
        std::vector<void*> output_ptrs;
        std::vector<size_t> output_sizes;
        
        auto start = std::chrono::high_resolution_clock::now();
        
        compressor.compressBatched(device_input_ptrs, input_sizes, 
                                 output_ptrs, output_sizes);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        // 计算压缩统计信息
        size_t total_input_size = 0;
        size_t total_output_size = 0;
        
        for (size_t i = 0; i < num_chunks; ++i) {
            total_input_size += input_sizes[i];
            total_output_size += output_sizes[i];
            
            std::cout << "块 " << i << ": " 
                     << input_sizes[i] << " -> " << output_sizes[i] 
                     << " bytes (压缩率: " 
                     << std::fixed << std::setprecision(2)
                     << (1.0 - (double)output_sizes[i] / input_sizes[i]) * 100 
                     << "%)\n";
        }
        
        std::cout << "\n总计:\n";
        std::cout << "输入大小: " << total_input_size / (1024.0 * 1024.0) << " MB\n";
        std::cout << "输出大小: " << total_output_size / (1024.0 * 1024.0) << " MB\n";
        std::cout << "总体压缩率: " 
                 << (1.0 - (double)total_output_size / total_input_size) * 100 << "%\n";
        std::cout << "压缩时间: " << duration.count() << " ms\n";
        std::cout << "吞吐量: " 
                 << (total_input_size / (1024.0 * 1024.0)) / (duration.count() / 1000.0) 
                 << " MB/s\n";
        
        // 验证压缩结果（可选）
        std::cout << "\n验证压缩结果...\n";
        bool verification_success = true;
        
        for (size_t i = 0; i < num_chunks; ++i) {
            // 这里可以添加解压缩验证代码
            if (output_sizes[i] == 0) {
                std::cerr << "块 " << i << " 压缩失败!\n";
                verification_success = false;
            }
        }
        
        if (verification_success) {
            std::cout << "所有块压缩成功!\n";
        }
        
        // 清理资源
        for (size_t i = 0; i < num_chunks; ++i) {
            cudaFree(device_input_ptrs[i]);
            cudaFree(output_ptrs[i]);
        }
        
        std::cout << "压缩完成!\n";
        
    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}