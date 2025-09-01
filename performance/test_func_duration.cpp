// #include <immintrin.h>
// #include <cstdint>
// #include <vector>
// #include <cstdio>
// #include <cstring>
// #include <cassert>
// #include <limits>
// #include <string>
// #include <iostream>

// #if defined(_MSC_VER)
// #include <intrin.h>
// #endif

// // ------------------------------
// // Small helpers
// // ------------------------------

// static inline uint32_t ctz32(uint32_t x) {
// #if defined(_MSC_VER)
//   unsigned long idx;
//   _BitScanForward(&idx, x); // x != 0 by contract
//   return static_cast<uint32_t>(idx);
// #else
//   return static_cast<uint32_t>(__builtin_ctz(x)); // x != 0 by contract
// #endif
// }

// static inline void write_u16_le(std::vector<uint8_t>& out, uint16_t v) {
//   out.push_back(static_cast<uint8_t>(v & 0xFF));
//   out.push_back(static_cast<uint8_t>((v >> 8) & 0xFF));
// }

// static inline uint16_t read_u16_le(const uint8_t* p) { return static_cast<uint16_t>(p[0] | (static_cast<uint16_t>(p[1]) << 8)); }

// // ------------------------------
// // SIMD-accelerated RLE compressor for bytes
// // Format: [value:1][length:2 little-endian] repeated
// // ------------------------------

// static size_t rle_runlen_simd(const uint8_t* base, size_t n, uint8_t v) {
//   // Count consecutive bytes equal to v starting at base, length <= n
//   const uint8_t* p = base;
//   size_t remain = n;

// #if defined(__AVX2__)
//   // Use 32-byte chunks
//   __m256i vv = _mm256_set1_epi8((char)v);
//   while (remain >= 32) {
//     __m256i x = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
//     __m256i eq = _mm256_cmpeq_epi8(x, vv);
//     uint32_t mask = static_cast<uint32_t>(_mm256_movemask_epi8(eq));
//     if (mask == 0xFFFFFFFFu) {
//       p += 32;
//       remain -= 32;
//     } else {
//       uint32_t inv = ~mask;
//       // find first zero (i.e., first not-equal)
//       uint32_t first_diff = ctz32(inv);
//       p += first_diff;
//       return static_cast<size_t>(p - base);
//     }
//   }
// #endif

//   // SSE2 path (or AVX2 tail < 32 but >= 16)
// #if defined(__SSE2__)
//   __m128i vs = _mm_set1_epi8((char)v);
//   while (remain >= 16) {
//     __m128i x = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
//     __m128i eq = _mm_cmpeq_epi8(x, vs);
//     uint32_t mask = static_cast<uint32_t>(_mm_movemask_epi8(eq));
//     if (mask == 0xFFFFu) {
//       p += 16;
//       remain -= 16;
//     } else {
//       uint32_t inv = ~mask & 0xFFFFu;
//       uint32_t first_diff = ctz32(inv);
//       p += first_diff;
//       return static_cast<size_t>(p - base);
//     }
//   }
// #endif

//   // Scalar tail
//   while (remain > 0 && *p == v) {
//     ++p;
//     --remain;
//   }
//   return static_cast<size_t>(p - base);
// }

// std::vector<uint8_t> rle_compress_simd(const uint8_t* data, size_t size) {
//   std::vector<uint8_t> out;
//   out.reserve(size / 2 + 8); // rough guess to reduce realloc

//   size_t i = 0;
//   while (i < size) {
//     uint8_t v = data[i];
//     size_t max_run = size - i;

//     // We cap each chunk to 65535 to fit into 2 bytes length
//     size_t limit = (max_run > std::numeric_limits<uint16_t>::max()) ? std::numeric_limits<uint16_t>::max() : max_run;

//     size_t run = rle_runlen_simd(&data[i], limit, v);
//     if (run == 0) run = 1; // safety, shouldn't happen

//     // Emit (value, length_16le)
//     out.push_back(v);
//     write_u16_le(out, static_cast<uint16_t>(run));

//     i += run;
//   }
//   return out;
// }

// std::vector<uint8_t> rle_decompress(const uint8_t* data, size_t size) {
//   std::vector<uint8_t> out;

//   size_t i = 0;
//   while (i < size) {
//     if (i + 3 > size) {
//       // malformed input
//       throw std::runtime_error("RLE stream truncated");
//     }
//     uint8_t v = data[i];
//     uint16_t len = read_u16_le(&data[i + 1]);
//     i += 3;

//     // Append len copies of v
//     size_t old = out.size();
//     out.resize(old + len);
//     std::memset(out.data() + old, v, len);
//   }
//   return out;
// }

// // ------------------------------
// // Convenience wrappers for std::vector<uint8_t>
// // ------------------------------

// std::vector<uint8_t> rle_compress_simd(const std::vector<uint8_t>& in) {
//   if (in.empty()) return {};
//   return rle_compress_simd(in.data(), in.size());
// }

// std::vector<uint8_t> rle_decompress(const std::vector<uint8_t>& in) {
//   if (in.empty()) return {};
//   return rle_decompress(in.data(), in.size());
// }

// // ------------------------------
// // Demo / Test
// // ------------------------------

// static std::vector<uint8_t> make_test_data() {
//   std::vector<uint8_t> v;
//   v.reserve(300000);

//   // Some repeated patterns
//   auto push_run = [&](uint8_t b, size_t len) {
//     size_t old = v.size();
//     v.resize(old + len);
//     std::memset(v.data() + old, b, len);
//   };

//   push_run(0x00, 100000);
//   push_run(0xFF, 70000);
//   push_run(0x12, 10);
//   push_run(0x34, 10);
//   push_run(0x56, 10);
//   push_run(0x78, 10);
//   push_run(0xAA, 20000);

//   // Some non-repetitive bytes
//   for (int i = 0; i < 1024; ++i) { v.push_back((i * 37) & 0xFF); }

//   push_run(0x9C, 65535);
//   push_run(0x9C, 123); // force multi-chunk for same value

//   return v;
// }

// int main() {
//   try {
//     auto input = make_test_data();

//     auto compressed = rle_compress_simd(input);
//     auto output = rle_decompress(compressed);

//     bool ok = (input == output);
//     std::cout << "Input size:      " << input.size() << " bytes\n";
//     std::cout << "Compressed size: " << compressed.size() << " bytes\n";
//     std::cout << "Decompressed:    " << (ok ? "OK" : "MISMATCH") << "\n";

//     if (!ok) return 1;
//   } catch (const std::exception& ex) {
//     std::cerr << "Error: " << ex.what() << "\n";
//     return 1;
//   }
//   return 0;
// }

#include <immintrin.h>
#include <vector>
#include <cstring>
#include <iostream>
#include <chrono>

// 1. RLE (Run Length Encoding) SIMD加速版本
class SIMDRLECompressor {
public:
    // 使用AVX2加速的RLE压缩
    std::vector<uint8_t> compress(const std::vector<uint8_t>& input) {
        std::vector<uint8_t> output;
        output.reserve(input.size() * 2); // 预分配空间
        
        size_t i = 0;
        const size_t simd_size = 32; // AVX2处理32字节
        
        while (i < input.size()) {
            uint8_t current = input[i];
            size_t run_length = 1;
            
            // 使用SIMD加速查找连续相同字节
            if (i + simd_size <= input.size()) {
                __m256i current_vec = _mm256_set1_epi8(current);
                
                while (i + run_length + simd_size <= input.size()) {
                    __m256i data_vec = _mm256_loadu_si256(
                        reinterpret_cast<const __m256i*>(&input[i + run_length]));
                    __m256i cmp = _mm256_cmpeq_epi8(current_vec, data_vec);
                    
                    int mask = _mm256_movemask_epi8(cmp);
                    if (mask != 0xFFFFFFFF) {
                        // 找到第一个不匹配的位置
                        int first_diff = __builtin_ctz(~mask);
                        run_length += first_diff;
                        break;
                    }
                    run_length += simd_size;
                }
            }
            
            // 处理剩余字节
            while (i + run_length < input.size() && 
                   input[i + run_length] == current && 
                   run_length < 255) {
                run_length++;
            }
            
            // 输出RLE格式: [长度][值]
            output.push_back(static_cast<uint8_t>(run_length));
            output.push_back(current);
            i += run_length;
        }
        
        return output;
    }
    
    // SIMD加速解压缩
    std::vector<uint8_t> decompress(const std::vector<uint8_t>& input) {
        std::vector<uint8_t> output;
        
        for (size_t i = 0; i < input.size(); i += 2) {
            uint8_t length = input[i];
            uint8_t value = input[i + 1];
            
            // 使用SIMD批量填充
            if (length >= 32) {
                __m256i value_vec = _mm256_set1_epi8(value);
                size_t simd_chunks = length / 32;
                
                size_t current_size = output.size();
                output.resize(current_size + length);
                
                for (size_t j = 0; j < simd_chunks; j++) {
                    _mm256_storeu_si256(
                        reinterpret_cast<__m256i*>(&output[current_size + j * 32]),
                        value_vec);
                }
                
                // 处理剩余字节
                for (size_t j = simd_chunks * 32; j < length; j++) {
                    output[current_size + j] = value;
                }
            } else {
                output.insert(output.end(), length, value);
            }
        }
        
        return output;
    }
};

// 2. Delta编码SIMD加速版本
class SIMDDeltaCompressor {
public:
    // Delta编码压缩
    std::vector<int32_t> compress(const std::vector<int32_t>& input) {
        if (input.empty()) return {};
        
        std::vector<int32_t> output;
        output.reserve(input.size());
        output.push_back(input[0]); // 第一个值不变
        
        size_t i = 1;
        const size_t simd_size = 8; // AVX2处理8个int32
        
        // SIMD处理批量delta计算
        while (i + simd_size <= input.size()) {
            __m256i current = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(&input[i]));
            __m256i previous = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(&input[i-1]));
            
            __m256i delta = _mm256_sub_epi32(current, previous);
            
            // 存储delta值
            int32_t deltas[8];
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(deltas), delta);
            
            for (int j = 0; j < 8; j++) {
                output.push_back(deltas[j]);
            }
            
            i += simd_size;
        }
        
        // 处理剩余元素
        while (i < input.size()) {
            output.push_back(input[i] - input[i-1]);
            i++;
        }
        
        return output;
    }
    
    // Delta解码解压缩
    std::vector<int32_t> decompress(const std::vector<int32_t>& input) {
        if (input.empty()) return {};
        
        std::vector<int32_t> output;
        output.reserve(input.size());
        output.push_back(input[0]); // 第一个值不变
        
        size_t i = 1;
        const size_t simd_size = 8;
        
        // SIMD累加解码
        while (i + simd_size <= input.size()) {
            __m256i delta = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(&input[i]));
            
            // 前缀和计算
            __m256i prefix_sum = delta;
            __m256i temp = _mm256_slli_si256(prefix_sum, 4);
            prefix_sum = _mm256_add_epi32(prefix_sum, temp);
            temp = _mm256_slli_si256(prefix_sum, 8);
            prefix_sum = _mm256_add_epi32(prefix_sum, temp);
            temp = _mm256_slli_si256(prefix_sum, 16);
            prefix_sum = _mm256_add_epi32(prefix_sum, temp);
            
            // 加上前一个值
            int32_t last_value = output.back();
            __m256i last_vec = _mm256_set1_epi32(last_value);
            __m256i result = _mm256_add_epi32(prefix_sum, last_vec);
            
            // 存储结果
            int32_t results[8];
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(results), result);
            
            for (int j = 0; j < 8; j++) {
                output.push_back(results[j]);
            }
            
            i += simd_size;
        }
        
        // 处理剩余元素
        while (i < input.size()) {
            output.push_back(output.back() + input[i]);
            i++;
        }
        
        return output;
    }
};

// 3. 简单字典压缩SIMD加速版本
class SIMDDictionaryCompressor {
private:
    static const size_t DICT_SIZE = 256;
    uint8_t dictionary[DICT_SIZE];
    
public:
    // 初始化字典
    void initDictionary() {
        for (int i = 0; i < DICT_SIZE; i++) {
            dictionary[i] = static_cast<uint8_t>(i);
        }
    }
    
    // SIMD加速查找
    int findInDictionary(uint8_t value) {
        const size_t simd_size = 32;
        __m256i target = _mm256_set1_epi8(value);
        
        for (size_t i = 0; i < DICT_SIZE; i += simd_size) {
            __m256i dict_chunk = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(&dictionary[i]));
            __m256i cmp = _mm256_cmpeq_epi8(target, dict_chunk);
            
            int mask = _mm256_movemask_epi8(cmp);
            if (mask != 0) {
                return i + __builtin_ctz(mask);
            }
        }
        return -1;
    }
};

// 4. 位压缩SIMD加速版本
class SIMDBitPacker {
public:
    // 将32位整数压缩到指定位数
    std::vector<uint8_t> pack(const std::vector<uint32_t>& input, int bits_per_int) {
        if (bits_per_int > 32 || bits_per_int < 1) return {};
        
        std::vector<uint8_t> output;
        size_t total_bits = input.size() * bits_per_int;
        output.resize((total_bits + 7) / 8);
        
        uint64_t buffer = 0;
        int buffer_bits = 0;
        size_t output_pos = 0;
        
        for (uint32_t value : input) {
            // 掩码确保只使用指定位数
            uint32_t masked_value = value & ((1U << bits_per_int) - 1);
            
            buffer |= (static_cast<uint64_t>(masked_value) << buffer_bits);
            buffer_bits += bits_per_int;
            
            // 输出完整的字节
            while (buffer_bits >= 8 && output_pos < output.size()) {
                output[output_pos++] = static_cast<uint8_t>(buffer & 0xFF);
                buffer >>= 8;
                buffer_bits -= 8;
            }
        }
        
        // 输出剩余位
        if (buffer_bits > 0 && output_pos < output.size()) {
            output[output_pos] = static_cast<uint8_t>(buffer & 0xFF);
        }
        
        return output;
    }
    
    // 解压缩位压缩数据
    std::vector<uint32_t> unpack(const std::vector<uint8_t>& input, 
                                 int bits_per_int, size_t count) {
        std::vector<uint32_t> output;
        output.reserve(count);
        
        uint64_t buffer = 0;
        int buffer_bits = 0;
        size_t input_pos = 0;
        uint32_t mask = (1U << bits_per_int) - 1;
        
        for (size_t i = 0; i < count; i++) {
            // 确保buffer中有足够的位
            while (buffer_bits < bits_per_int && input_pos < input.size()) {
                buffer |= (static_cast<uint64_t>(input[input_pos++]) << buffer_bits);
                buffer_bits += 8;
            }
            
            if (buffer_bits >= bits_per_int) {
                output.push_back(static_cast<uint32_t>(buffer & mask));
                buffer >>= bits_per_int;
                buffer_bits -= bits_per_int;
            }
        }
        
        return output;
    }
};

// 性能测试函数
void benchmark() {
    const size_t data_size = 1000000;
    
    // 生成测试数据
    std::vector<uint8_t> test_data(data_size);
    for (size_t i = 0; i < data_size; i++) {
        test_data[i] = static_cast<uint8_t>(i % 256);
    }
    
    SIMDRLECompressor rle;
    
    // 压缩测试
    auto start = std::chrono::high_resolution_clock::now();
    auto compressed = rle.compress(test_data);
    auto end = std::chrono::high_resolution_clock::now();
    
    auto compress_time = std::chrono::duration_cast<std::chrono::microseconds>(
        end - start).count();
    
    std::cout << "压缩时间: " << compress_time << " 微秒" << std::endl;
    std::cout << "原始大小: " << test_data.size() << " 字节" << std::endl;
    std::cout << "压缩大小: " << compressed.size() << " 字节" << std::endl;
    std::cout << "压缩比: " << (double)compressed.size() / test_data.size() << std::endl;
    
    // 解压缩测试
    start = std::chrono::high_resolution_clock::now();
    auto decompressed = rle.decompress(compressed);
    end = std::chrono::high_resolution_clock::now();
    
    auto decompress_time = std::chrono::duration_cast<std::chrono::microseconds>(
        end - start).count();
    
    std::cout << "解压时间: " << decompress_time << " 微秒" << std::endl;
    std::cout << "数据正确性: " << (test_data == decompressed ? "正确" : "错误") << std::endl;
}

int main() {
    std::cout << "SIMD加速压缩解压缩测试" << std::endl;
    benchmark();
    
    // Delta压缩测试
    std::vector<int32_t> int_data = {1, 2, 4, 7, 11, 16, 22, 29, 37, 46};
    SIMDDeltaCompressor delta;
    
    auto compressed_delta = delta.compress(int_data);
    auto decompressed_delta = delta.decompress(compressed_delta);
    
    std::cout << "\nDelta压缩测试:" << std::endl;
    std::cout << "原始: ";
    for (int val : int_data) std::cout << val << " ";
    std::cout << "\n解压: ";
    for (int val : decompressed_delta) std::cout << val << " ";
    std::cout << std::endl;
    
    return 0;
}