#include <iostream>
#include <fstream>
#include <vector>
#include <cuda_runtime.h>
#include "nvcomp.hpp"
#include "nvcomp/lz4.hpp"

#define CUDA_CHECK(err)                                                                                               \
  do {                                                                                                                \
    cudaError_t error = (err);                                                                                        \
    if (error != cudaSuccess) {                                                                                       \
      std::cerr << "CUDA error: " << cudaGetErrorString(error) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
      exit(1);                                                                                                        \
    }                                                                                                                 \
  } while (0)

using namespace std;
using namespace nvcomp;

int main(int argc, char* argv[]) {
  if (argc != 3) {
    cout << "Usage: " << argv[0] << " input_file output_file" << endl;
    return 1;
  }

  // Read input file
  std::ifstream in(argv[1], ios::binary);
  if (!in) {
    cerr << "Cannot open input file: " << argv[1] << endl;
    return 1;
  }
  in.seekg(0, ios::end);
  size_t input_size = in.tellg();
  in.seekg(0, ios::beg);
  std::vector<uint8_t> input_data(input_size);
  in.read(reinterpret_cast<char*>(input_data.data()), input_size);
  in.close();

  const size_t chunk_size = 1 << 16; // 64KB chunks
  // create GPU only input buffer
  uint8_t* d_in_data;
  const size_t in_bytes = sizeof(uint8_t) * input_data.size();
  CUDA_CHECK(cudaMalloc((void**)&d_in_data, in_bytes));
  CUDA_CHECK(cudaMemcpy(d_in_data, input_data.data(), in_bytes, cudaMemcpyHostToDevice));

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  LZ4Manager manager{chunk_size, NVCOMP_TYPE_UCHAR, stream};
  auto comp_config = manager.configure_compression(in_bytes);

  // Allocate output buffer
  uint8_t* d_comp_out;
  CUDA_CHECK(cudaMalloc(&d_comp_out, comp_config.max_compressed_buffer_size));

  manager.compress(reinterpret_cast<const uint8_t*>(d_in_data), d_comp_out, comp_config);

  CUDA_CHECK(cudaStreamSynchronize(stream));

  size_t comp_out_bytes = manager.get_compressed_output_size(d_comp_out);

  cudaFree(d_in_data);

  // Copy compressed data back to host
  std::vector<uint8_t> comp_data(comp_out_bytes);
  CUDA_CHECK(cudaMemcpy(comp_data.data(), d_comp_out, comp_out_bytes, cudaMemcpyDeviceToHost));

  // Write to output file
  std::ofstream out(argv[2], ios::binary);
  if (!out) {
    cerr << "Cannot open output file: " << argv[2] << endl;
    return 1;
  }
  out.write(reinterpret_cast<const char*>(comp_data.data()), comp_out_bytes);
  out.close();

  // Cleanup
  CUDA_CHECK(cudaFree(d_comp_out));
  CUDA_CHECK(cudaStreamDestroy(stream));

  cout << "Compression completed. Compressed size: " << comp_out_bytes << " bytes" << endl;
  return 0;
}
