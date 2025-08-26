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
    cout << "Usage: " << argv[0] << " input_compressed_file output_decompressed_file" << endl;
    return 1;
  }

  // Read compressed input file
  std::ifstream in(argv[1], ios::binary);
  if (!in) {
    cerr << "Cannot open input file: " << argv[1] << endl;
    return 1;
  }
  in.seekg(0, ios::end);
  size_t comp_size = in.tellg();
  in.seekg(0, ios::beg);
  std::vector<uint8_t> comp_data(comp_size);
  in.read(reinterpret_cast<char*>(comp_data.data()), comp_size);
  in.close();

  const size_t chunk_size = 1 << 16; // 64KB chunks
  // create GPU only input buffer
  uint8_t* d_in_data;
  const size_t in_bytes = sizeof(uint8_t) * comp_data.size();
  CUDA_CHECK(cudaMalloc((void**)&d_in_data, in_bytes));
  CUDA_CHECK(cudaMemcpy(d_in_data, comp_data.data(), in_bytes, cudaMemcpyHostToDevice));

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  LZ4Manager manager{chunk_size, NVCOMP_TYPE_UCHAR, stream};

  auto decomp_config = manager.configure_decompression(d_in_data);
  uint8_t* d_decomp;
  CUDA_CHECK(cudaMalloc(&d_decomp, decomp_config.decomp_data_size));
  manager.decompress(d_decomp, d_in_data, decomp_config);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaFree(d_in_data));

  std::vector<uint8_t> output_data(decomp_config.decomp_data_size);
  CUDA_CHECK(cudaMemcpy(output_data.data(), d_decomp, decomp_config.decomp_data_size, cudaMemcpyDeviceToHost));

  // Write to output file
  std::ofstream out(argv[2], ios::binary);
  if (!out) {
    cerr << "Cannot open output file: " << argv[2] << endl;
    return 1;
  }
  out.write(reinterpret_cast<const char*>(output_data.data()), decomp_config.decomp_data_size);
  out.close();

  // Cleanup
  CUDA_CHECK(cudaFree(d_decomp));
  CUDA_CHECK(cudaStreamDestroy(stream));

  cout << "Decompression completed. Decompressed size: " << decomp_config.decomp_data_size << " bytes" << endl;

  return 0;
}
