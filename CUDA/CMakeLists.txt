project(test_cuda)

find_package(CUDA REQUIRED)
set(CUDA_NVCC_FLAGS "-G;-g;-std=c++17") # -arch=sm_87; to specify GPU arch for orin

set(OPENCV_DIR /mnt/d/dev/cuda/opencv)
include_directories(${OPENCV_DIR}/include/opencv4)
link_directories(${OPENCV_DIR}/lib)

cuda_add_library(test_cuda test_cuda.cu)
add_executable(test_opencv test_opencv.cpp)
target_link_libraries(test_opencv test_cuda opencv_world)

include_directories(/mnt/d/dev/cuda/cuda_examples/common/inc)
cuda_add_executable(test_cuda_ipc test_cuda_ipc.cu)

cuda_add_executable(app1 app1.cu)
cuda_add_executable(app2 app2.cu)
