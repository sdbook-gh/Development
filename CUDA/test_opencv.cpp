
#include <cstdio>

#include "api.h"
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/opencv.hpp"

int main() {
    // const int arraySize = 5;
    // const int a[arraySize] = {1, 2, 3, 4, 5};
    // const int b[arraySize] = {10, 20, 30, 40, 50};
    // int c[arraySize] = {0};
    // // Add vectors in parallel.
    // auto status = addWithCuda(c, a, b, arraySize);
    // if (status != 0) {
    //     fprintf(stderr, "addWithCuda failed!");
    //     return 1;
    // }
    // printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n", c[0], c[1], c[2], c[3], c[4]);
    // // cudaDeviceReset must be called before exiting in order for profiling and
    // // tracing tools such as Nsight and Visual Profiler to show complete traces.
    // resetCudaDevice();
    // getchar();  // here we want the console to hold for a while

    cv::Mat image;
    image = cv::imread("/mnt/d/dev/NETA/code/test.jpg");
    if (image.data == nullptr) {
        fprintf(stderr, "read image failed!");
    }
    // cv::imshow("", image);
    // cv::waitKey(0);  // 按任意键在0秒后退出窗口，不写这句话是不会显示出窗口的
    uint8_t *img_date = image.data;
    uint32_t img_size = image.rows * image.cols * 3;
    cv::Mat processed_image(cv::Size(image.cols, image.rows), CV_8UC1);

    while (true) {
        processWithCuda(image.data, image.rows, image.rows, processed_image.data);
    }

    // grayWithCuda(image.data, image.rows, image.rows, processed_image.data);
    // cv::imshow("", processed_image);
    // cv::waitKey(0);

    // cv::Mat src = cv::imread("/mnt/d/dev/NETA/code/test.jpg");
    // cv::imshow("", src);
    // cv::waitKey(0);
    // cv::cuda::GpuMat G_image;
    // G_image.upload(src);
    // // cv::cuda::GpuMat G_image(src);
    // cv::cuda::GpuMat G_gray;
    // cv::cuda::cvtColor(G_image, G_gray, cv::COLOR_BGR2GRAY); // cuda版本里也有cvtColor这个API
    // cv::Mat c_gray;
    // G_gray.download(c_gray); // 创建一个CPU mat对象后，由于imshow无法显示GMat对象，所以需要从GMat中取出来赋给CMat对象
    // cv::imshow("gray", c_gray); // imshow显示不了GMat对象
    // cv::waitKey(0);

    cv::destroyAllWindows();

    return 0;
}
