// app 2, part of a 2-part IPC example
#include <stdio.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#define DSIZE 1024

#define cudaCheckErrors(msg)                                                                                        \
    do {                                                                                                            \
        cudaError_t __err = cudaGetLastError();                                                                     \
        if (__err != cudaSuccess) {                                                                                 \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", msg, cudaGetErrorString(__err), __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n");                                                             \
            exit(1);                                                                                                \
        }                                                                                                           \
    } while (0)

__global__ void set_kernel(volatile int *d, int val) {
    int i = threadIdx.x;
    // int i = blockIdx.x;
    d[i] = val;
}

int main() {
    int *data;
    cudaIpcMemHandle_t my_handle;
    unsigned char handle_buffer[sizeof(my_handle) + 1];
    memset(handle_buffer, 0, sizeof(my_handle) + 1);
    printf("waiting for app1\n");
    getchar();
    FILE *fp;
    fp = fopen("testfifo", "r");
    if (fp == NULL) {
        printf("fifo open fail \n");
        return 1;
    }
    int ret;
    for (int i = 0; i < sizeof(my_handle); i++) {
        ret = fscanf(fp, "%c", handle_buffer + i);
        if (ret == EOF) {
            printf("received EOF\n");
        }
        else if (ret != 1) {
            printf("fscanf returned %d\n", ret);
        }
    }
    memcpy((unsigned char *)(&my_handle), handle_buffer, sizeof(my_handle));
    printf("cudaIpcOpenMemHandle\n");
    getchar();
    cudaIpcOpenMemHandle((void **)&data, my_handle, cudaIpcMemLazyEnablePeerAccess);
    cudaCheckErrors("IPC handle fail");
    set_kernel<<<1, DSIZE>>>(data, 5678);
    set_kernel<<<1, DSIZE / 2>>>(data, 1234);
    cudaDeviceSynchronize();
    cudaCheckErrors("memset fail");
    printf("complete\n");
    return 0;
}
