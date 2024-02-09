// app 1, part of a 2-part IPC example
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

int main() {
    system("rm -f testfifo");           // remove any debris
    int ret = mkfifo("testfifo", 0600); // create fifo
    if (ret != 0) {
        printf("mkfifo error: %d\n", ret);
        return 1;
    }
    int *data;
    cudaMalloc(&data, DSIZE * sizeof(int));
    cudaCheckErrors("malloc fail");
    cudaMemset(data, 0, DSIZE * sizeof(int));
    cudaCheckErrors("memset fail");
    cudaIpcMemHandle_t my_handle;
    cudaIpcGetMemHandle(&my_handle, data);
    unsigned char handle_buffer[sizeof(my_handle) + 1];
    memset(handle_buffer, 0, sizeof(my_handle) + 1);
    memcpy(handle_buffer, (unsigned char *)(&my_handle), sizeof(my_handle));
    cudaCheckErrors("get IPC handle fail");
    FILE *fp;
    printf("waiting for app2\n");
    fp = fopen("testfifo", "w");
    if (fp == NULL) {
        printf("fifo open fail \n");
        return 1;
    }
    for (int i = 0; i < sizeof(my_handle); i++) {
        ret = fprintf(fp, "%c", handle_buffer[i]);
        if (ret == EOF) {
            printf("received EOF\n");
            break;
        }
        if (ret != 1) printf("ret = %d\n", ret);
    }
    fclose(fp);
    printf("read from fifo\n");
    getchar();
    int *result = (int *)malloc(DSIZE * sizeof(int));
    cudaMemcpy(result, data, DSIZE * sizeof(int), cudaMemcpyDeviceToHost);
    if (!(*result)) {
        printf("Fail!\n");
    } else {
        for (int i = 0; i < DSIZE; i++) {
            printf("%d", result[i]);
        }
        printf("\nSuccess!\n");
    }
    system("rm testfifo");
    return 0;
}
