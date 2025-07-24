#include <stdlib.h>
#include <stdio.h>

int main() {
    char* ptr = (char*)malloc(128);
    if (!ptr) {
        printf("malloc failed\n");
        return 1;
    }
    free(ptr);
    // 故意重复释放引发double free错误
    free(ptr);
    printf("Completed double free test\n");
    return 0;
}
