g++ target.cpp -o target
clang -g -O2 -target bpf -D__TARGET_ARCH_x86 -I/usr/include/x86_64-linux-gnu -I./libbpf-bootstrap/bpftool/src/bootstrap/libbpf/include -c bpf.c
g++ -g -O2 -D__TARGET_ARCH_x86 -I./libbpf-bootstrap/bpftool/src/bootstrap/libbpf/include -L./libbpf-bootstrap/bpftool/src/bootstrap/libbpf -o main main.cpp -lbpf -lelf -lz
