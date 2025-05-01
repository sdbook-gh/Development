# build elfutil
wget https://sourceware.org/elfutils/ftp/elfutils-latest.tar.bz2
sudo apt install pkg-config libtool
configure && make && make install
# build libbpf-bootstrap
export PATH=$PATH:/data/shenda/ebpf/clang_install/bin
export C_INCLUDE_PATH=/share/personal/github/Development/ebpf/helloworld/elfutils-0.193/_install/include
export LIBRARY_PATH=/share/personal/github/Development/ebpf/helloworld/elfutils-0.193/_install/lib
cd libbpf-bootstrap/bpftool/src/
make bootstrap
# build ebpf
clang -g -O2 -target bpf -D__TARGET_ARCH_x86 -I/usr/include/x86_64-linux-gnu -I./libbpf-bootstrap/bpftool/src/bootstrap/libbpf/include -c helloworld.bpf.c
./libbpf-bootstrap/bpftool/src/bootstrap/bpftool gen skeleton helloworld.bpf.o > helloworld.skel.h
g++ -g -O2 -D__TARGET_ARCH_x86 -I./libbpf-bootstrap/bpftool/src/bootstrap/libbpf/include -L./libbpf-bootstrap/bpftool/src/bootstrap/libbpf -o helloworld helloworld.c -lbpf -lelf -lz
# run helloworld
sudo mount -t debugfs nodev /sys/kernel/debug  # 挂载 DebugFS
sudo mount -t tracefs nodev /sys/kernel/tracing  # 挂载 TraceFS
