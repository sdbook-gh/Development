export PATH=$PATH:/data/shenda/ebpf/clang_install/bin
export C_INCLUDE_PATH=/data/shenda/ebpf/libbpf-bootstrap/elfutils_install/include
export LIBRARY_PATH="/data/shenda/ebpf/libbpf-bootstrap/elfutils_install/lib"
rm -fr * && cmake .. && make
