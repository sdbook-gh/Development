from __future__ import print_function
import os
from bcc import BPF
from time import sleep
import numpy as np
import argparse
from collections import defaultdict

# Argument parsing
parser = argparse.ArgumentParser(description="Measure function duration for a specific PID")
parser.add_argument("pid", help="The PID of the process")
parser.add_argument("function_name", help="The function to trace")
args = parser.parse_args()

# load BPF program
b = BPF(text="""
#include <uapi/linux/ptrace.h>

struct key_t {
    u64 pid;
    char comm[16];
};

BPF_HASH(start, struct key_t);
BPF_HISTOGRAM(dist);

int trace_start(struct pt_regs *ctx) {
    struct key_t key = {};
    u64 ts;

    key.pid = bpf_get_current_pid_tgid();
    bpf_get_current_comm(&(key.comm), sizeof(key.comm));

    ts = bpf_ktime_get_ns();
    start.update(&key, &ts);

    return 0;
}

int trace_end(struct pt_regs *ctx) {
    struct key_t key = {};
    u64 *tsp, delta;

    key.pid = bpf_get_current_pid_tgid();
    bpf_get_current_comm(&(key.comm), sizeof(key.comm));

    tsp = start.lookup(&key);
    if (tsp != 0) {
        delta = bpf_ktime_get_ns() - *tsp;
        dist.increment(bpf_log2l(delta / 1000));
        start.delete(&key);
    }

    return 0;
}
""")

pid = int(args.pid)
mangled_name = args.function_name

binary_path = os.readlink(f"/proc/{pid}/exe")

b.attach_uprobe(name=binary_path, sym=mangled_name, fn_name="trace_start")
b.attach_uretprobe(name=binary_path, sym=mangled_name, fn_name="trace_end")
print("Tracing... Hit Ctrl-C to end.")

# sleep until Ctrl-C
try:
    sleep(99999999)
except KeyboardInterrupt:
    pass

# print output
print()
print("Latency (us) to function entry:")
b["dist"].print_log2_hist("us")
