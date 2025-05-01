#include "vmlinux.h"
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>

struct event {
  u64 timestamp;
  u32 pid;
  int is_entry;
  char func_name[64];
};

struct {
  __uint(type, BPF_MAP_TYPE_RINGBUF);
  __uint(max_entries, 256 * 1024);
} events SEC(".maps");

SEC("uprobe/example_function")
int handle_entry(struct pt_regs *ctx) {
  struct event *ev;
  ev = bpf_ringbuf_reserve(&events, sizeof(*ev), 0);
  if (!ev)
    return 1;

  ev->timestamp = bpf_ktime_get_ns();
  ev->pid = bpf_get_current_pid_tgid() >> 32;
  ev->is_entry = 1;
  bpf_get_current_comm(ev->func_name, sizeof(ev->func_name));

  bpf_ringbuf_submit(ev, 0);
  return 0;
}

SEC("uretprobe/example_function")
int handle_exit(struct pt_regs *ctx) {
  struct event *ev;
  ev = bpf_ringbuf_reserve(&events, sizeof(*ev), 0);
  if (!ev)
    return 1;

  ev->timestamp = bpf_ktime_get_ns();
  ev->pid = bpf_get_current_pid_tgid() >> 32;
  ev->is_entry = 0;
  bpf_get_current_comm(ev->func_name, sizeof(ev->func_name));

  bpf_ringbuf_submit(ev, 0);
  return 0;
}

char LICENSE[] SEC("license") = "Dual BSD/GPL";