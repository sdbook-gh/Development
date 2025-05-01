#include <bpf/libbpf.h>
#include <cstdint>
#include <fcntl.h>
#include <iostream>
#include <linux/bpf.h>
#include <signal.h>

static volatile bool exiting = false;

struct event {
  uint64_t timestamp;
  uint32_t pid;
  int is_entry;
  char func_name[64];
};

static int handle_event(void *ctx, void *data, size_t size) {
  auto *ev = static_cast<event *>(data);
  printf("%s - Time: %lu ns, PID: %d, Function: %s\n",
         ev->is_entry ? "Entry" : "Exit ", ev->timestamp, ev->pid,
         ev->func_name);
  return 0;
}

int main(int argc, char **argv) {
  struct bpf_object *obj{nullptr};
  struct bpf_program *prog{nullptr};
  struct ring_buffer *rb{nullptr};
  int err{0};
  auto clear = [&]() {
    if (rb) {
      ring_buffer__free(rb);
      rb = nullptr;
    }
    if (obj) {
      bpf_object__close(obj);
      obj = nullptr;
    }
  };

  // 加载BPF对象
  obj = bpf_object__open("bpf.o");
  if (!obj) {
    fprintf(stderr, "Failed to open BPF object\n");
    clear();
    return 1;
  }

  // 加载并验证BPF程序
  err = bpf_object__load(obj);
  if (err) {
    fprintf(stderr, "Failed to load BPF object: %d\n", err);
    clear();
    return 1;
  }

  // 附加uprobe和uretprobe
  prog = bpf_object__find_program_by_name(obj, "handle_entry");
  if (!prog) {
    fprintf(stderr, "Entry probe not found\n");
    clear();
    return 1;
  }
  auto *link_entry = bpf_program__attach_uprobe(
      prog, false, -1, "/share/dev/perf/target", 0x0000000000001329);
  if (!link_entry) {
    fprintf(stderr, "Failed to attach entry probe\n");
    clear();
    return 1;
  }

  prog = bpf_object__find_program_by_name(obj, "handle_exit");
  if (!prog) {
    fprintf(stderr, "Exit probe not found\n");
    clear();
    return 1;
  }
  auto *link_exit = bpf_program__attach_uprobe(
      prog, true, -1, "/share/dev/perf/target", 0x0000000000001329);
  if (!link_exit) {
    fprintf(stderr, "Failed to attach exit probe\n");
    clear();
    return 1;
  }

  // 设置环形缓冲区
  rb =
      ring_buffer__new(bpf_map__fd(bpf_object__find_map_by_name(obj, "events")),
                       handle_event, nullptr, nullptr);
  if (!rb) {
    fprintf(stderr, "Failed to create ring buffer\n");
    clear();
    return 1;
  }

  printf("Monitoring function entries/exits. Ctrl-C to exit.\n");
  while (!exiting) {
    err = ring_buffer__poll(rb, 100 /* timeout ms */);
    if (err == -EINTR) {
      err = 0;
      break;
    }
    if (err < 0) {
      fprintf(stderr, "Error polling ring buffer: %d\n", err);
      break;
    }
  }
  clear();
  return err != 0;
}