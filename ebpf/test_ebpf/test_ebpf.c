// #include <bpf/libbpf.h>

// #include <stdio.h>
// #include <sys/resource.h>
// #include <unistd.h>
// #include <sys/stat.h>

// #include "test_ebpf_load.h"

// static int libbpf_print_fn(enum libbpf_print_level level, const char *format,
//                            va_list args) {
//   return vfprintf(stderr, format, args);
// }

// int main(int argc, char **argv) {
//   struct test_ebpf_load *skel;
//   int err;

//   // libbpf_set_strict_mode(LIBBPF_STRICT_ALL);
//   /* Set up libbpf errors and debug info callback */
//   libbpf_set_print(libbpf_print_fn);

//   /* Open BPF application */
//   skel = test_ebpf_load__open();
//   if (!skel) {
//     fprintf(stderr, "Failed to open BPF skeleton\n");
//     return 1;
//   }

//   struct stat sb;
//   /* ensure BPF program only handles write() syscalls from our process */
//   if (stat("/proc/self/ns/pid", &sb) == -1) {
//     fprintf(stderr, "Failed to acquire namespace information");
//     return 1;
//   }
//   skel->bss->dev = sb.st_dev;
//   skel->bss->ino = sb.st_ino;
//   skel->bss->my_pid = getpid();

//   /* Load & verify BPF programs */
//   err = test_ebpf_load__load(skel);
//   if (err) {
//     fprintf(stderr, "Failed to load and verify BPF skeleton\n");
//     goto cleanup;
//   }

//   /* Attach tracepoint handler */
//   err = test_ebpf_load__attach(skel);
//   if (err) {
//     fprintf(stderr, "Failed to attach BPF skeleton\n");
//     goto cleanup;
//   }

//   printf("Successfully started! Please run `sudo cat "
//          "/sys/kernel/debug/tracing/trace_pipe` "
//          "to see output of the BPF programs.\n");

//   for (;;) {
//     /* trigger our BPF program */
//     fprintf(stderr, ".");
//     sleep(1);
//   }

// cleanup:
//   test_ebpf_load__destroy(skel);
//   return -err;
// }

// #include <bpf/libbpf.h>

// #include "test_ebpf.h"
// #include "test_ebpf_load.h"

// #include <argp.h>
// #include <signal.h>
// #include <stdio.h>
// #include <sys/resource.h>
// #include <time.h>

// static struct env {
//   bool verbose;
//   long min_duration_ms;
// } env;

// const char *argp_program_version = "bootstrap 0.0";
// const char *argp_program_bug_address = "<bpf@vger.kernel.org>";
// const char argp_program_doc[] =
//     "BPF bootstrap demo application.\n"
//     "\n"
//     "It traces process start and exits and shows associated \n"
//     "information (filename, process duration, PID and PPID, etc).\n"
//     "\n"
//     "USAGE: ./bootstrap [-d <min-duration-ms>] [-v]\n";

// static const struct argp_option opts[] = {
//     {"verbose", 'v', NULL, 0, "Verbose debug output"},
//     {"duration", 'd', "DURATION-MS", 0,
//      "Minimum process duration (ms) to report"},
//     {},
// };

// static error_t parse_arg(int key, char *arg, struct argp_state *state) {
//   switch (key) {
//   case 'v':
//     env.verbose = true;
//     break;
//   case 'd':
//     errno = 0;
//     env.min_duration_ms = strtol(arg, NULL, 10);
//     if (errno || env.min_duration_ms <= 0) {
//       fprintf(stderr, "Invalid duration: %s\n", arg);
//       argp_usage(state);
//     }
//     break;
//   case ARGP_KEY_ARG:
//     argp_usage(state);
//     break;
//   default:
//     return ARGP_ERR_UNKNOWN;
//   }
//   return 0;
// }

// static const struct argp argp = {
//     .options = opts,
//     .parser = parse_arg,
//     .doc = argp_program_doc,
// };

// static int libbpf_print_fn(enum libbpf_print_level level, const char *format,
//                            va_list args) {
//   if (level == LIBBPF_DEBUG && !env.verbose)
//     return 0;
//   return vfprintf(stderr, format, args);
// }

// static volatile bool exiting = false;

// static void sig_handler(int sig) { exiting = true; }

// static int handle_event(void *ctx, void *data, size_t data_sz) {
//   const struct event *e = data;
//   struct tm *tm;
//   char ts[32];
//   time_t t;

//   time(&t);
//   tm = localtime(&t);
//   strftime(ts, sizeof(ts), "%H:%M:%S", tm);

//   if (e->exit_event) {
//     printf("%-8s %-5s %-16s %-7d %-7d [%u]", ts, "EXIT", e->comm, e->pid,
//            e->ppid, e->exit_code);
//     if (e->duration_ns)
//       printf(" (%llums)", e->duration_ns / 1000000);
//     printf("\n");
//   } else {
//     printf("%-8s %-5s %-16s %-7d %-7d %s\n", ts, "EXEC", e->comm, e->pid,
//            e->ppid, e->filename);
//   }

//   return 0;
// }

// int main(int argc, char **argv) {
//   struct ring_buffer *rb = NULL;
//   struct test_ebpf_load *skel;
//   int err;

//   /* Parse command line arguments */
//   err = argp_parse(&argp, argc, argv, 0, NULL, NULL);
//   if (err)
//     return err;

//   /* Set up libbpf errors and debug info callback */
//   libbpf_set_print(libbpf_print_fn);

//   /* Cleaner handling of Ctrl-C */
//   signal(SIGINT, sig_handler);
//   signal(SIGTERM, sig_handler);

//   /* Load and verify BPF application */
//   skel = test_ebpf_load__open();
//   if (!skel) {
//     fprintf(stderr, "Failed to open and load BPF skeleton\n");
//     return 1;
//   }

//   /* Parameterize BPF code with minimum duration parameter */
//   skel->rodata->min_duration_ns = env.min_duration_ms * 1000000ULL;

//   /* Load & verify BPF programs */
//   err = test_ebpf_load__load(skel);
//   if (err) {
//     fprintf(stderr, "Failed to load and verify BPF skeleton\n");
//     goto cleanup;
//   }

//   /* Attach tracepoints */
//   err = test_ebpf_load__attach(skel);
//   if (err) {
//     fprintf(stderr, "Failed to attach BPF skeleton\n");
//     goto cleanup;
//   }

//   /* Set up ring buffer polling */
//   rb = ring_buffer__new(bpf_map__fd(skel->maps.rb), handle_event, NULL,
//   NULL); if (!rb) {
//     err = -1;
//     fprintf(stderr, "Failed to create ring buffer\n");
//     goto cleanup;
//   }

//   /* Process events */
//   printf("%-8s %-5s %-16s %-7s %-7s %s\n", "TIME", "EVENT", "COMM", "PID",
//          "PPID", "FILENAME/EXIT CODE");
//   while (!exiting) {
//     err = ring_buffer__poll(rb, 100 /* timeout, ms */);
//     /* Ctrl-C will cause -EINTR */
//     if (err == -EINTR) {
//       err = 0;
//       break;
//     }
//     if (err < 0) {
//       printf("Error polling perf buffer: %d\n", err);
//       break;
//     }
//   }

// cleanup:
//   /* Clean up */
//   ring_buffer__free(rb);
//   test_ebpf_load__destroy(skel);

//   return err < 0 ? -err : 0;
// }

#include <bpf/libbpf.h>

#include <errno.h>
#include <stdio.h>
#include <sys/resource.h>
#include <unistd.h>

#include "test_ebpf_load.h"

static int libbpf_print_fn(enum libbpf_print_level level, const char *format,
                           va_list args) {
  return vfprintf(stderr, format, args);
}

int main(int argc, char **argv) {
  struct test_ebpf_load *skel;
  int err, i;
  LIBBPF_OPTS(bpf_uprobe_opts, uprobe_opts);

  /* Set up libbpf errors and debug info callback */
  libbpf_set_print(libbpf_print_fn);

  /* Load and verify BPF application */
  skel = test_ebpf_load__open_and_load();
  if (!skel) {
    fprintf(stderr, "Failed to open and load BPF skeleton\n");
    return 1;
  }

  /* Attach tracepoint handler */
  uprobe_opts.func_name = "uprobed_add";
  uprobe_opts.retprobe = false;
  /* uprobe/uretprobe expects relative offset of the function to attach
   * to. libbpf will automatically find the offset for us if we provide the
   * function name. If the function name is not specified, libbpf will try
   * to use the function offset instead.
   */
  skel->links.uprobe_add = bpf_program__attach_uprobe_opts(
      skel->progs.uprobe_add, -1 /* self pid */,
      "/home/shenda/remote/ebpf/test_ebpf/app", 0 /* offset for function */,
      &uprobe_opts /* opts */);
  if (!skel->links.uprobe_add) {
    err = -errno;
    fprintf(stderr, "Failed to attach uprobe: %d\n", err);
    goto cleanup;
  }

  /* we can also attach uprobe/uretprobe to any existing or future
   * processes that use the same binary executable; to do that we need
   * to specify -1 as PID, as we do here
   */
  uprobe_opts.func_name = "uprobed_add";
  uprobe_opts.retprobe = true;
  skel->links.uretprobe_add = bpf_program__attach_uprobe_opts(
      skel->progs.uretprobe_add, -1 /* self pid */,
      "/home/shenda/remote/ebpf/test_ebpf/app", 0 /* offset for function */,
      &uprobe_opts /* opts */);
  if (!skel->links.uretprobe_add) {
    err = -errno;
    fprintf(stderr, "Failed to attach uprobe: %d\n", err);
    goto cleanup;
  }

  uprobe_opts.func_name = "uprobed_sub";
  uprobe_opts.retprobe = false;
  skel->links.uprobe_sub = bpf_program__attach_uprobe_opts(
      skel->progs.uprobe_sub, -1, "/home/shenda/remote/ebpf/test_ebpf/app", 0,
      &uprobe_opts);
  if (!skel->links.uprobe_sub) {
    err = -errno;
    fprintf(stderr, "Failed to attach uprobe_sub: %d\n", err);
    goto cleanup;
  }
  uprobe_opts.func_name = "uprobed_sub";
  uprobe_opts.retprobe = true;
  skel->links.uretprobe_sub = bpf_program__attach_uprobe_opts(
      skel->progs.uretprobe_sub, -1, "/home/shenda/remote/ebpf/test_ebpf/app",
      0, &uprobe_opts);
  if (!skel->links.uretprobe_sub) {
    err = -errno;
    fprintf(stderr, "Failed to attach uretprobe_sub: %d\n", err);
    goto cleanup;
  }

  /* Let libbpf perform auto-attach for uprobe_sub/uretprobe_sub
   * NOTICE: we provide path and symbol info in SEC for BPF programs
   */
  err = test_ebpf_load__attach(skel);
  if (err) {
    fprintf(stderr, "Failed to auto-attach BPF skeleton: %d\n", err);
    goto cleanup;
  }

  printf("Successfully started! Please run `sudo cat "
         "/sys/kernel/debug/tracing/trace_pipe` to see output of the BPF "
         "programs.\n");

  for (i = 0;; i++) {
    fprintf(stderr, ".");
    sleep(1);
  }

cleanup:
  test_ebpf_load__destroy(skel);
  return -err;
}
