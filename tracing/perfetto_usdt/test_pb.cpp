#include "amessage.pb.h"
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <thread>
#include <vector>

#include <perfetto.h>
PERFETTO_DEFINE_CATEGORIES(
    perfetto::Category("normal").SetTags("normal").SetDescription("normal"),
    perfetto::Category("debug").SetTags("debug").SetDescription(
        "Verbose network events"));

#include <sys/sdt.h>

enum TraceType { DISABLE = 0, PERFETTO, USDT };
static std::atomic_bool print_log{false};
static std::atomic_int8_t trace_type{TraceType::DISABLE};
static std::atomic_bool trace_high_freq{false};
static std::atomic_bool trace_normal{false};
static std::atomic_bool trace_detail{false};
static std::atomic_bool trace_detail_ckeck{false};
static std::atomic_bool trace_detail_stopped{false};
static auto start_trace_detail = [] {
  std::thread trace_detail_trigger_th([] {
    if (trace_high_freq.load(std::memory_order_acquire)) {
      trace_detail_ckeck.store(true, std::memory_order_release);
    } else {
      while (!trace_detail_stopped.load(std::memory_order_acquire)) {
        trace_detail_ckeck.store(true, std::memory_order_release);
        std::this_thread::sleep_for(std::chrono::microseconds(1));
      }
    }
  });
  trace_detail_trigger_th.detach();
};

class ScopedTimer {
public:
  ScopedTimer(const char *name)
      : m_name(name), m_beg(std::chrono::high_resolution_clock::now()) {}
  ScopedTimer(const char *name, uint64_t &ns) : ScopedTimer(name) {
    m_ns = &ns;
  }
  ~ScopedTimer() {
    auto end = std::chrono::high_resolution_clock::now();
    auto dur =
        std::chrono::duration_cast<std::chrono::microseconds>(end - m_beg);
    if (m_ns)
      *m_ns = dur.count();
    else
      std::cout << std::left << std::setw(45) << m_name << " : " << std::right
                << std::setw(12) << dur.count() << " us\n";
  }

private:
  const char *m_name;
  std::chrono::time_point<std::chrono::high_resolution_clock> m_beg;
  uint64_t *m_ns = nullptr;
};

void pb_serialize(std::string &out_buf) {
  test::protobuf::cpp::PointCloud pointCloud;
  pointCloud.set_height(1080);
  pointCloud.set_width(1920);
  pointCloud.set_point_step(1);
  pointCloud.set_row_step(1);
  for (int i = 0; i < pointCloud.height() * pointCloud.width(); i++) {
    if (trace_type.load(std::memory_order_acquire) == TraceType::PERFETTO) {
      if (trace_detail.load(std::memory_order_acquire) &&
          trace_detail_ckeck.load(std::memory_order_acquire)) {
        TRACE_EVENT_BEGIN("normal", "pb_add_point_cloud");
      }
    } else if (trace_type.load(std::memory_order_acquire) == TraceType::USDT) {
      if (trace_detail.load(std::memory_order_acquire) &&
          trace_detail_ckeck.load(std::memory_order_acquire)) {
        DTRACE_PROBE(normal, pb_add_point_cloud_begin);
      }
    }
    auto *point = pointCloud.add_points();
    point->set_time(time(nullptr));
    point->set_x(i);
    point->set_y(i);
    point->set_z(i);
    if (trace_type.load(std::memory_order_acquire) == TraceType::PERFETTO) {
      if (trace_detail.load(std::memory_order_acquire) &&
          trace_detail_ckeck.load(std::memory_order_acquire)) {
        if (!trace_high_freq.load(std::memory_order_acquire)) {
          trace_detail_ckeck.store(false, std::memory_order_release);
        }
        TRACE_EVENT_END("normal");
      }
    } else if (trace_type.load(std::memory_order_acquire) == TraceType::USDT) {
      if (trace_detail.load(std::memory_order_acquire) &&
          trace_detail_ckeck.load(std::memory_order_acquire)) {
        if (!trace_high_freq.load(std::memory_order_acquire)) {
          trace_detail_ckeck.store(false, std::memory_order_release);
        }
        DTRACE_PROBE(normal, pb_add_point_cloud_end);
      }
    }
  }
  pointCloud.SerializeToString(&out_buf);
}

void pb_deserialize(std::string &out_buf) {
  test::protobuf::cpp::PointCloud pointCloud;
  pointCloud.ParseFromString(out_buf);
  if (print_log.load(std::memory_order_acquire))
    printf("%s points %d\n", __FUNCTION__, pointCloud.points_size());
  double result = 0;
  for (size_t i = 0; i < pointCloud.points_size(); ++i) {
    if (trace_type.load(std::memory_order_acquire) == TraceType::PERFETTO) {
      if (trace_detail.load(std::memory_order_acquire) &&
          trace_detail_ckeck.load(std::memory_order_acquire)) {
        TRACE_EVENT_BEGIN("normal", "pb_get_point_cloud");
      }
    } else if (trace_type.load(std::memory_order_acquire) == TraceType::USDT) {
      if (trace_detail.load(std::memory_order_acquire) &&
          trace_detail_ckeck.load(std::memory_order_acquire)) {
        DTRACE_PROBE(normal, pb_get_point_cloud_begin);
      }
    }
    result += (pointCloud.points(i).x() + pointCloud.points(i).y() +
               pointCloud.points(i).z());
    if (trace_type.load(std::memory_order_acquire) == TraceType::PERFETTO) {
      if (trace_detail.load(std::memory_order_acquire) &&
          trace_detail_ckeck.load(std::memory_order_acquire)) {
        if (!trace_high_freq.load(std::memory_order_acquire)) {
          trace_detail_ckeck.store(false, std::memory_order_release);
        }
        TRACE_EVENT_END("normal");
      }
    } else if (trace_type.load(std::memory_order_acquire) == TraceType::USDT) {
      if (trace_detail.load(std::memory_order_acquire) &&
          trace_detail_ckeck.load(std::memory_order_acquire)) {
        if (!trace_high_freq.load(std::memory_order_acquire)) {
          trace_detail_ckeck.store(false, std::memory_order_release);
        }
        DTRACE_PROBE(normal, pb_get_point_cloud_end);
      }
    }
  }
  if (print_log.load(std::memory_order_acquire))
    printf("%s %f\n", __FUNCTION__, result);
}

void bench(int count) {
  {
    ScopedTimer pb_serialize_timer("pb_serialize");
    for (int i = 0; i < count; ++i) {
      std::string pb_str;
      if (trace_type.load(std::memory_order_acquire) == TraceType::PERFETTO) {
        if (trace_normal.load(std::memory_order_acquire)) {
          TRACE_EVENT_BEGIN("normal", "pb_serialize");
        }
      } else if (trace_type.load(std::memory_order_acquire) ==
                 TraceType::USDT) {
        if (trace_normal.load(std::memory_order_acquire)) {
          DTRACE_PROBE(normal, pb_serialize_start);
        }
      }
      pb_serialize(pb_str);
      if (trace_type.load(std::memory_order_acquire) == TraceType::PERFETTO) {
        if (trace_normal.load(std::memory_order_acquire)) {
          TRACE_EVENT_END("normal");
        }
      } else if (trace_type.load(std::memory_order_acquire) ==
                 TraceType::USDT) {
        if (trace_normal.load(std::memory_order_acquire)) {
          DTRACE_PROBE(normal, pb_serialize_finish);
        }
      }
    }
  }
  {
    std::string pb_str;
    { pb_serialize(pb_str); }
    ScopedTimer pb_deserialize_timer("pb_deserialize");
    for (int i = 0; i < count; ++i) {
      if (trace_type.load(std::memory_order_acquire) == TraceType::PERFETTO) {
        if (trace_normal.load(std::memory_order_acquire)) {
          TRACE_EVENT_BEGIN("normal", "pb_deserialize");
        }
      } else if (trace_type.load(std::memory_order_acquire) ==
                 TraceType::USDT) {
        if (trace_normal.load(std::memory_order_acquire)) {
          DTRACE_PROBE(normal, pb_deserialize_start);
        }
      }
      pb_deserialize(pb_str);
      if (trace_type.load(std::memory_order_acquire) == TraceType::PERFETTO) {
        if (trace_normal.load(std::memory_order_acquire)) {
          TRACE_EVENT_END("normal");
        }
      } else if (trace_type.load(std::memory_order_acquire) ==
                 TraceType::USDT) {
        if (trace_normal.load(std::memory_order_acquire)) {
          DTRACE_PROBE(normal, pb_deserialize_fnish);
        }
      }
    }
  }
}

class Observer : public perfetto::TrackEventSessionObserver {
public:
  Observer() { perfetto::TrackEvent::AddSessionObserver(this); }
  ~Observer() override { perfetto::TrackEvent::RemoveSessionObserver(this); }
  void OnStart(const perfetto::DataSourceBase::StartArgs &) override {
    std::unique_lock<std::mutex> lock(mutex);
    cv.notify_one();
  }
  void WaitForTracingStart() {
    printf("Waiting for tracing to start...\n");
    std::unique_lock<std::mutex> lock(mutex);
    cv.wait(lock, [] { return perfetto::TrackEvent::IsEnabled(); });
    printf("Tracing started\n");
  }

private:
  std::mutex mutex;
  std::condition_variable cv;
};

void InitializePerfetto() {
  perfetto::TracingInitArgs args;
  // The backends determine where trace events are recorded. For this example we
  // are going to use the system-wide tracing service, so that we can see our
  // app's events in context with system profiling information.
  args.backends = perfetto::kSystemBackend;
  args.enable_system_consumer = false;
  perfetto::Tracing::Initialize(args);
  perfetto::TrackEvent::Register();
}

int main() {
  if (std::filesystem::exists("print_log")) {
    print_log.store(true, std::memory_order_release);
  }
  std::fstream type_file("trace_type", std::ios::in);
  if (type_file) {
    std::string trace_type_str;
    type_file >> trace_type_str;
    if (trace_type_str == "perfetto") {
      trace_type = TraceType::PERFETTO;
    } else if (trace_type_str == "usdt") {
      trace_type = TraceType::USDT;
    }
  }
  if (std::filesystem::exists("trace_normal")) {
    trace_normal.store(true, std::memory_order_release);
  }
  if (std::filesystem::exists("trace_detail")) {
    trace_detail.store(true, std::memory_order_release);
    start_trace_detail();
    if (std::filesystem::exists("trace_high_freq")) {
      trace_high_freq.store(true, std::memory_order_release);
    }
  }
  printf("trace_type:%d trace_normal:%d trace_detail:%d trace_high_freq:%d\n",
         (int)trace_type, (int)trace_normal, (int)trace_detail,
         (int)trace_high_freq);

  if (trace_type.load(std::memory_order_acquire) == TraceType::PERFETTO) {
    if (trace_normal.load(std::memory_order_acquire) ||
        trace_detail.load(std::memory_order_acquire)) {
      InitializePerfetto();
      Observer observer;
      observer.WaitForTracingStart();
    }
  }
  printf("start bench\n");
  bench(100);
  printf("complete bench\n");
  if (trace_type.load(std::memory_order_acquire) == TraceType::PERFETTO) {
    if (trace_normal.load(std::memory_order_acquire) ||
        trace_detail.load(std::memory_order_acquire)) {
      perfetto::TrackEvent::Flush();
    }
  }
  trace_detail_stopped.store(true, std::memory_order_release);
  return 0;
}
