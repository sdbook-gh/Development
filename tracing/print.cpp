#include <cstdio>

static FILE* outfile = nullptr;

extern "C" {
    void start_tracing() {
        outfile = fopen("out", "wb");
    }
    void do_testing() {
    }
    void stop_tracing() {
        fclose(outfile);
        outfile = nullptr;
    }
	// void __cyg_profile_func_enter(void *this_fn, void *call_site) __attribute__((no_instrument_function));
	// void __cyg_profile_func_exit(void *this_fn, void *call_site) __attribute__((no_instrument_function));

	// void __cyg_profile_func_enter(void *func,  void *caller) {
	// 	static const int unused = []() __attribute__((no_instrument_function)) {
	// 		void* enter = (void*)(&__cyg_profile_func_enter);
	// 		std::printf("print %p\n\n", enter);
	// 		return 0;
	// 	}(); (void) unused;
	// 	if (outfile == nullptr) return;
	// 	// std::printf("> %p\n", func);
	// 	fprintf(outfile, ">>");
	// 	fprintf(outfile, "%p\n", func);
	// }
	// void __cyg_profile_func_exit (void *func, void *caller) {
	// 	if (outfile == nullptr) return;
	// 	// std::printf("< %p\n", func);
	// 	fprintf(outfile, "<<");
	// 	fprintf(outfile, "%p\n", func);
	// }
}

#include "dlfcn.h"
static Dl_info get_dl_info(void *func,  void *caller) __attribute__((no_instrument_function));
Dl_info get_dl_info(void *func,  void *caller) {
	Dl_info info;
	if(!dladdr(func, &info)){
		info.dli_fname = "?";
		info.dli_sname = "?";
	}
	if(!info.dli_fname) {
		info.dli_fname = "?";
	}
	if(!info.dli_sname) {
		info.dli_sname = "?";
	}
	return info;
}

static void __cyg_profile_func_enter(void *this_fn, void *call_site) __attribute__((no_instrument_function));
static void __cyg_profile_func_exit(void *this_fn, void *call_site) __attribute__((no_instrument_function));

void __cyg_profile_func_enter(void *func,  void *caller) {
	auto info = get_dl_info(func, caller);
	std::printf("> %p [%s] %s\n", func, info.dli_fname, info.dli_sname);
}
void __cyg_profile_func_exit (void *func, void *caller) {
	auto info = get_dl_info(func, caller);
	std::printf("< %p [%s] %s\n", func, info.dli_fname, info.dli_sname);
}
