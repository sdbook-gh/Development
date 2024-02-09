#include <cstdio>

static FILE* outfile = nullptr;

extern "C" {
    void start_tracingnew() {
        outfile = fopen("outnew", "wb");
    }
    void do_testingnew() {
    }
    void stop_tracingnew() {
        fclose(outfile);
        outfile = nullptr;
    }
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
