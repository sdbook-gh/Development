#include <cstdio>
#include <algorithm>
#include <fstream>

struct C0 {
	C0() {}
};

struct C1 {
	C1() = default;
	virtual void doit() {}
	virtual ~C1() = default;
};
struct C2 : C1 {
	virtual void doit() override {}
	virtual ~C2() = default;
};

namespace {
	void fun_in_unnamed_namespace(C1& c1) {
		c1.doit();
	}
}

extern "C" {
    void start_tracing();
    void do_testing();
    void stop_tracing();
    void start_tracingnew();
    void do_testingnew();
    void stop_tracingnew();
};

// FILE* outfile = nullptr;

// extern "C" {
// 	void __cyg_profile_func_enter(void *this_fn, void *call_site) __attribute__((no_instrument_function));
// 	void __cyg_profile_func_exit(void *this_fn, void *call_site) __attribute__((no_instrument_function));

// 	void __cyg_profile_func_enter(void *func,  void *caller) {
// 		static const int unused = []() __attribute__((no_instrument_function)) {
// 			void* enter = (void*)(&__cyg_profile_func_enter);
// 			std::printf("%p\n\n", enter);
// 			return 0;
// 		}(); (void) unused;
// 		if (outfile == nullptr) return;
// 		// std::printf("> %p\n", func);
// 		fprintf(outfile, ">>");
// 		fprintf(outfile, "%p\n", func);
// 	}
// 	void __cyg_profile_func_exit (void *func, void *caller) {
// 		if (outfile == nullptr) return;
// 		// std::printf("< %p\n", func);
// 		fprintf(outfile, "<<");
// 		fprintf(outfile, "%p\n", func);
// 	}
// };

void throwing_foo() {
	throw 1;
}


using callback = int (*)(int);

constexpr int foo(int a) {
	return a;
}

void foo_with_callback(callback f) {
	f(1);
	int arr[]{1,2,3,4};
	std::sort(arr, arr+3);
}

int main(int argc, char**) {
	// outfile = fopen("out", "wb");
	start_tracing();
	do_testing();
	stop_tracing();
	for (auto i = 0; i < 1000000; i++) {
	try {
		C0 c0;
		C1 c1;
		C2 c2;
		fun_in_unnamed_namespace(c1);
		fun_in_unnamed_namespace(c2);
		foo_with_callback(foo);
		constexpr int _1 = foo(1);
		int _2 = foo(argc);

		auto l = [](){
			return 12;
		};
		auto _3 = l();
		foo_with_callback([](int){return 1;});
		throwing_foo();
	} catch(...) {
	}
	}
	start_tracingnew();
	do_testingnew();
	stop_tracingnew();
	// fclose(outfile);
	return 0;
}
