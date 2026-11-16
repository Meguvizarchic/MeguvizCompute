#include "../src/core/code_image_lease.hpp"
#include <cstdio>
#include <utility>
using meguviz::detail::CodeImageKind;
using meguviz::detail::CodeImageLease;
#define REQUIRE(condition) do { if (!(condition)) { \
    std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #condition); return 1; } } while (0)
static int executable_anchor() { return 7; }

int main(int argc, char** argv) {
    REQUIRE(argc == 2);
    CodeImageLease executable, invalid;
    REQUIRE(invalid.acquire(nullptr) == CodeImageKind::Unavailable && !invalid.available());
    REQUIRE(executable.acquire(reinterpret_cast<const void*>(&executable_anchor)) == CodeImageKind::MainExecutable);
    REQUIRE(executable.reset());
#if defined(_WIN32)
    auto module = LoadLibraryA(argv[1]);
    REQUIRE(module);
    auto symbol = GetProcAddress(module, "mgc_code_image_fixture");
#else
    auto module = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    REQUIRE(module);
    auto symbol = dlsym(module, "mgc_code_image_fixture");
#endif
    REQUIRE(symbol);
    auto call = reinterpret_cast<int(*)()>(symbol);
    CodeImageLease image;
    REQUIRE(image.acquire(reinterpret_cast<const void*>(symbol), "mgc_code_image_fixture") == CodeImageKind::DynamicLibrary);
#if defined(_WIN32)
    REQUIRE(FreeLibrary(module));
#else
    REQUIRE(dlclose(module) == 0);
#endif
    // Only the test host owns/executes this lease implementation at final drop.
    // The fixture's caller reference is gone, so successful execution proves
    // that acquiring the image added a real loader reference.
    REQUIRE(call() == 42);
    CodeImageLease moved(std::move(image));
    REQUIRE(image.kind() == CodeImageKind::Empty && moved.available());
    REQUIRE(moved.reset() && moved.kind() == CodeImageKind::Empty);
#if defined(_WIN32)
    HMODULE still_loaded = nullptr;
    REQUIRE(!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
        GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
        reinterpret_cast<LPCWSTR>(symbol), &still_loaded));
#endif
    std::puts("PASS main executable, invalid input, actual loader retention, move, release");
    return 0;
}
