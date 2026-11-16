#pragma once
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#elif defined(__linux__) || defined(__ANDROID__)
#include <dlfcn.h>
#include <link.h>
#include <sys/auxv.h>
#endif

namespace meguviz::detail {

enum class CodeImageKind { Empty, DynamicLibrary, MainExecutable, Unavailable };

// A normal loader reference, never a permanent PIN/NODELETE. The caller must
// keep the image loaded and synchronize unloading during acquire(). On POSIX,
// exported_anchor names a real exported symbol in the address's image: NOLOAD
// alone proves a pathname match, not that the acquired handle is this image.
//
// Final release must execute outside that image, or while a separate host
// loader/execution reference still keeps the image loaded through return. An
// internal RAII object cannot safely unload its own last executing code image.
// This lease does not establish GPU retirement, stop threads or make callbacks
// in another module safe. Unknown image identity fails closed.
class CodeImageLease final {
public:
    CodeImageLease() noexcept = default;
    ~CodeImageLease() { (void)reset(); }
    CodeImageLease(const CodeImageLease&) = delete;
    CodeImageLease& operator=(const CodeImageLease&) = delete;
    CodeImageLease(CodeImageLease&& other) noexcept
        : handle_(std::exchange(other.handle_, nullptr)),
          kind_(std::exchange(other.kind_, CodeImageKind::Empty)) {}
    CodeImageLease& operator=(CodeImageLease&&) = delete;

    CodeImageKind kind() const noexcept { return kind_; }
    bool available() const noexcept {
        return kind_ == CodeImageKind::DynamicLibrary || kind_ == CodeImageKind::MainExecutable;
    }

    // An already acquired lease is left intact; reset explicitly before reuse.
    CodeImageKind acquire(const void* address, const char* exported_anchor = nullptr) noexcept {
        if (handle_ || available()) return kind_;
        kind_ = CodeImageKind::Unavailable;
        if (!address) return kind_;
#if defined(_WIN32)
        (void)exported_anchor;
        HMODULE module = nullptr;
        if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS,
                              reinterpret_cast<LPCWSTR>(address), &module)) return kind_;
        handle_ = module;
        kind_ = module == GetModuleHandleW(nullptr) ? CodeImageKind::MainExecutable
                                                  : CodeImageKind::DynamicLibrary;
#elif defined(__linux__) || defined(__ANDROID__)
        if (is_main_executable(address)) {
            kind_ = CodeImageKind::MainExecutable;
            return kind_;
        }
        Dl_info before{};
        if (!exported_anchor || !*exported_anchor ||
            !dladdr(address, &before) || !before.dli_fbase ||
            !before.dli_fname || !*before.dli_fname) return kind_;
        // Never fall back to ordinary dlopen: that could load a different image
        // from a renamed/replaced path, or cross an unintended linker namespace.
        void* module = dlopen(before.dli_fname, RTLD_NOW | RTLD_LOCAL | RTLD_NOLOAD);
        if (!module) return kind_;
        void* symbol = dlsym(module, exported_anchor);
        Dl_info anchor{}, after{};
        if (!symbol || !dladdr(symbol, &anchor) || !dladdr(address, &after) ||
            anchor.dli_fbase != before.dli_fbase || after.dli_fbase != before.dli_fbase) {
            // A failed rollback keeps our counted reference for reset(), but
            // never converts an unverified image into an available lease.
            if (dlclose(module) != 0) handle_ = module;
            return kind_;
        }
        handle_ = module;
        kind_ = CodeImageKind::DynamicLibrary;
#else
        (void)exported_anchor;
#endif
        return kind_;
    }

    // On a loader error retain the handle for an explicit retry. The destructor
    // attempts release but cannot report an error; lifecycle owners that need a
    // close result must call reset() and check it before destroying the lease.
    bool reset() noexcept {
        if (handle_) {
#if defined(_WIN32)
            if (!FreeLibrary(static_cast<HMODULE>(handle_))) return false;
#elif defined(__linux__) || defined(__ANDROID__)
            if (dlclose(handle_) != 0) return false;
#else
            return false;
#endif
            handle_ = nullptr;
        }
        kind_ = CodeImageKind::Empty;
        return true;
    }

private:
#if defined(__linux__) || defined(__ANDROID__)
    struct MainImage {
        const ElfW(Phdr)* headers;
        std::uintptr_t address;
        bool contains = false;
    };
    static int find_main(dl_phdr_info* info, std::size_t, void* context) noexcept {
        auto& main = *static_cast<MainImage*>(context);
        if (info->dlpi_phdr != main.headers) return 0;
        for (ElfW(Half) i = 0; i < info->dlpi_phnum; ++i) {
            const auto& segment = info->dlpi_phdr[i];
            if (segment.p_type != PT_LOAD) continue;
            const auto base = static_cast<std::uintptr_t>(info->dlpi_addr);
            const auto offset = static_cast<std::uintptr_t>(segment.p_vaddr);
            if (base > std::numeric_limits<std::uintptr_t>::max() - offset) continue;
            const auto begin = base + offset;
            if (main.address >= begin && main.address - begin < segment.p_memsz) {
                main.contains = true;
                break;
            }
        }
        return 1;
    }
    static bool is_main_executable(const void* address) noexcept {
        const auto* headers = reinterpret_cast<const ElfW(Phdr)*>(getauxval(AT_PHDR));
        if (!headers) return false;
        MainImage main{headers, reinterpret_cast<std::uintptr_t>(address)};
        dl_iterate_phdr(find_main, &main);
        return main.contains;
    }
#endif
    void* handle_ = nullptr;
    CodeImageKind kind_ = CodeImageKind::Empty;
};
}
