#pragma once

#include <dlfcn.h>
#include <spawn.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
#include <stdexcept>
#include <filesystem>
#include <sstream>
#include "lru_cache.h"
#include <memory>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <openssl/evp.h>
#include <iomanip>
#include <fmt/ranges.h>
#include <mutex>
#include <cctype>
#include <algorithm>
#include <array>
#include <string_view>
#include <utility>
#include <vector>

extern "C" char** environ;

// Set -DAITER_ENABLE_JIT=0 to build without the runtime code-generation path.
// Packaged builds should do this and ship prebuilt libraries under
// -DAITER_INSTALL_LIBDIR="<prefix>" instead.
#ifndef AITER_ENABLE_JIT
#define AITER_ENABLE_JIT 1
#endif

namespace aiter{

#define DIVIDE_ROUND_UP(a, b) (((a) + (b)-1) / (b))

static std::once_flag init_libs_lru_cache, init_func_names_lru_cache, init_root_dir_flag;

template<typename K, typename V>
__inline__ void init_lru_cache(std::unique_ptr<LRUCache<K, V>>& lru_cache){
    auto AITER_MAX_CACHE_SIZE = getenv("AITER_MAX_CACHE_SIZE");
    if(!AITER_MAX_CACHE_SIZE){
        AITER_MAX_CACHE_SIZE = "-1";
    }
    int aiter_max_cache_size = atoi(AITER_MAX_CACHE_SIZE);
    lru_cache = std::make_unique<LRUCache<K, V>>(aiter_max_cache_size);
}

// Every value that reaches a command line or a cache path is a kernel
// identifier: a dtype spelling, an integer, a bool, or a hashed function name.
// Restricting them to this charset keeps path separators and "." out of file
// paths, so a folder name can never escape its parent directory.
__inline__ bool is_safe_token(std::string_view token){
    if(token.empty() || token.size() > 64){
        return false;
    }
    return std::all_of(token.begin(), token.end(), [](unsigned char c){
        return std::isalnum(c) != 0 || c == '_' || c == '-';
    });
}

// Rejected values are deliberately not echoed back: they are attacker-influenced
// and would land in logs verbatim.
__inline__ const std::string& check_token(const std::string& token, const char* what){
    if(!is_safe_token(token)){
        throw std::invalid_argument(
            fmt::format("aiter: invalid {} (expected [A-Za-z0-9_-]{{1,64}})", what));
    }
    return token;
}

__inline__ bool is_safe_module(std::string_view name){
    if(name.empty() || name.size() > 128){
        return false;
    }
    if(name.front() == '.' || name.back() == '.' || name.find("..") != std::string_view::npos){
        return false;
    }
    return std::all_of(name.begin(), name.end(), [](unsigned char c){
        return std::isalnum(c) != 0 || c == '_' || c == '.';
    });
}

// A directory or library owned by another unprivileged user, or writable by
// group/other, lets a co-tenant swap in code that we would then dlopen().
__inline__ bool is_writable_by_others(const struct stat& st){
    return (st.st_mode & (S_IWGRP | S_IWOTH)) != 0;
}

__inline__ bool is_foreign_owner(const struct stat& st){
    return st.st_uid != ::geteuid() && st.st_uid != 0;
}

__inline__ void check_trusted_dir(const std::filesystem::path& dir){
    struct stat st{};
    if(::stat(dir.c_str(), &st) != 0){
        return; // Does not exist yet; it will be created with our own umask.
    }
    if(is_foreign_owner(st) || is_writable_by_others(st)){
        throw std::runtime_error(
            fmt::format("aiter: refusing to use \"{}\": not owned by the current user "
                        "or writable by group/other", dir.string()));
    }
}

__inline__ void check_trusted_file(const std::filesystem::path& file){
    struct stat st{};
    if(::lstat(file.c_str(), &st) != 0){
        throw std::runtime_error(fmt::format("aiter: cannot stat \"{}\"", file.string()));
    }
    if(S_ISLNK(st.st_mode)){
        throw std::runtime_error(
            fmt::format("aiter: refusing to load \"{}\": symlink", file.string()));
    }
    if(is_foreign_owner(st) || is_writable_by_others(st)){
        throw std::runtime_error(
            fmt::format("aiter: refusing to load \"{}\": not owned by the current user "
                        "or writable by group/other", file.string()));
    }
}

static std::filesystem::path aiter_root_dir;

__inline__ void init_root_dir(){
    const char* root = nullptr;
    // A setuid/setgid process must not take its library search root from the
    // environment of whoever invoked it.
    const bool privileged = ::geteuid() != ::getuid() || ::getegid() != ::getgid();
    if(!privileged){
        root = std::getenv("AITER_ROOT_DIR");
        if(!root){
            root = std::getenv("HOME");
        }
    }
    if(!root || *root == '\0'){
        throw std::runtime_error(
            "aiter: cannot determine the JIT cache root; set AITER_ROOT_DIR "
            "(ignored for setuid/setgid processes)");
    }
    std::filesystem::path candidate = std::filesystem::path(root) / ".aiter";
    check_trusted_dir(candidate);
    aiter_root_dir = std::move(candidate);
}

__inline__ std::filesystem::path get_root_dir(){
    std::call_once(init_root_dir_flag, init_root_dir);
    return aiter_root_dir;
}

__inline__ std::filesystem::path get_build_dir(){
    return get_root_dir() / "build";
}

// Runs argv[0] directly. There is no shell, so no argument can be interpreted
// as a metacharacter regardless of its contents.
__inline__ const std::pair<std::string, int> execute_argv(const std::vector<std::string>& argv){
    if(argv.empty()){
        throw std::invalid_argument("aiter: empty argv");
    }

    int fds[2];
    if(::pipe(fds) != 0){
        throw std::runtime_error("aiter: pipe() failed");
    }

    posix_spawn_file_actions_t actions;
    posix_spawn_file_actions_init(&actions);
    posix_spawn_file_actions_addclose(&actions, fds[0]);
    posix_spawn_file_actions_adddup2(&actions, fds[1], STDOUT_FILENO);
    posix_spawn_file_actions_adddup2(&actions, fds[1], STDERR_FILENO);
    posix_spawn_file_actions_addclose(&actions, fds[1]);

    std::vector<char*> cargv;
    cargv.reserve(argv.size() + 1);
    for(const auto& arg : argv){
        cargv.push_back(const_cast<char*>(arg.c_str()));
    }
    cargv.push_back(nullptr);

    pid_t pid          = 0;
    const int spawn_rc = ::posix_spawnp(&pid, cargv[0], &actions, nullptr, cargv.data(), environ);
    posix_spawn_file_actions_destroy(&actions);
    ::close(fds[1]);

    if(spawn_rc != 0){
        ::close(fds[0]);
        throw std::runtime_error(
            fmt::format("aiter: failed to run \"{}\": {}", argv[0], std::strerror(spawn_rc)));
    }

    std::string output;
    std::array<char, 4096> buffer{};
    ssize_t nread;
    while((nread = ::read(fds[0], buffer.data(), buffer.size())) > 0){
        output.append(buffer.data(), static_cast<size_t>(nread));
    }
    ::close(fds[0]);

    int status = 0;
    while(::waitpid(pid, &status, 0) < 0 && errno == EINTR){
    }
    return {output, WIFEXITED(status) ? WEXITSTATUS(status) : -1};
}

// Invokes "python3 -m <module> --<key>=<value> ...".
__inline__ void run_codegen(const std::string& module,
                            const std::vector<std::pair<std::string, std::string>>& options){
#if !AITER_ENABLE_JIT
    (void)options;
    throw std::runtime_error(
        fmt::format("aiter: no prebuilt library for \"{}\" and JIT is disabled "
                    "(built with AITER_ENABLE_JIT=0)", module));
#else
    if(!is_safe_module(module)){
        throw std::invalid_argument("aiter: invalid codegen module name");
    }

    std::vector<std::string> argv{"python3", "-m", module};
    argv.reserve(options.size() + 3);
    for(const auto& [key, value] : options){
        argv.push_back(fmt::format("--{}={}",
                                   check_token(key, "codegen option name"),
                                   check_token(value, "codegen option value")));
    }

    AITER_LOG_INFO(fmt::format("{}", fmt::join(argv, " ")));
    const auto [output, exit_code] = execute_argv(argv);
    AITER_LOG_INFO(output);
    if(exit_code != 0){
        throw std::runtime_error(
            fmt::format("aiter: codegen for \"{}\" failed with exit code {}", module, exit_code));
    }
#endif
}

class SharedLibrary {
private:
    void* handle;

public:
    SharedLibrary(const std::string& path) {
        handle = dlopen(path.c_str(), RTLD_LAZY);
        if (!handle) {
            throw std::runtime_error(dlerror());
        }
    }

    ~SharedLibrary() {
        if (handle) {
            dlclose(handle);
        }
    }

    // Get raw function pointer
    void* getRawFunction(const char* funcName) {
        dlerror(); // Clear any existing error
        void* funcPtr = dlsym(handle, funcName);
        const char* error = dlerror();
        if (error) {
            throw std::runtime_error(error);
        }
        return funcPtr;
    }

    // Template to call function with any return type and arguments
    template<typename ReturnType = void, typename... Args>
    ReturnType call(const std::string& func_name, Args... args) {
        auto func = reinterpret_cast<ReturnType(*)(Args...)>(getRawFunction(func_name.c_str()));
        return func(std::forward<Args>(args)...);
    }
};

// Resolves <base>/<folder>/lib.so and proves the result is still under <base>
// after symlinks are resolved. Containment is compared component-wise; a string
// prefix test would accept a sibling such as "<base>-attacker".
__inline__ std::filesystem::path resolve_lib_path(const std::filesystem::path& base,
                                                  const std::string& folder){
    check_token(folder, "folder");

    std::error_code ec;
    const auto base_canon = std::filesystem::weakly_canonical(base, ec);
    if(ec){
        throw std::runtime_error(fmt::format("aiter: cannot resolve \"{}\"", base.string()));
    }
    const auto full = std::filesystem::weakly_canonical(base_canon / folder / "lib.so", ec);
    if(ec){
        throw std::runtime_error(fmt::format("aiter: cannot resolve a library under \"{}\"",
                                             base_canon.string()));
    }

    const auto mismatched =
        std::mismatch(base_canon.begin(), base_canon.end(), full.begin(), full.end());
    if(mismatched.first != base_canon.end()){
        throw std::runtime_error(
            fmt::format("aiter: refusing to load a library outside \"{}\"", base_canon.string()));
    }
    return full;
}

// Prefers a read-only packaged library over the per-user JIT cache.
__inline__ std::filesystem::path resolve_lib(const std::string& folder){
#ifdef AITER_INSTALL_LIBDIR
    {
        std::error_code ec;
        auto packaged = resolve_lib_path(AITER_INSTALL_LIBDIR, folder);
        if(std::filesystem::exists(packaged, ec)){
            return packaged;
        }
    }
#endif
    return resolve_lib_path(get_build_dir(), folder);
}

static std::unique_ptr<LRUCache<std::string, std::shared_ptr<SharedLibrary>>> libs;
static std::unique_ptr<LRUCache<std::string, std::string>> func_names;

template<typename... Args>
__inline__ void run_lib(std::string func_name, std::string folder, Args... args) {
    std::call_once(init_libs_lru_cache, init_lru_cache<std::string, std::shared_ptr<SharedLibrary>>, libs);
    auto func_lib = libs->get(func_name);
    if(!func_lib){
        const auto lib_path = resolve_lib(folder);
        check_trusted_file(lib_path);
        libs->put(func_name, std::make_shared<SharedLibrary>(lib_path.string()));
        func_lib = libs->get(func_name);
    }
    (*func_lib)->call(func_name, std::forward<Args>(args)...);
}


__inline__ std::string hash_signature(const std::string& signature) {
    unsigned char digest[EVP_MAX_MD_SIZE];
    unsigned int digest_len;

    EVP_MD_CTX* ctx = EVP_MD_CTX_new();
    EVP_DigestInit_ex(ctx, EVP_md5(), NULL);
    EVP_DigestUpdate(ctx, signature.data(), signature.size());
    EVP_DigestFinal_ex(ctx, digest, &digest_len);
    EVP_MD_CTX_free(ctx);

    std::stringstream ss;
    for (unsigned int i = 0; i < digest_len; i++) {
        ss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(digest[i]);
    }
    return ss.str();
}


__inline__ std::string get_default_func_name(const std::string& md_name, std::list<std::string>& args) {
    std::call_once(init_func_names_lru_cache, init_lru_cache<std::string, std::string>, func_names);
    std::string args_str = fmt::format("{}", fmt::join(args, "_"));
    std::transform(args_str.begin(), args_str.end(), args_str.begin(),
    [](unsigned char c){ return std::tolower(c); });
    auto func_name = func_names->get(args_str);
    if(!func_name){
        func_names->put(args_str, fmt::format("{}_{}", md_name, hash_signature(args_str)));
        func_name = func_names->get(args_str);
    }
    return *func_name;
}


__inline__ bool not_built(const std::string& folder) {
    std::error_code ec;
#ifdef AITER_INSTALL_LIBDIR
    if(std::filesystem::exists(resolve_lib_path(AITER_INSTALL_LIBDIR, folder), ec)){
        return false;
    }
#endif
#if !AITER_ENABLE_JIT
    // Nothing packaged, and there is no cache to consult; let the caller's
    // codegen path report that JIT is disabled.
    (void)ec;
    check_token(folder, "folder");
    return true;
#else
    return !std::filesystem::exists(resolve_lib_path(get_build_dir(), folder), ec);
#endif
}
}
