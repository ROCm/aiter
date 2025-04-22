#pragma once

#include <dlfcn.h>
#include <stdexcept>
#include <filesystem>
#include <sstream>
#include "lru_cache.h"
#include <memory>
#include <cstdlib>
#include <openssl/md5.h>  
#include <iomanip> 
#include <fmt/ranges.h>
#include <mutex>


namespace aiter{

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

static std::filesystem::path aiter_root_dir;

__inline__ void init_root_dir(){
    char* AITER_ROOT_DIR = std::getenv("AITER_ROOT_DIR");
    if (!AITER_ROOT_DIR){
        AITER_ROOT_DIR = std::getenv("HOME");
    }
    aiter_root_dir=std::filesystem::path(AITER_ROOT_DIR)/".aiter";
}

__inline__ std::filesystem::path get_root_dir(){
    std::call_once(init_root_dir_flag, init_root_dir);
    return aiter_root_dir;
}

template<typename T>
class NamedArg {
    const char* name;
    T value;
public:
    NamedArg(const char* n, T v) : name(n), value(v) {}
    
    std::string toString() const {
        std::stringstream ss;
        ss << "--" << name << "=" << value;
        return ss.str();
    }
};

#define NAMED(x) NamedArg(#x, x)

template<typename... Args>
__inline__ std::string generate_cmd(std::string& cmd, Args... args) {
    std::stringstream ss;
    ss << cmd << " ";
    ((ss << NAMED(args).toString() << " "), ...);
    return ss.str();
}

__inline__ std::pair<std::string, int> execute_cmd(const std::string& cmd) {
    std::array<char, 128> buffer;
    std::string result;
    int exitCode;
    
    FILE* pipe = popen(cmd.c_str(), "r");
    
    if (!pipe) {
        throw std::runtime_error("popen() failed!");
    }
    
    try {
        while (fgets(buffer.data(), buffer.size(), pipe) != nullptr) {
            result += buffer.data();
        }
    } catch (...) {
        pclose(pipe);
        throw;
    }
    
    exitCode = pclose(pipe);
    return {result, exitCode};
}


class SharedLibrary {
private:
    void* handle;

public:
    SharedLibrary(std::string& path) {
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
    ReturnType call(std::string& func_name, Args... args) {
        auto func = reinterpret_cast<ReturnType(*)(Args...)>(getRawFunction(func_name.c_str()));
        return func(std::forward<Args>(args)...);
    }
};

static std::unique_ptr<LRUCache<std::string, std::shared_ptr<SharedLibrary>>> libs;
static std::unique_ptr<LRUCache<std::string, std::string>> func_names;

template<typename... Args>
__inline__ void run_lib(std::string func_name, std::string folder, Args... args) {
    std::call_once(init_libs_lru_cache, init_lru_cache<std::string, std::shared_ptr<SharedLibrary>>, libs);
    auto func_lib = libs->get(func_name);
    if(!func_lib){
        std::string lib_path = (get_root_dir()/"build"/folder/"lib.so").string();
        libs->put(func_name, std::make_shared<SharedLibrary>(lib_path));
        func_lib = libs->get(func_name);
    }
    (*func_lib)->call(func_name, std::forward<Args>(args)...);
}


__inline__ std::string hash_signature(const std::string& signature) {
    unsigned char digest[MD5_DIGEST_LENGTH];
    MD5_CTX context;

    // Compute the MD5 hash
    MD5_Init(&context);
    MD5_Update(&context, signature.data(), signature.size());
    MD5_Final(digest, &context);

    // Convert binary digest to hex string
    std::stringstream ss;
    for (int i = 0; i < MD5_DIGEST_LENGTH; i++) {
        ss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(digest[i]);
    }
    return ss.str();
}


__inline__ std::string get_default_func_name(const std::string& md_name, std::vector<std::string>& args) {
    std::call_once(init_func_names_lru_cache, init_lru_cache<std::string, std::string>, func_names);
    std::string args_str = fmt::format("{}", fmt::join(args, "_"));
    auto func_name = func_names->get(args_str);
    if(!func_name){
        func_names->put(args_str, fmt::format("{}_{}", md_name, hash_signature(args_str)));
        func_name = func_names->get(args_str);
    }
    return *func_name;
}


__inline__ bool not_built(const std::string& folder) {
    return !std::filesystem::exists(get_root_dir() / "build" / folder / "lib.so");
}
}