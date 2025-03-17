#pragma once

#include <dlfcn.h>
#include <stdexcept>
#include <filesystem>
#include <sstream>
#include "lru_cache.h"
#include <memory>
#include <cstdlib>

static std::filesystem::path aiter_root_dir;
__inline__ void init_root_dir(){
    char* AITER_ROOT_DIR = std::getenv("AITER_ROOT_DIR");
    if (!AITER_ROOT_DIR){
        AITER_ROOT_DIR = std::getenv("HOME");
    }
    aiter_root_dir=std::filesystem::path(AITER_ROOT_DIR)/".aiter";
}

__inline__ std::filesystem::path get_root_dir(){
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
__inline__ std::string generateCmd(std::string& cmd, Args... args) {
    std::stringstream ss;
    ss << cmd << " ";
    ((ss << NAMED(args).toString() << " "), ...);
    return ss.str();
}

__inline__ std::pair<std::string, int> executeCmd(const std::string& cmd) {
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
    ReturnType call(Args... args) {
        auto func = reinterpret_cast<ReturnType(*)(Args...)>(getRawFunction("call"));
        return func(std::forward<Args>(args)...);
    }
};


template<typename... Args>
__inline__ void run_lib(std::string folder,Args... args) {
    auto AITER_MAX_CACHE_SIZE = getenv("AITER_MAX_CACHE_SIZE");
    if(!AITER_MAX_CACHE_SIZE){
        AITER_MAX_CACHE_SIZE = "-1";
    }
    int aiter_max_cache_size = atoi(AITER_MAX_CACHE_SIZE);
    static LRUCache<std::string, std::shared_ptr<SharedLibrary>> libs(aiter_max_cache_size);
    std::string lib_path = (aiter_root_dir/"build"/folder/"lib.so").string();
    libs.put(folder, std::make_shared<SharedLibrary>(lib_path));
    (*libs.get(folder))->call(std::forward<Args>(args)...);
}