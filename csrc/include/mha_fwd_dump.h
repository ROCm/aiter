#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Runtime dump of mha forward args (torch-free core).
//
// Env vars:
//   AITER_DUMP_MHA_FWD_INFO
//     unset / 0  => disabled
//     N (>0)     => sample: emit once every N calls (N==1 means every call)
//
//   AITER_DUMP_MHA_FWD_INFO_FILE
//     Path to write the collected shape/args log to. When dumping is
//     enabled but this variable is unset, the default path is
//     "/tmp/mha_dump_info_<pid>.log" (<pid> is the running process id).
//     When the variable is set, its value is used verbatim: no shell-like
//     expansion is performed. The file is opened in append mode and
//     every line is flushed immediately, so partial data survives an
//     external SIGKILL.
//
// Thread-safety:
//   - Sampling counter is thread_local (lock-free).
//   - The final emission to the sink FILE* is guarded by a shared
//     std::mutex so that log lines from different threads never
//     interleave.
//
// This header intentionally has no torch dependencies so it can be included
// from files (e.g. csrc/cpp_itfs/mha_fwd.cu) that must not pull in torch.
// The group-mode dumper (which needs at::Tensor) lives in mha_common.h and
// reuses the primitives defined here.

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <unistd.h>

#include "mha_fwd.h"

namespace aiter {

inline int get_mha_dump_stride()
{
    static const int stride = [] {
        const char* env = std::getenv("AITER_DUMP_MHA_FWD_INFO");
        if(!env || env[0] == '\0')
            return 0;
        int v = std::atoi(env);
        return v > 0 ? v : 0;
    }();
    return stride;
}

inline bool mha_dump_should_emit()
{
    const int stride = get_mha_dump_stride();
    if(stride == 0)
        return false;
    thread_local uint64_t counter = 0;
    return (counter++ % static_cast<uint64_t>(stride)) == 0;
}

inline std::mutex& mha_dump_mutex()
{
    static std::mutex m;
    return m;
}

// Resolve the output-file path once. When AITER_DUMP_MHA_FWD_INFO_FILE is
// set, its value is used verbatim (no shell-like expansion). Otherwise
// the default "/tmp/mha_dump_info_<pid>.log" is used, where <pid> is the
// running process id. An empty return means "fall back to std::cerr"
// (only when the user explicitly sets the variable to an empty string).
inline const std::string& mha_dump_output_path()
{
    static const std::string path = [] {
        const char* env = std::getenv("AITER_DUMP_MHA_FWD_INFO_FILE");
        if(env && env[0] != '\0')
            return std::string(env);
        return std::string("/tmp/mha_dump_info_") +
               std::to_string(static_cast<long>(getpid())) + ".log";
    }();
    return path;
}

// One-time initialised sink FILE*. Returns nullptr if the file could not
// be opened, in which case callers should fall back to std::cerr. The
// FILE* lives inside a function-local static so the `std::atexit` handler
// registered on first open can safely close it during normal shutdown.
inline std::FILE* mha_dump_sink()
{
    static std::FILE* fp = nullptr;
    static std::once_flag once;
    std::call_once(once, [] {
        const std::string& path = mha_dump_output_path();
        if(!path.empty())
            fp = std::fopen(path.c_str(), "a");
        if(fp)
        {
            // Announce the sink path once, so operators know where to
            // look. Written to stderr because at this point the file is
            // still empty and users need the pointer regardless.
            std::fprintf(stderr,
                         "[MHA_FWD] AITER_DUMP_MHA_FWD_INFO enabled, "
                         "writing to %s\n",
                         path.c_str());
            // Flush residual buffer on normal shutdown. SIGKILL bypasses
            // this, but per-write fflush() below still keeps the log
            // durable on disk up to the last completed line.
            std::atexit([] {
                if(std::FILE* g = mha_dump_sink())
                    std::fclose(g);
            });
        }
        else
        {
            std::fprintf(stderr,
                         "[MHA_FWD] AITER_DUMP_MHA_FWD_INFO enabled but "
                         "failed to open '%s' for append (%s); falling "
                         "back to stderr.\n",
                         path.c_str(),
                         std::strerror(errno));
        }
    });
    return fp;
}

// Write one already-formatted record to the configured sink, holding the
// shared mutex so multi-thread output cannot interleave, and flushing
// immediately so external kills do not lose in-flight lines.
inline void mha_dump_write(const std::string& line)
{
    std::FILE* fp = mha_dump_sink();
    std::lock_guard<std::mutex> lk(mha_dump_mutex());
    if(fp)
    {
        std::fwrite(line.data(), 1, line.size(), fp);
        std::fflush(fp);
    }
    else
    {
        std::cerr << line;
    }
}

// Append fields shared by batch/group modes.
inline void append_mha_common_fields(std::ostringstream& os,
                                     const mha_fwd_args& a,
                                     const char* mode)
{
    os << "[MHA_FWD]"
       << " mode=" << mode
       << " dtype=" << a.data_type
       << " hdim_q=" << a.hdim_q
       << " hdim_v=" << a.hdim_v
       << " nhead_q=" << a.nhead_q
       << " nhead_k=" << a.nhead_k
       << " batch=" << a.batch
       << " max_seqlen_q=" << a.max_seqlen_q
       << " mask_type=" << a.mask_type
       << " bias_type=" << a.bias_type
       << " has_lse=" << (a.has_lse ? 1 : 0)
       << " has_dropout=" << ((a.p_drop > 0.f) ? 1 : 0);
}

// Batch-mode dumper: only two extra scalar seqlens.
inline void dump_mha_fwd_info_batch(const mha_fwd_args& a)
{
    if(!mha_dump_should_emit())
        return;
    std::ostringstream os;
    append_mha_common_fields(os, a, "batch");
    os << " seqlen_q=" << a.seqlen_q << " seqlen_k=" << a.seqlen_k << "\n";
    mha_dump_write(os.str());
}

} // namespace aiter
