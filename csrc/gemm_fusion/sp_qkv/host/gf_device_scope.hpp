// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// Process-global HIP state hygiene for the gemm_fusion host code (plain clang++ -x c++; includes <hip/hip_runtime.h> only).
//   * The runtime is embedded in a framework process (torch/ATOM) that owns the thread's CURRENT DEVICE. This code must never leave it
//     changed: DeviceScope makes `device` current for one call and restores the caller's device on exit.
//   * The device index is the HIP device of the communicator (ccoComm::hipDev, taken by MORI from hipGetDevice() at ccoCommCreate), NOT the
//     SP-group rank: with TP x SP, several groups or HIP_VISIBLE_DEVICES remapping the two differ.
//   * A failed/unsupported HIP call leaves a sticky last-error that a LATER torch op can pick up via hipGetLastError() and raise as
//     "operation not supported" (MORI documents this for hipDeviceEnablePeerAccess on gfx950). Every entry point therefore flushes the sticky
//     error on exit (flush_sticky) so this library never leaks one into the host process.
#pragma once
#include <hip/hip_runtime.h>

namespace gf {

// Device the thread-local override points at while a scope is active (-1 = none). Used by the macro that reroutes the unmodified Phase-1
// driver's hipSetDevice(c->rank) calls (see sm_fusion_driver.cpp).
inline thread_local int g_dev = -1;

// hipSetDevice replacement for code we must not edit: while a DeviceScope is active every requested index is replaced by the scope's device.
inline hipError_t set_device(int requested) { return hipSetDevice(g_dev >= 0 ? g_dev : requested); }

// Consume (and return) the sticky runtime error so it cannot surface in a later framework call.
inline hipError_t flush_sticky() { return hipGetLastError(); }

class DeviceScope {
public:
    // device < 0: keep the caller's current device (no switch).
    explicit DeviceScope(int device) : prev_g_dev_(g_dev) {
        if (device < 0) return;
        if (hipGetDevice(&prev_) != hipSuccess) prev_ = -1;
        if (prev_ != device) {
            err_ = hipSetDevice(device);
            changed_ = (err_ == hipSuccess);
        }
        g_dev = device;
    }
    ~DeviceScope() {
        if (changed_ && prev_ >= 0) (void)hipSetDevice(prev_);
        g_dev = prev_g_dev_;
    }
    DeviceScope(const DeviceScope&) = delete;
    DeviceScope& operator=(const DeviceScope&) = delete;
    bool ok() const { return err_ == hipSuccess; }
    hipError_t err() const { return err_; }
private:
    int prev_ = -1, prev_g_dev_ = -1;
    bool changed_ = false;
    hipError_t err_ = hipSuccess;
};

}  // namespace gf