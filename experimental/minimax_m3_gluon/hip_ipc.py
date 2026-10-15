"""Owned hipMalloc IPC allocations; no PyTorch private storage/IPC APIs."""

import ctypes as C
from pathlib import Path

import torch
import torch.distributed as dist

HIP_DEVICE_ATTRIBUTE_WALL_CLOCK_RATE = 10017  # kHz, HIP runtime API


class Handle(C.Structure):
    _fields_ = [("data", C.c_byte * 64)]


class PeerMemory:
    def __init__(self, nbytes, rank, world, group=None):
        self.group = group
        # PyTorch wheels can bundle HIP outside the dynamic loader search path.
        # Use the exact runtime already loaded by PyTorch, never a second HIP.
        mapped = [
            line.split()[-1]
            for line in Path("/proc/self/maps").read_text().splitlines()
            if "/libamdhip64.so" in line
        ]
        if not mapped:
            raise RuntimeError("PyTorch has not loaded the HIP runtime")
        self.hip = C.CDLL(mapped[0])
        signatures = {
            "hipMalloc": [C.POINTER(C.c_void_p), C.c_size_t],
            "hipMemset": [C.c_void_p, C.c_int, C.c_size_t],
            "hipFree": [C.c_void_p],
            "hipIpcGetMemHandle": [C.POINTER(Handle), C.c_void_p],
            "hipIpcOpenMemHandle": [C.POINTER(C.c_void_p), Handle, C.c_uint],
            "hipIpcCloseMemHandle": [C.c_void_p],
        }
        for name, args in signatures.items():
            fn = getattr(self.hip, name)
            fn.argtypes, fn.restype = args, C.c_int
        self.hip.hipGetErrorString.argtypes = [C.c_int]
        self.hip.hipGetErrorString.restype = C.c_char_p
        self.hip.hipDeviceGetAttribute.argtypes = [C.POINTER(C.c_int), C.c_int, C.c_int]
        self.hip.hipDeviceGetAttribute.restype = C.c_int
        rate = C.c_int()
        self.call(
            "hipDeviceGetAttribute",
            C.byref(rate),
            HIP_DEVICE_ATTRIBUTE_WALL_CLOCK_RATE,
            torch.cuda.current_device(),
        )
        self.wall_clock_khz = rate.value
        assert self.wall_clock_khz > 0
        self.ptr = C.c_void_p()
        self.remote = []
        self.closed = False
        self.call("hipMalloc", C.byref(self.ptr), nbytes)
        self.call("hipMemset", self.ptr, 0, nbytes)
        torch.cuda.synchronize()
        handle = Handle()
        self.call("hipIpcGetMemHandle", C.byref(handle), self.ptr)
        handles = [None] * world
        dist.all_gather_object(handles, bytes(handle), group=group)
        addresses = []
        for peer, data in enumerate(handles):
            if peer == rank:
                addresses.append(self.ptr.value)
            else:
                mapped = C.c_void_p()
                self.call(
                    "hipIpcOpenMemHandle",
                    C.byref(mapped),
                    Handle.from_buffer_copy(data),
                    1,
                )
                self.remote.append(mapped)
                addresses.append(mapped.value)
        self.addresses = torch.tensor(addresses, dtype=torch.int64, device="cuda")
        dist.barrier(group=group)

    def call(self, name, *args):
        code = getattr(self.hip, name)(*args)
        if code:
            raise RuntimeError(f"{name}: {self.hip.hipGetErrorString(code).decode()}")

    def close(self):
        if self.closed:
            return
        torch.cuda.synchronize()
        dist.barrier(group=self.group)
        for ptr in self.remote:
            self.call("hipIpcCloseMemHandle", ptr)
        self.call("hipFree", self.ptr)
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        # An asymmetric rank failure must propagate to the process supervisor.
        # Entering a collective here can hide that failure forever while other
        # ranks are waiting in a different collective. Leave the allocations
        # alive on failure; process teardown owns their eventual cleanup.
        if exc_type is None:
            self.close()


class DevicePointer:
    """Triton's data_ptr/dtype interface for an owned raw HIP allocation."""

    def __init__(self, address):
        self.address = address
        self.dtype = torch.uint64

    def data_ptr(self):
        return self.address
