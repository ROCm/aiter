// lamport_ipc.cpp -- pybind11 module (_lamport_ipc) for the Gluon push
// probe (gluon_lamport.py). 2026-11-15.
//
// Same exchange as trms_ipc.cpp's FusedIpcBuffer, but the allocation is
// UNCACHED device memory (hipExtMallocWithFlags + hipDeviceMallocUncached)
// filled with one byte value. A receiver polls its own inbox with plain loads;
// on gfx950 only uncached memory makes peers' XGMI stores visible to those
// loads without cache maintenance (Gluon CDNA4 cache hints never affect
// coherence). aiter's gfx9 Lamport kernels use the same kind of inbox.
//
//   UncachedIpcBuffer(rank, nGPU, bytes, fill_byte)
//     .exchange(gloo_group)   collective
//     .get_peer_bases_cpu()   int64 CPU tensor[nGPU]
//     .local_view()           uint8 CUDA tensor over the local allocation

#include <hip/hip_runtime.h>
#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include <cstring>
#include <stdexcept>
#include <vector>

#include "hip_check.h"

namespace py = pybind11;

class UncachedIpcBuffer {
public:
    UncachedIpcBuffer(int rank, int nGPU, size_t bytes, int fill)
        : rank_(rank), nGPU_(nGPU), bytes_(bytes)
    {
        HIP_CALL(hipSetDevice(rank_));
        HIP_CALL(hipExtMallocWithFlags(&base_, bytes_, hipDeviceMallocUncached));
        HIP_CALL(hipMemset(base_, fill, bytes_));
        HIP_CALL(hipDeviceSynchronize());
    }

    ~UncachedIpcBuffer()
    {
        (void)hipSetDevice(rank_);
        for (int r = 0; r < (int)peers_.size(); ++r)
            if (r != rank_ && peers_[r]) (void)hipIpcCloseMemHandle(peers_[r]);
        if (base_) (void)hipFree(base_);
    }

    UncachedIpcBuffer(const UncachedIpcBuffer&) = delete;
    UncachedIpcBuffer& operator=(const UncachedIpcBuffer&) = delete;

    void exchange(py::object& gloo_group)
    {
        if (!peers_.empty()) return;
        HIP_CALL(hipSetDevice(rank_));
        hipIpcMemHandle_t h;
        HIP_CALL(hipIpcGetMemHandle(&h, base_));
        const int n = (int)sizeof(h);
        auto opts = torch::TensorOptions().dtype(torch::kByte).device(at::kCPU);
        at::Tensor mine = at::empty(n, opts);
        std::memcpy(mine.data_ptr(), &h, n);
        std::vector<at::Tensor> all(nGPU_);
        py::list lst;
        for (int i = 0; i < nGPU_; ++i) {
            all[i] = at::empty_like(mine);
            lst.append(all[i]);
        }
        py::module::import("torch.distributed").attr("all_gather")(
            lst, mine, py::arg("group") = gloo_group, py::arg("async_op") = false);
        peers_.assign(nGPU_, nullptr);
        for (int r = 0; r < nGPU_; ++r) {
            if (r == rank_) { peers_[r] = base_; continue; }
            hipIpcMemHandle_t ph;
            std::memcpy(&ph, all[r].data_ptr(), n);
            HIP_CALL(hipIpcOpenMemHandle(&peers_[r], ph, hipIpcMemLazyEnablePeerAccess));
        }
    }

    at::Tensor get_peer_bases_cpu() const
    {
        if (peers_.empty()) throw std::runtime_error("call exchange() first");
        auto t = at::empty(nGPU_, torch::TensorOptions().dtype(torch::kInt64));
        auto a = t.accessor<int64_t, 1>();
        for (int r = 0; r < nGPU_; ++r) a[r] = (int64_t)(uintptr_t)peers_[r];
        return t;
    }

    at::Tensor local_view() const
    {
        auto opts = torch::TensorOptions().dtype(torch::kByte).device(torch::kCUDA, rank_);
        return torch::from_blob(base_, {(int64_t)bytes_}, opts);
    }

private:
    int rank_, nGPU_;
    size_t bytes_;
    void* base_ = nullptr;
    std::vector<void*> peers_;
};

PYBIND11_MODULE(_lamport_ipc, m)
{
    py::class_<UncachedIpcBuffer>(m, "UncachedIpcBuffer", py::module_local())
        .def(py::init<int, int, size_t, int>(), py::arg("rank"), py::arg("nGPU"),
             py::arg("bytes"), py::arg("fill"))
        .def("exchange", &UncachedIpcBuffer::exchange, py::arg("gloo_group"))
        .def("get_peer_bases_cpu", &UncachedIpcBuffer::get_peer_bases_cpu)
        .def("local_view", &UncachedIpcBuffer::local_view, py::keep_alive<0, 1>());
}
