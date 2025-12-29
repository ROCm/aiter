#include <hip/hip_runtime.h>
#include <torch/torch.h>
#include <c10/hip/HIPStream.h>
#include <c10/hip/HIPGuard.h>
#include <ATen/hip/HIPGraph.h>
#include <chrono>
#include <iostream>
#include <memory>

#ifdef BUILD_PYBIND_MODULE
#include <torch/extension.h>
#include <pybind11/pybind11.h>
namespace py = pybind11;
#endif

torch::Tensor pa_fwd(torch::Tensor& Q, torch::Tensor& K, torch::Tensor& V,
                     torch::Tensor& block_tables, torch::Tensor& context_lens,
                     int block_tables_stride0, int max_qlen,
                     std::optional<torch::Tensor> K_QScale, std::optional<torch::Tensor> V_QScale,
                     std::optional<torch::Tensor> out_, std::optional<torch::Tensor> qo_indptr,
                     std::optional<int> high_precision, std::string kernelName);

class PABenchmark {
public:
    int batch, num_heads, num_kv_heads, head_size, block_size, blocks_per_seq;
    int num_kernels, num_replays, warmup_iters;
    torch::Tensor Q, K, V, block_tables, context_lens, output;
    std::unique_ptr<at::cuda::CUDAGraph> graph;
    c10::hip::HIPStream stream = c10::hip::getDefaultHIPStream();
    bool captured = false;

    PABenchmark(int bs=32, int nh=32, int nkv=8, int hs=128, int blk=16, int bps=8, int nk=10, int nr=5, int wi=5)
        : batch(bs), num_heads(nh), num_kv_heads(nkv), head_size(hs),
          block_size(blk), blocks_per_seq(bps), num_kernels(nk), num_replays(nr), warmup_iters(wi) {
        auto fp16 = torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA, 0);
        auto i32 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA, 0);
        int nb = batch * blocks_per_seq, x = 8;
        Q = torch::randn({batch, num_heads, head_size}, fp16);
        K = torch::randn({nb, num_kv_heads, head_size/x, block_size, x}, fp16);
        V = torch::randn({nb, num_kv_heads, head_size, block_size}, fp16);
        block_tables = torch::arange(nb, i32).reshape({batch, blocks_per_seq});
        context_lens = torch::full({batch}, blocks_per_seq * block_size, i32);
        output = torch::empty_like(Q);
        hipDeviceSynchronize();
    }

    void run_kernel() {
        pa_fwd(Q, K, V, block_tables, context_lens, blocks_per_seq, 1,
               std::nullopt, std::nullopt, output, std::nullopt, 1, "");
    }

    void warmup() {
        for (int i = 0; i < warmup_iters; ++i)
            for (int k = 0; k < num_kernels; ++k) run_kernel();
        hipDeviceSynchronize();
    }

    void capture_graph() {
        stream = c10::hip::getStreamFromPool(false, 0);
        { c10::hip::HIPStreamGuard g(stream); for (int k = 0; k < num_kernels; ++k) run_kernel(); }
        stream.synchronize();
        graph = std::make_unique<at::cuda::CUDAGraph>();
        { c10::hip::HIPStreamGuard g(stream); graph->capture_begin();
          for (int k = 0; k < num_kernels; ++k) run_kernel(); graph->capture_end(); }
        captured = true;
    }

    void replay() { graph->replay(); hipDeviceSynchronize(); }

#ifdef BUILD_PYBIND_MODULE
    py::dict benchmark() {
#else
    void benchmark() {
#endif
        warmup();
        if (!captured) capture_graph();
        for (int i = 0; i < warmup_iters; ++i) replay();
        hipDeviceSynchronize();

        auto t1 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < num_replays; ++i) replay();
        hipDeviceSynchronize();
        auto t2 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < num_replays; ++i) { replay(); hipDeviceSynchronize(); }
        auto t3 = std::chrono::high_resolution_clock::now();

        float no_sync = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count() / (float)num_replays;
        float with_sync = std::chrono::duration_cast<std::chrono::microseconds>(t3-t2).count() / (float)num_replays;

#ifdef BUILD_PYBIND_MODULE
        py::dict r;
        r["num_kernels"] = num_kernels; r["num_replays"] = num_replays;
        r["no_sync_us"] = no_sync; r["with_sync_us"] = with_sync;
        r["per_kernel_us"] = with_sync / num_kernels;
        return r;
#else
        std::cout << "kernels=" << num_kernels << " replays=" << num_replays << std::endl;
        std::cout << "no_sync: " << no_sync << " us, with_sync: " << with_sync << " us" << std::endl;
        std::cout << "per_kernel: " << with_sync / num_kernels << " us" << std::endl;
#endif
    }

    bool is_captured() { return captured; }
};

#ifdef BUILD_PYBIND_MODULE
PYBIND11_MODULE(run_asm_pa, m) {
    py::class_<PABenchmark>(m, "PABenchmark")
        .def(py::init<int,int,int,int,int,int,int,int,int>(),
             py::arg("batch")=32, py::arg("num_heads")=32, py::arg("num_kv_heads")=8,
             py::arg("head_size")=128, py::arg("block_size")=16, py::arg("blocks_per_seq")=8,
             py::arg("num_kernels")=10, py::arg("num_replays")=5, py::arg("warmup")=5)
        .def("warmup", &PABenchmark::warmup)
        .def("capture_graph", &PABenchmark::capture_graph)
        .def("replay", &PABenchmark::replay)
        .def("benchmark", &PABenchmark::benchmark)
        .def("is_captured", &PABenchmark::is_captured);
}
#else
int main(int argc, char* argv[]) {
    int batch = 32, nk = 10, nr = 5;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--batch" && i+1 < argc) batch = std::stoi(argv[++i]);
        else if (a == "--num-kernels" && i+1 < argc) nk = std::stoi(argv[++i]);
        else if (a == "--num-replays" && i+1 < argc) nr = std::stoi(argv[++i]);
    }
    PABenchmark b(batch, 32, 8, 128, 16, 8, nk, nr, 5);
    b.benchmark();
    return 0;
}
#endif
