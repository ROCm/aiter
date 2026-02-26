
# 📈Using the ROCm profiler to extract kernel execution time.

[← Back to GPU profiling](../gpu_profiling.md)

---

The ROC profiler can be used to extract kernel runtime information as follows.

```bash
rocprof --stats <HIP application to profile>
```
This generates a csv file that containing some basic kernel statistics. E.g...

- Kernel Calls
- Total duration in nanoseconds
- The average duration if called more than one
- The percentage of the overall time spend in a kernel


## 📚 References

[ROC Profiler Documentation](https://rocm.docs.amd.com/projects/rocprofiler/en/docs-5.1.3/rocprof.html)

---

[← Back to GPU profiling](../gpu_profiling.md)