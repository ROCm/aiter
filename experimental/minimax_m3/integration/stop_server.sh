#!/bin/bash
# Stop our vLLM server (inside ix_dsv41 only) and show GPU processes.
pkill -INT -f "vllm serve" ; sleep 15; pkill -9 -f "VLLM::|vllm serve|EngineCore|VllmWorker" 2>/dev/null; sleep 5
rocm-smi --showpids | grep -E "^[0-9]" || echo "no GPU processes"
