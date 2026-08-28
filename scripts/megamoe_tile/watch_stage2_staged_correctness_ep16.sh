#!/usr/bin/env bash
# One-shot EP16 watchdog. It only starts the staged correctness run after the
# local node is genuinely idle. Deploy/run this on both node ranks because an
# EP16 torchrun needs both hosts to join the same rendezvous.
set -uo pipefail

node_rank="${1:?node rank required}"
master_port="${2:-29820}"
tag="${3:-staged_auto_20260827}"
poll_seconds="${4:-60}"
max_vram_pct="${5:-5}"
heap_size="${6:-8G}"
master_addr="${7:-10.2.80.17}"
handshake_port="${8:-29819}"

log_dir=/home/hzm/logs/megamoe_route_store_validation_20260826
watch_log="${log_dir}/${tag}_watch_node${node_rank}.log"
mkdir -p "${log_dir}"

log() {
  printf '[%s] %s\n' "$(date '+%F %T %Z')" "$*" | tee -a "${watch_log}"
}

gpu_idle() {
  rocm-smi --showpids 2>/dev/null \
    | grep -q 'No KFD PIDs currently running' || return 1
  rocm-smi --showmemuse 2>/dev/null \
    | awk -F'VRAM%): ' \
      -v max_pct="${max_vram_pct}" \
      '/GPU Memory Allocated/ {
         value=$2; gsub(/[[:space:]]/, "", value)
         if ((value + 0) > (max_pct + 0)) bad=1
         count++
       }
       END { exit (count == 8 && !bad) ? 0 : 1 }' || return 1
  if pgrep -af 'torch.distributed.run|validate_megamoe_tile_ep16|stress_megamoe_tile_ep16_sparse_routes' \
      >/dev/null 2>&1; then
    return 1
  fi
  return 0
}

for status in \
  "${log_dir}/${tag}_pair_rank_paired-rank-half-remote_node${node_rank}.status" \
  "${log_dir}/${tag}_arb_rank_permuted-arbitrary-topk_node${node_rank}.status"; do
  if [[ -e "${status}" ]]; then
    log "refusing to reuse existing status ${status}; choose a new tag"
    exit 2
  fi
done

log "watching node=${node_rank}, max_vram=${max_vram_pct}%, heap=${heap_size}"
while ! gpu_idle; do
  log "node is still occupied; sleeping ${poll_seconds}s"
  sleep "${poll_seconds}"
done

if [[ "${node_rank}" == "0" ]]; then
  log "node idle; waiting for node1 handshake on ${master_addr}:${handshake_port}"
  python3 - "${handshake_port}" <<'PY'
import socket
import sys

port = int(sys.argv[1])
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("0.0.0.0", port))
    server.listen(1)
    while True:
        conn, _ = server.accept()
        with conn:
            if conn.recv(16) == b"ready":
                conn.sendall(b"ack")
                break
PY
  log "node1 handshake acknowledged"
else
  log "node idle; waiting for node0 handshake at ${master_addr}:${handshake_port}"
  peer_ready() {
    python3 - "${master_addr}" "${handshake_port}" <<'PY'
import socket
import sys

host = sys.argv[1]
port = int(sys.argv[2])
try:
    with socket.create_connection((host, port), timeout=3) as conn:
        conn.sendall(b"ready")
        if conn.recv(3) != b"ack":
            raise OSError("invalid watchdog handshake")
except OSError:
    raise SystemExit(1)
PY
  }
  while ! peer_ready; do
    log "node0 is not ready; sleeping ${poll_seconds}s"
    sleep "${poll_seconds}"
  done
  log "node0 handshake acknowledged"
fi

log "both nodes passed watchdog handshake; starting staged paired/arbitrary correctness"
export MORI_SHMEM_HEAP_SIZE="${heap_size}"
run() {
  bash scripts/megamoe_tile/run_stage2_route_store_validation_ep16.sh "$@"
}

run "${node_rank}" "${master_port}" direct paired-rank-half-remote \
  "${tag}_pair" 4 8 interleaved static_strided 0 expanded atomic &&
run "${node_rank}" "$((master_port + 1))" rank paired-rank-half-remote \
  "${tag}_pair" 4 8 load_first static_strided 0 expanded staged_reduce &&
run "${node_rank}" "$((master_port + 2))" direct permuted-arbitrary-topk \
  "${tag}_arb" 4 8 interleaved static_strided 0 expanded atomic &&
run "${node_rank}" "$((master_port + 3))" rank permuted-arbitrary-topk \
  "${tag}_arb" 4 8 load_first static_strided 0 expanded staged_reduce
rc=$?
log "watch run finished rc=${rc}"
exit "${rc}"
