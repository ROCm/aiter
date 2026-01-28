from contextlib import redirect_stdout, redirect_stderr
import io
import logging
import shlex


def disable_aiter_logs() -> None:
    logging.getLogger("aiter").disabled = True


disable_aiter_logs()
from bench_mha import main as bench_mha_main  # noqa: E402


def run_bench_mha(args: str) -> tuple[str, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        bench_mha_main(shlex.split(args))
    return stdout.getvalue(), stderr.getvalue()


def main() -> None:
    args = "--dtype bf16 -mode fwd -b 1 -hq 128 -hk 128 -sq 4096 -sk 4096 -d 192 -dv 128 -causal true --layout bshd -metric time"
    stdout, stderr = run_bench_mha(args)
    if stderr:
        print(f"Errors / warnings:\n{stderr}")
    print(f"Output:\n{stdout}")


if __name__ == "__main__":
    main()
