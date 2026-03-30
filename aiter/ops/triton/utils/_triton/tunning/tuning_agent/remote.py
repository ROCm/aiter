"""SSH + Docker exec remote execution wrapper for the agentic Triton tuning pipeline."""

import shlex
import subprocess
import time
from typing import Dict, List, Optional

from .types import MachineInfo


class RemoteCommandError(Exception):
    """Raised when a remote command fails or times out."""

    def __init__(
        self,
        message: str,
        returncode: Optional[int] = None,
        stdout: str = "",
        stderr: str = "",
    ) -> None:
        super().__init__(message)
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class RemoteExecutor:
    """Executes commands on a remote machine via SSH and inside Docker containers."""

    def __init__(self, machine: MachineInfo) -> None:
        self.machine = machine
        self.container_id: Optional[str] = None

    # ------------------------------------------------------------------
    # SSH helpers
    # ------------------------------------------------------------------

    def _build_ssh_command(self, command: str) -> List[str]:
        """Build an SSH command list with standard options."""
        return [
            "ssh",
            "-i", self.machine.ssh_key,
            "-o", "StrictHostKeyChecking=no",
            "-o", "ConnectTimeout=10",
            "-o", "BatchMode=yes",
            f"{self.machine.user}@{self.machine.host}",
            command,
        ]

    def ssh_run(
        self,
        command: str,
        timeout: Optional[float] = None,
        check: bool = False,
        retries: int = 0,
        backoff: float = 2.0,
    ) -> subprocess.CompletedProcess:
        """Run *command* on the remote machine via SSH.

        Parameters
        ----------
        command:
            Shell command string to execute remotely.
        timeout:
            Seconds to wait before raising :exc:`RemoteCommandError`.
        check:
            If ``True``, raise :exc:`RemoteCommandError` on non-zero exit.
        retries:
            How many times to retry when SSH itself fails (exit code 255).
        backoff:
            Base for exponential back-off between retries (seconds * 2^attempt).
        """
        ssh_cmd = self._build_ssh_command(command)
        attempt = 0
        while True:
            try:
                result = subprocess.run(
                    ssh_cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
            except subprocess.TimeoutExpired:
                raise RemoteCommandError(
                    f"SSH command timed out after {timeout}s: {command}",
                    returncode=None,
                    stdout="",
                    stderr="",
                )

            if result.returncode == 255:
                if attempt < retries:
                    sleep_time = backoff * (2 ** attempt)
                    time.sleep(sleep_time)
                    attempt += 1
                    continue
                # Retries exhausted — SSH connection error
                raise RemoteCommandError(
                    f"SSH connection failed (exit 255) after {retries} retries: {command}",
                    returncode=result.returncode,
                    stdout=result.stdout,
                    stderr=result.stderr,
                )

            if check and result.returncode != 0:
                raise RemoteCommandError(
                    f"SSH command failed with exit code {result.returncode}: {command}",
                    returncode=result.returncode,
                    stdout=result.stdout,
                    stderr=result.stderr,
                )

            return result

    # ------------------------------------------------------------------
    # Docker helpers
    # ------------------------------------------------------------------

    def docker_exec(
        self,
        command: str,
        env: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
        check: bool = True,
        workdir: Optional[str] = None,
    ) -> subprocess.CompletedProcess:
        """Execute *command* inside the current container via ``docker exec`` over SSH.

        The command is run as ``bash -c '<quoted-command>'`` inside the container.

        Raises
        ------
        RemoteCommandError
            If :attr:`container_id` is not set, or if *check* is ``True`` and
            the command exits with a non-zero code.
        """
        if self.container_id is None:
            raise RemoteCommandError(
                "No container_id set — call create_container() first.",
                returncode=None,
            )

        docker_parts: List[str] = ["docker", "exec"]

        if env:
            for key, value in env.items():
                docker_parts += ["-e", f"{key}={shlex.quote(value)}"]

        if workdir:
            docker_parts += ["-w", workdir]

        docker_parts += [self.container_id, "bash", "-c", shlex.quote(command)]

        docker_cmd = " ".join(docker_parts)
        return self.ssh_run(docker_cmd, timeout=timeout, check=check)

    def create_container(
        self,
        image: str,
        run_script: Optional[str] = None,
        name: Optional[str] = None,
    ) -> str:
        """Create a Docker container on the remote machine.

        Parameters
        ----------
        image:
            Docker image to use.
        run_script:
            Optional path (on the remote host) to a shell script that creates
            the container and prints its ID as the last line of stdout.
        name:
            Optional ``--name`` argument for ``docker run``.

        Returns
        -------
        str
            The container ID (stripped of whitespace).
        """
        if run_script:
            result = self.ssh_run(run_script, check=True)
            container_id = result.stdout.strip().splitlines()[-1].strip()
        else:
            docker_parts: List[str] = ["docker", "run", "-d"]
            if name:
                docker_parts += ["--name", name]
            docker_parts += [
                "--device=/dev/kfd",
                "--device=/dev/dri",
                "--security-opt", "seccomp=unconfined",
                image,
                "sleep", "infinity",
            ]
            result = self.ssh_run(" ".join(docker_parts), check=True)
            container_id = result.stdout.strip()

        self.container_id = container_id
        return container_id

    def destroy_container(self) -> None:
        """Stop and remove the current container, then clear :attr:`container_id`."""
        cid = self.container_id
        self.ssh_run(f"docker stop {cid}", check=False)
        self.ssh_run(f"docker rm {cid}", check=False)
        self.container_id = None

    # ------------------------------------------------------------------
    # File transfer
    # ------------------------------------------------------------------

    def copy_from_container(self, remote_path: str, local_path: str) -> None:
        """Copy *remote_path* inside the container to *local_path* on localhost.

        Strategy:
        1. ``docker cp <container>:<remote_path> <tmp>`` on the remote host.
        2. ``scp`` the tmp file from the remote host to *local_path*.
        """
        tmp_path = f"/tmp/_remote_copy_{int(time.time())}"
        # Step 1: docker cp inside container to remote host tmp
        docker_cp_cmd = f"docker cp {self.container_id}:{remote_path} {tmp_path}"
        self.ssh_run(docker_cp_cmd, check=True)

        # Step 2: scp from remote host to local
        scp_src = f"{self.machine.user}@{self.machine.host}:{tmp_path}"
        scp_cmd = [
            "scp",
            "-i", self.machine.ssh_key,
            "-o", "StrictHostKeyChecking=no",
            "-o", "ConnectTimeout=10",
            scp_src,
            local_path,
        ]
        result = subprocess.run(scp_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RemoteCommandError(
                f"scp from remote failed",
                returncode=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
            )

    def copy_to_container(self, local_path: str, remote_path: str) -> None:
        """Copy *local_path* on localhost to *remote_path* inside the container.

        Strategy:
        1. ``scp`` from *local_path* to a tmp on the remote host.
        2. ``docker cp <tmp> <container>:<remote_path>`` on the remote host.
        """
        tmp_path = f"/tmp/_remote_copy_{int(time.time())}"

        # Step 1: scp from local to remote host
        scp_dst = f"{self.machine.user}@{self.machine.host}:{tmp_path}"
        scp_cmd = [
            "scp",
            "-i", self.machine.ssh_key,
            "-o", "StrictHostKeyChecking=no",
            "-o", "ConnectTimeout=10",
            local_path,
            scp_dst,
        ]
        result = subprocess.run(scp_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RemoteCommandError(
                f"scp to remote failed",
                returncode=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
            )

        # Step 2: docker cp from remote host tmp into container
        docker_cp_cmd = f"docker cp {tmp_path} {self.container_id}:{remote_path}"
        self.ssh_run(docker_cp_cmd, check=True)

    # ------------------------------------------------------------------
    # GPU process management
    # ------------------------------------------------------------------

    def kill_stale_gpu_processes(self) -> List[int]:
        """Kill all processes currently using the GPU via ``rocm-smi``.

        Returns
        -------
        List[int]
            The PIDs that were killed (may be empty).
        """
        result = self.ssh_run("rocm-smi --showpids", check=False)
        pids: List[int] = []
        for line in result.stdout.splitlines():
            parts = line.split()
            if parts and parts[0].isdigit():
                pids.append(int(parts[0]))

        for pid in pids:
            self.ssh_run(f"kill -9 {pid}", check=False)

        return pids

    # ------------------------------------------------------------------
    # Environment verification
    # ------------------------------------------------------------------

    def verify_environment(
        self,
        expected_triton_commit: Optional[str] = None,
        expected_aiter_branch: Optional[str] = None,
    ) -> Dict[str, str]:
        """Verify Triton version and aiter branch inside the container.

        Parameters
        ----------
        expected_triton_commit:
            If given, the actual Triton version must match this string.
        expected_aiter_branch:
            If given, the ``git rev-parse --abbrev-ref HEAD`` output must match.

        Returns
        -------
        Dict[str, str]
            Keys ``triton_version`` and ``aiter_branch``.

        Raises
        ------
        RemoteCommandError
            On any mismatch between expected and actual values.
        """
        triton_result = self.docker_exec(
            'python -c "import triton; print(triton.__version__)"',
            check=True,
        )
        triton_version = triton_result.stdout.strip()

        branch_result = self.docker_exec(
            "git rev-parse --abbrev-ref HEAD",
            check=True,
        )
        aiter_branch = branch_result.stdout.strip()

        if expected_triton_commit is not None and triton_version != expected_triton_commit:
            raise RemoteCommandError(
                f"Triton version mismatch: expected {expected_triton_commit!r}, "
                f"got {triton_version!r}",
                returncode=None,
                stdout=triton_version,
                stderr="",
            )

        if expected_aiter_branch is not None and aiter_branch != expected_aiter_branch:
            raise RemoteCommandError(
                f"aiter branch mismatch: expected {expected_aiter_branch!r}, "
                f"got {aiter_branch!r}",
                returncode=None,
                stdout=aiter_branch,
                stderr="",
            )

        return {"triton_version": triton_version, "aiter_branch": aiter_branch}

    # ------------------------------------------------------------------
    # Container / connectivity probes
    # ------------------------------------------------------------------

    def is_container_running(self) -> bool:
        """Return ``True`` if the current container is in the running state."""
        if self.container_id is None:
            return False
        result = self.ssh_run(
            f"docker inspect --format='{{{{.State.Running}}}}' {self.container_id}",
            check=False,
        )
        if result.returncode != 0:
            return False
        return result.stdout.strip().lower() == "true"

    def check_ssh_connectivity(self) -> bool:
        """Return ``True`` if the remote host is reachable via SSH."""
        try:
            result = self.ssh_run("echo ok", timeout=15, check=False)
            return result.returncode == 0
        except RemoteCommandError:
            return False
