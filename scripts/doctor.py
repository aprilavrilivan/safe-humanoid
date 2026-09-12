#!/usr/bin/env python3
"""Diagnose the host and pinned Isaac Sim / Isaac Lab software stack.

This program is deliberately read-only and uses only the Python standard library until the
optional full-stack probes run. It can therefore explain an unsupported Mac development host,
inspect a fresh Linux machine before installation, and validate the completed GPU environment.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "stack" / "isaaclab-v2.3.2.yaml"
VALID_STATUSES = {"PASS", "WARN", "FAIL", "SKIP"}


@dataclass(frozen=True)
class CheckResult:
    """One independently reportable diagnostic result."""

    category: str
    name: str
    status: str
    actual: str
    expected: str = ""
    detail: str = ""


def parse_flat_yaml(path: Path) -> dict[str, Any]:
    """Parse the scalar-only stack manifest without requiring PyYAML.

    The manifest intentionally contains no nested mappings or lists. Quoted strings, integers,
    floats, booleans, and unquoted strings are supported.
    """

    if not path.is_file():
        raise ValueError(f"Stack manifest not found: {path}")

    config: dict[str, Any] = {}
    for line_number, original_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        stripped = original_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if original_line[:1].isspace():
            raise ValueError(
                f"{path}:{line_number}: nested YAML is not supported by the bootstrap manifest"
            )
        if ":" not in original_line:
            raise ValueError(f"{path}:{line_number}: expected 'key: value'")

        key, raw_value = original_line.split(":", 1)
        key = key.strip()
        raw_value = re.sub(r"\s+#.*$", "", raw_value).strip()
        if not key or not raw_value:
            raise ValueError(f"{path}:{line_number}: key and scalar value are required")
        if key in config:
            raise ValueError(f"{path}:{line_number}: duplicate key '{key}'")

        if len(raw_value) >= 2 and raw_value[0] == raw_value[-1] and raw_value[0] in {'"', "'"}:
            value: Any = raw_value[1:-1]
        elif raw_value.lower() in {"true", "false"}:
            value = raw_value.lower() == "true"
        elif re.fullmatch(r"[-+]?\d+", raw_value):
            value = int(raw_value)
        elif re.fullmatch(r"[-+]?(?:\d+\.\d*|\d*\.\d+)", raw_value):
            value = float(raw_value)
        else:
            value = raw_value
        config[key] = value

    required = {
        "schema_version",
        "supported_linux_distribution",
        "supported_linux_version",
        "supported_architecture",
        "minimum_glibc",
        "minimum_ram_gb",
        "minimum_vram_gb",
        "recommended_free_disk_gb",
        "recommended_nvidia_driver",
        "virtualenv_directory",
        "isaac_lab_directory",
        "python_version",
        "isaac_sim_version",
        "torch_version",
        "torchvision_version",
        "cuda_runtime",
        "cuda_wheel_tag",
        "nvidia_python_index",
        "torch_python_index",
        "isaac_lab_repository",
        "isaac_lab_ref",
        "isaac_lab_commit",
        "rl_library",
        "official_task_id",
    }
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(f"Missing required stack keys: {', '.join(missing)}")
    if config["schema_version"] != 1:
        raise ValueError(f"Unsupported stack-manifest schema: {config['schema_version']}")
    return config


def version_tuple(value: str) -> tuple[int, ...]:
    """Return the numeric components used for minimum-version comparisons."""

    return tuple(int(part) for part in re.findall(r"\d+", value))


def version_matches(actual: str, expected: str) -> bool:
    """Accept common package suffixes such as 5.1.0.0 and 2.7.0+cu128."""

    return actual == expected or actual.startswith(f"{expected}.") or actual.startswith(
        f"{expected}+"
    )


def run_command(command: list[str], timeout: int = 20) -> subprocess.CompletedProcess[str]:
    """Run a read-only probe without raising on a normal diagnostic failure."""

    try:
        return subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return subprocess.CompletedProcess(command, 127, "", str(exc))


def read_os_release() -> dict[str, str]:
    """Read Linux distribution metadata when available."""

    path = Path("/etc/os-release")
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line or line.startswith("#"):
            continue
        key, value = line.split("=", 1)
        values[key] = value.strip().strip('"')
    return values


def total_memory_gb() -> float | None:
    """Return physical memory using portable sysconf values where possible."""

    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, OSError, ValueError):
        return None
    return pages * page_size / 1024**3


def resolve_project_path(configured: str) -> Path:
    """Resolve a manifest path while keeping it inside the repository."""

    path = Path(configured)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Configured project path must remain inside the repository: {configured}")
    resolved = (REPO_ROOT / path).resolve()
    try:
        resolved.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise ValueError(f"Configured project path escapes the repository: {configured}") from exc
    return resolved


def package_version(*distribution_names: str) -> str | None:
    """Return the first installed distribution version from a list of aliases."""

    for name in distribution_names:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


class Doctor:
    """Collect and render checks for one stack manifest."""

    def __init__(self, config: dict[str, Any], mode: str, probe_task: bool) -> None:
        self.config = config
        self.mode = mode
        self.probe_task = probe_task
        self.results: list[CheckResult] = []

    def add(
        self,
        category: str,
        name: str,
        status: str,
        actual: Any,
        expected: Any = "",
        detail: str = "",
    ) -> None:
        if status not in VALID_STATUSES:
            raise ValueError(f"Invalid diagnostic status: {status}")
        self.results.append(
            CheckResult(category, name, status, str(actual), str(expected), detail)
        )

    def check_host(self) -> None:
        system = platform.system()
        expected_arch = str(self.config["supported_architecture"])
        actual_arch = platform.machine()
        self.add(
            "host",
            "operating system",
            "PASS" if system == "Linux" else "FAIL",
            system,
            "Linux",
        )
        self.add(
            "host",
            "architecture",
            "PASS" if actual_arch == expected_arch else "FAIL",
            actual_arch,
            expected_arch,
        )

        if system == "Linux":
            release = read_os_release()
            actual_distro = release.get("ID", "unknown").lower()
            actual_version = release.get("VERSION_ID", "unknown")
            expected_distro = str(self.config["supported_linux_distribution"]).lower()
            expected_version = str(self.config["supported_linux_version"])
            self.add(
                "host",
                "Linux distribution",
                "PASS" if actual_distro == expected_distro else "FAIL",
                actual_distro,
                expected_distro,
            )
            self.add(
                "host",
                "Linux version",
                "PASS" if actual_version == expected_version else "FAIL",
                actual_version,
                expected_version,
            )

            libc_name, libc_version = platform.libc_ver()
            minimum_glibc = str(self.config["minimum_glibc"])
            glibc_ok = libc_name.lower() == "glibc" and version_tuple(
                libc_version
            ) >= version_tuple(minimum_glibc)
            self.add(
                "host",
                "GLIBC",
                "PASS" if glibc_ok else "FAIL",
                f"{libc_name} {libc_version}".strip() or "unknown",
                f">= {minimum_glibc}",
            )
        else:
            self.add(
                "host",
                "Linux distribution",
                "SKIP",
                "not Linux",
                (
                    f"{self.config['supported_linux_distribution']} "
                    f"{self.config['supported_linux_version']}"
                ),
            )
            self.add(
                "host",
                "GLIBC",
                "SKIP",
                "not Linux",
                f">= {self.config['minimum_glibc']}",
            )

        memory_gb = total_memory_gb()
        minimum_ram = float(self.config["minimum_ram_gb"])
        if memory_gb is None:
            self.add("host", "physical RAM", "WARN", "unknown", f">= {minimum_ram:g} GB")
        else:
            self.add(
                "host",
                "physical RAM",
                "PASS" if memory_gb >= minimum_ram else "FAIL",
                f"{memory_gb:.1f} GB",
                f">= {minimum_ram:g} GB",
            )

        free_disk_gb = shutil.disk_usage(REPO_ROOT).free / 1024**3
        recommended_disk = float(self.config["recommended_free_disk_gb"])
        self.add(
            "host",
            "free project-volume disk",
            "PASS" if free_disk_gb >= recommended_disk else "WARN",
            f"{free_disk_gb:.1f} GB",
            f"recommended >= {recommended_disk:g} GB",
        )

    def check_tools(self) -> None:
        tools = {
            "git": "source checkout",
            "uv": "Python environment and package installation",
            "cmake": "Isaac Lab native build dependencies",
            "gcc": "Isaac Lab native build dependencies",
            "g++": "Isaac Lab native build dependencies",
            "make": "Isaac Lab native build dependencies",
        }
        for executable, purpose in tools.items():
            location = shutil.which(executable)
            self.add(
                "tools",
                executable,
                "PASS" if location else "FAIL",
                location or "not found",
                "available on PATH",
                purpose,
            )

    def check_gpu(self) -> None:
        nvidia_smi = shutil.which("nvidia-smi")
        if not nvidia_smi:
            status = "FAIL" if platform.system() == "Linux" else "SKIP"
            self.add("gpu", "nvidia-smi", status, "not found", "available on Linux GPU host")
            return

        query = run_command(
            [
                nvidia_smi,
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ]
        )
        if query.returncode != 0:
            self.add(
                "gpu",
                "nvidia-smi",
                "FAIL",
                "query failed",
                "working NVIDIA driver",
                query.stderr.strip(),
            )
            return

        gpu_rows: list[tuple[str, float, str]] = []
        for line in query.stdout.splitlines():
            fields = [field.strip() for field in line.split(",")]
            if len(fields) != 3:
                continue
            try:
                gpu_rows.append((fields[0], float(fields[1]) / 1024, fields[2]))
            except ValueError:
                continue
        if not gpu_rows:
            self.add("gpu", "NVIDIA GPUs", "FAIL", "none parsed", ">= 1 GPU")
            return

        minimum_vram = float(self.config["minimum_vram_gb"])
        best_vram = max(row[1] for row in gpu_rows)
        descriptions = "; ".join(f"{name} ({vram:.1f} GB)" for name, vram, _ in gpu_rows)
        self.add("gpu", "NVIDIA GPUs", "PASS", descriptions, ">= 1 GPU")
        self.add(
            "gpu",
            "maximum GPU VRAM",
            "PASS" if best_vram >= minimum_vram else "FAIL",
            f"{best_vram:.1f} GB",
            f">= {minimum_vram:g} GB",
        )

        driver = gpu_rows[0][2]
        recommended_driver = str(self.config["recommended_nvidia_driver"])
        self.add(
            "gpu",
            "NVIDIA driver",
            "PASS" if version_tuple(driver) >= version_tuple(recommended_driver) else "WARN",
            driver,
            f"recommended >= {recommended_driver}",
        )

    def check_python_environment(self) -> None:
        expected_python = str(self.config["python_version"])
        actual_python = f"{sys.version_info.major}.{sys.version_info.minor}"
        self.add(
            "python",
            "Python version",
            "PASS" if actual_python == expected_python else "FAIL",
            actual_python,
            expected_python,
            sys.executable,
        )

        expected_venv = resolve_project_path(str(self.config["virtualenv_directory"]))
        in_virtualenv = sys.prefix != sys.base_prefix
        if not in_virtualenv:
            self.add("python", "virtual environment", "FAIL", "not active", str(expected_venv))
        else:
            actual_prefix = Path(sys.prefix).resolve()
            self.add(
                "python",
                "virtual environment",
                "PASS" if actual_prefix == expected_venv else "WARN",
                actual_prefix,
                expected_venv,
            )

        expected_packages = {
            "Isaac Sim": (("isaacsim",), str(self.config["isaac_sim_version"])),
            "PyTorch": (("torch",), str(self.config["torch_version"])),
            "torchvision": (("torchvision",), str(self.config["torchvision_version"])),
        }
        for display_name, (aliases, expected) in expected_packages.items():
            actual = package_version(*aliases)
            self.add(
                "packages",
                display_name,
                "PASS" if actual and version_matches(actual, expected) else "FAIL",
                actual or "not installed",
                expected,
            )

        torch_probe = run_command(
            [
                sys.executable,
                "-c",
                (
                    "import json, torch; "
                    "print(json.dumps({'available': torch.cuda.is_available(), "
                    "'cuda': torch.version.cuda, 'count': torch.cuda.device_count(), "
                    "'devices': [torch.cuda.get_device_name(i) "
                    "for i in range(torch.cuda.device_count())]}))"
                ),
            ]
        )
        if torch_probe.returncode != 0:
            self.add(
                "python",
                "PyTorch CUDA probe",
                "FAIL",
                "failed",
                "CUDA available",
                torch_probe.stderr.strip(),
            )
        else:
            try:
                probe = json.loads(torch_probe.stdout.strip().splitlines()[-1])
            except (IndexError, json.JSONDecodeError) as exc:
                self.add(
                    "python",
                    "PyTorch CUDA probe",
                    "FAIL",
                    "invalid output",
                    "JSON",
                    str(exc),
                )
            else:
                expected_cuda = str(self.config["cuda_runtime"])
                cuda_version = str(probe.get("cuda"))
                cuda_ok = bool(probe.get("available")) and cuda_version.startswith(expected_cuda)
                self.add(
                    "python",
                    "PyTorch CUDA",
                    "PASS" if cuda_ok else "FAIL",
                    (
                        f"available={probe.get('available')}, runtime={cuda_version}, "
                        f"devices={probe.get('devices')}"
                    ),
                    f"available=True, runtime={expected_cuda}",
                )

    def check_isaac_lab(self) -> None:
        checkout = resolve_project_path(str(self.config["isaac_lab_directory"]))
        git_dir = checkout / ".git"
        if not git_dir.exists():
            self.add("Isaac Lab", "source checkout", "FAIL", "not found", str(checkout))
        else:
            revision = run_command(["git", "-C", str(checkout), "rev-parse", "HEAD"])
            actual_commit = revision.stdout.strip() if revision.returncode == 0 else "unknown"
            expected_commit = str(self.config["isaac_lab_commit"])
            self.add(
                "Isaac Lab",
                "source revision",
                "PASS" if actual_commit == expected_commit else "FAIL",
                actual_commit,
                expected_commit,
                str(self.config["isaac_lab_ref"]),
            )

        for module in ("isaaclab", "isaaclab_rl", "rsl_rl"):
            probe = run_command(
                [
                    sys.executable,
                    "-c",
                    f"import importlib; importlib.import_module({module!r}); print('ok')",
                ],
                timeout=60,
            )
            detail = probe.stderr.strip().splitlines()[-1] if probe.stderr.strip() else ""
            self.add(
                "Isaac Lab",
                f"import {module}",
                "PASS" if probe.returncode == 0 else "FAIL",
                "ok" if probe.returncode == 0 else "failed",
                "import succeeds",
                detail,
            )

        task_id = str(self.config["official_task_id"])
        if not self.probe_task:
            self.add(
                "task",
                task_id,
                "SKIP",
                "simulator probe not requested",
                "registered in Gymnasium",
                "rerun full mode with --probe-task after accepting the NVIDIA EULA",
            )
            return

        task_probe_code = "\n".join(
            (
                "from isaaclab.app import AppLauncher",
                "launcher = AppLauncher(headless=True)",
                "simulation_app = launcher.app",
                "import gymnasium as gym",
                "import isaaclab_tasks",
                f"print('registered' if {task_id!r} in gym.registry else 'missing')",
                "simulation_app.close()",
            )
        )
        task_probe = run_command([sys.executable, "-c", task_probe_code], timeout=180)
        task_output = task_probe.stdout.strip().splitlines()
        registered = task_probe.returncode == 0 and any(
            line.strip() == "registered" for line in task_output
        )
        detail = task_probe.stderr.strip().splitlines()[-1] if task_probe.stderr.strip() else ""
        self.add(
            "task",
            task_id,
            "PASS" if registered else "FAIL",
            "registered" if registered else "missing or import failed",
            "registered in Gymnasium",
            detail,
        )

    def run(self) -> list[CheckResult]:
        self.check_host()
        self.check_tools()
        self.check_gpu()
        if self.mode == "full":
            self.check_python_environment()
            self.check_isaac_lab()
        return self.results


def print_results(results: list[CheckResult]) -> None:
    """Render a concise terminal report grouped by category."""

    colors = {
        "PASS": "\033[32m",
        "WARN": "\033[33m",
        "FAIL": "\033[31m",
        "SKIP": "\033[36m",
    }
    use_color = sys.stdout.isatty() and "NO_COLOR" not in os.environ
    reset = "\033[0m" if use_color else ""
    last_category = ""
    for result in results:
        if result.category != last_category:
            if last_category:
                print()
            print(f"[{result.category}]")
            last_category = result.category
        status = f"{colors[result.status] if use_color else ''}{result.status:4}{reset}"
        expectation = f" (expected {result.expected})" if result.expected else ""
        detail = f" - {result.detail}" if result.detail else ""
        print(f"  {status}  {result.name}: {result.actual}{expectation}{detail}")

    counts = {
        status: sum(item.status == status for item in results) for status in VALID_STATUSES
    }
    print(
        "\nSummary: "
        f"{counts['PASS']} passed, {counts['WARN']} warnings, "
        f"{counts['FAIL']} failed, {counts['SKIP']} skipped"
    )


def write_json_report(
    destination: Path,
    config_path: Path,
    mode: str,
    results: list[CheckResult],
) -> None:
    """Write a machine-readable provenance and diagnostics report."""

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repository": str(REPO_ROOT),
        "config": str(config_path),
        "mode": mode,
        "python_executable": sys.executable,
        "results": [asdict(item) for item in results],
        "summary": {
            status.lower(): sum(item.status == status for item in results)
            for status in sorted(VALID_STATUSES)
        },
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="flat stack manifest")
    parser.add_argument(
        "--mode",
        choices=("auto", "preflight", "full"),
        default="auto",
        help=(
            "preflight checks the host; full also validates installed packages; "
            "add --probe-task to launch the task-registration probe"
        ),
    )
    parser.add_argument("--json", type=Path, help="optional path for a machine-readable report")
    parser.add_argument(
        "--probe-task",
        action="store_true",
        help=(
            "start Isaac Sim headlessly and verify the official Gym task registration "
            "(full mode only)"
        ),
    )
    parser.add_argument(
        "--no-fail",
        action="store_true",
        help="always exit zero after printing results (useful for inventory collection)",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config_path = args.config.expanduser().resolve()
    try:
        config = parse_flat_yaml(config_path)
        mode = args.mode
        if mode == "auto":
            isaac_sim_installed = package_version("isaacsim") is not None
            mode = "full" if sys.prefix != sys.base_prefix and isaac_sim_installed else "preflight"
        if args.probe_task and mode != "full":
            raise ValueError(
                "--probe-task requires --mode full or an installed stack detected by auto mode"
            )
        doctor = Doctor(config, mode, args.probe_task)
        results = doctor.run()
    except (OSError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    print(f"safe-humanoid doctor | mode={mode} | config={config_path}")
    print_results(results)
    if args.json:
        write_json_report(args.json.expanduser().resolve(), config_path, mode, results)
        print(f"Report written to {args.json.expanduser().resolve()}")

    failed = any(result.status == "FAIL" for result in results)
    return 0 if args.no_fail or not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
