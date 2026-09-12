#!/usr/bin/env bash
set -euo pipefail

# Install the pinned Isaac Sim / Isaac Lab stack into project-local directories.
#
# This script intentionally does not install an NVIDIA driver, invoke sudo, remove an existing
# environment, modify shell startup files, or accept the NVIDIA EULA. Those actions require an
# explicit human decision. Run scripts/doctor.py first when diagnosing an existing host.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." >/dev/null 2>&1 && pwd)"
DEFAULT_CONFIG="${REPO_ROOT}/configs/stack/isaaclab-v2.3.2.yaml"

CONFIG_PATH="${DEFAULT_CONFIG}"
DRY_RUN=0

usage() {
    cat <<'EOF'
Usage: scripts/setup/bootstrap_linux.sh [--config PATH] [--dry-run]

Creates a project-local Python environment and Isaac Lab checkout using the versions declared
in configs/stack/isaaclab-v2.3.2.yaml.

Options:
  --config PATH  Read a different flat stack manifest.
  --dry-run      Validate and print the resolved plan without changing the machine.
  -h, --help     Show this help message.
EOF
}

die() {
    echo "[ERROR] $*" >&2
    exit 1
}

info() {
    echo "[INFO] $*"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)
            [[ $# -ge 2 ]] || die "--config requires a path."
            CONFIG_PATH="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "Unknown argument: $1"
            ;;
    esac
done

[[ -f "${CONFIG_PATH}" ]] || die "Stack manifest not found: ${CONFIG_PATH}"

# Read one scalar from the deliberately flat YAML manifest without requiring PyYAML.
read_config() {
    local key="$1"
    awk -v wanted="${key}" '
        /^[[:space:]]*#/ || /^[[:space:]]*$/ { next }
        {
            separator = index($0, ":")
            if (separator == 0) { next }
            candidate = substr($0, 1, separator - 1)
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", candidate)
            if (candidate != wanted) { next }
            value = substr($0, separator + 1)
            sub(/[[:space:]]+#.*$/, "", value)
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", value)
            if ((substr(value, 1, 1) == "\"" && substr(value, length(value), 1) == "\"") ||
                (substr(value, 1, 1) == "\047" && substr(value, length(value), 1) == "\047")) {
                value = substr(value, 2, length(value) - 2)
            }
            print value
            exit
        }
    ' "${CONFIG_PATH}"
}

require_config() {
    local key="$1"
    local value
    value="$(read_config "${key}")"
    [[ -n "${value}" ]] || die "Missing required key '${key}' in ${CONFIG_PATH}"
    printf '%s' "${value}"
}

resolve_project_path() {
    local configured="$1"
    [[ "${configured}" != /* ]] || \
        die "Configured paths must be relative to the repository: ${configured}"
    [[ "${configured}" != *".."* ]] || \
        die "Configured paths may not traverse parent directories: ${configured}"
    [[ "${configured}" != "~"* ]] || \
        die "Configured paths may not use a home-directory shortcut: ${configured}"
    printf '%s/%s' "${REPO_ROOT}" "${configured#./}"
}

require_command() {
    local command_name="$1"
    local help_text="$2"
    command -v "${command_name}" >/dev/null 2>&1 || die "Missing '${command_name}'. ${help_text}"
}

SCHEMA_VERSION="$(require_config schema_version)"
SUPPORTED_DISTRO="$(require_config supported_linux_distribution)"
SUPPORTED_DISTRO_VERSION="$(require_config supported_linux_version)"
SUPPORTED_ARCH="$(require_config supported_architecture)"
PYTHON_VERSION="$(require_config python_version)"
ISAAC_SIM_VERSION="$(require_config isaac_sim_version)"
TORCH_VERSION="$(require_config torch_version)"
TORCHVISION_VERSION="$(require_config torchvision_version)"
NVIDIA_INDEX="$(require_config nvidia_python_index)"
TORCH_INDEX="$(require_config torch_python_index)"
ISAACLAB_REPOSITORY="$(require_config isaac_lab_repository)"
ISAACLAB_REF="$(require_config isaac_lab_ref)"
ISAACLAB_COMMIT="$(require_config isaac_lab_commit)"
RL_LIBRARY="$(require_config rl_library)"
OFFICIAL_TASK_ID="$(require_config official_task_id)"
VENV_DIR="$(resolve_project_path "$(require_config virtualenv_directory)")"
ISAACLAB_DIR="$(resolve_project_path "$(require_config isaac_lab_directory)")"
VENV_PYTHON="${VENV_DIR}/bin/python"

[[ "${SCHEMA_VERSION}" == "1" ]] || die "Unsupported stack-manifest schema: ${SCHEMA_VERSION}"

if [[ "${DRY_RUN}" -eq 1 ]]; then
    cat <<EOF
[DRY-RUN] Repository:       ${REPO_ROOT}
[DRY-RUN] Configuration:    ${CONFIG_PATH}
[DRY-RUN] Virtual env:      ${VENV_DIR}
[DRY-RUN] Python:           ${PYTHON_VERSION}
[DRY-RUN] Isaac Sim:        ${ISAAC_SIM_VERSION}
[DRY-RUN] PyTorch:          ${TORCH_VERSION} / torchvision ${TORCHVISION_VERSION}
[DRY-RUN] Isaac Lab:        ${ISAACLAB_REF} (${ISAACLAB_COMMIT})
[DRY-RUN] Isaac Lab path:   ${ISAACLAB_DIR}
[DRY-RUN] RL library:       ${RL_LIBRARY}
[DRY-RUN] Smoke-test task:  ${OFFICIAL_TASK_ID}
[DRY-RUN] No files, packages, repositories, or system settings were changed.
EOF
    exit 0
fi

[[ "$(uname -s)" == "Linux" ]] || \
    die "This bootstrap supports Linux only. Run doctor.py for a diagnostic report."
[[ "$(uname -m)" == "${SUPPORTED_ARCH}" ]] || \
    die "Expected architecture ${SUPPORTED_ARCH}; found $(uname -m)."

if [[ -r /etc/os-release ]]; then
    # shellcheck disable=SC1091
    source /etc/os-release
    ACTUAL_DISTRO="${ID:-unknown}"
    ACTUAL_DISTRO_VERSION="${VERSION_ID:-unknown}"
    [[ "${ACTUAL_DISTRO,,}" == "${SUPPORTED_DISTRO,,}" ]] || \
        die "Expected ${SUPPORTED_DISTRO}; found ${ACTUAL_DISTRO}."
    [[ "${ACTUAL_DISTRO_VERSION}" == "${SUPPORTED_DISTRO_VERSION}" ]] || \
        die "Expected ${SUPPORTED_DISTRO} ${SUPPORTED_DISTRO_VERSION};" \
            "found ${ACTUAL_DISTRO_VERSION}."
else
    die "Cannot read /etc/os-release to verify the Linux distribution."
fi

require_command python3 "Install the distribution's Python 3 package so preflight checks can run."
require_command git "Install Git before bootstrapping the project."
require_command uv "Install uv from https://docs.astral.sh/uv/getting-started/installation/."
require_command nvidia-smi "Install and validate the NVIDIA production driver first."

# Isaac Lab's installer may invoke apt/sudo when cmake is absent. Requiring build tools here
# keeps privileged system changes outside this script.
for BUILD_TOOL in cmake gcc g++ make; do
    require_command \
        "${BUILD_TOOL}" \
        "Install build-essential and cmake explicitly before continuing."
done

info "Running host preflight checks."
python3 "${REPO_ROOT}/scripts/doctor.py" --config "${CONFIG_PATH}" --mode preflight

if [[ -e "${VENV_DIR}" && ! -x "${VENV_PYTHON}" ]]; then
    die "${VENV_DIR} exists but is not a usable virtual environment. Inspect or move it manually."
fi

if [[ ! -x "${VENV_PYTHON}" ]]; then
    info "Creating Python ${PYTHON_VERSION} environment at ${VENV_DIR}."
    uv venv --python "${PYTHON_VERSION}" --seed "${VENV_DIR}"
else
    info "Reusing existing virtual environment at ${VENV_DIR}."
fi

ACTUAL_PYTHON_VERSION="$(
    "${VENV_PYTHON}" -c \
        'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")'
)"
[[ "${ACTUAL_PYTHON_VERSION}" == "${PYTHON_VERSION}" ]] || \
    die "Existing environment uses Python ${ACTUAL_PYTHON_VERSION};" \
        "expected ${PYTHON_VERSION}. Move it manually and rerun."

info "Installing Isaac Sim ${ISAAC_SIM_VERSION}. The first simulator launch will" \
    "ask you to accept NVIDIA's EULA."
uv pip install --python "${VENV_PYTHON}" \
    "isaacsim[all,extscache]==${ISAAC_SIM_VERSION}" \
    --extra-index-url "${NVIDIA_INDEX}"

info "Installing CUDA PyTorch ${TORCH_VERSION} from ${TORCH_INDEX}."
uv pip install --python "${VENV_PYTHON}" --upgrade \
    "torch==${TORCH_VERSION}" \
    "torchvision==${TORCHVISION_VERSION}" \
    --index-url "${TORCH_INDEX}"

ISAACLAB_PARENT="$(dirname -- "${ISAACLAB_DIR}")"
mkdir -p "${ISAACLAB_PARENT}"

if [[ ! -e "${ISAACLAB_DIR}" ]]; then
    info "Cloning Isaac Lab ${ISAACLAB_REF} into ${ISAACLAB_DIR}."
    git clone --depth 1 --branch "${ISAACLAB_REF}" "${ISAACLAB_REPOSITORY}" "${ISAACLAB_DIR}"
elif [[ ! -d "${ISAACLAB_DIR}/.git" ]]; then
    die "${ISAACLAB_DIR} exists but is not a Git checkout. Inspect or move it manually."
else
    info "Reusing existing Isaac Lab checkout at ${ISAACLAB_DIR}."
    ACTUAL_ORIGIN="$(git -C "${ISAACLAB_DIR}" remote get-url origin)"
    [[ "${ACTUAL_ORIGIN}" == "${ISAACLAB_REPOSITORY}" ]] || \
        die "Isaac Lab origin is ${ACTUAL_ORIGIN}; expected ${ISAACLAB_REPOSITORY}."
fi

ACTUAL_ISAACLAB_COMMIT="$(git -C "${ISAACLAB_DIR}" rev-parse HEAD)"
[[ "${ACTUAL_ISAACLAB_COMMIT}" == "${ISAACLAB_COMMIT}" ]] || \
    die "Isaac Lab is at ${ACTUAL_ISAACLAB_COMMIT}; expected ${ISAACLAB_COMMIT}." \
        "This script will not overwrite an existing checkout."

info "Installing Isaac Lab extensions and ${RL_LIBRARY}."
# Activating the environment makes the upstream helper use this project-local Python. Build
# tools were checked above, so its installation path will not need to invoke apt or sudo.
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
"${ISAACLAB_DIR}/isaaclab.sh" --install "${RL_LIBRARY}"

info "Running the full installed-stack diagnostic."
"${VENV_PYTHON}" "${REPO_ROOT}/scripts/doctor.py" \
    --config "${CONFIG_PATH}" \
    --mode full

cat <<EOF

[SUCCESS] The pinned Isaac Lab environment is installed.

Activate it with:
  source "${VENV_DIR}/bin/activate"

Isaac Lab checkout:
  ${ISAACLAB_DIR}

Next, launch the official task with Isaac Lab's zero/random-agent or RSL-RL scripts:
  ${OFFICIAL_TASK_ID}

The first Isaac Sim launch requires your explicit acceptance of NVIDIA's EULA.
EOF
