#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
LOCK_FILE="${LLAMA_CPP_LOCK_FILE:-${REPO_DIR}/catalog/runtime-locks/llama-cpp-cuda.yaml}"

if [[ -x "${REPO_DIR}/.venv/bin/python" ]]; then
  PYTHON_BIN="${REPO_DIR}/.venv/bin/python"
else
  PYTHON_BIN="python3"
fi

mapfile -t LOCK_VALUES < <(
  "${PYTHON_BIN}" - "${LOCK_FILE}" <<'PY'
import sys
from llm_lab.catalog import load_runtime_lock_file

lock = load_runtime_lock_file(sys.argv[1])
print(lock.source)
print(lock.commit)
print(lock.build["options"]["CMAKE_CUDA_ARCHITECTURES"])
print(lock.binary)
print(lock.binary_sha256)
print(lock.version_contains)
print("OFF" if not lock.build["options"].get("BUILD_SHARED_LIBS", False) else "ON")
PY
)
if [[ "${#LOCK_VALUES[@]}" -ne 7 ]]; then
  echo "Could not parse the reviewed runtime lock: ${LOCK_FILE}" >&2
  exit 2
fi

LOCK_SOURCE="${LOCK_VALUES[0]}"
LOCK_COMMIT="${LOCK_VALUES[1]}"
LOCK_CUDA_ARCH="${LOCK_VALUES[2]}"
LOCK_BINARY="${LOCK_VALUES[3]}"
LOCK_BINARY_SHA256="${LOCK_VALUES[4]}"
LOCK_VERSION_CONTAINS="${LOCK_VALUES[5]}"
LOCK_SHARED_LIBS="${LOCK_VALUES[6]}"
if [[ "${LOCK_SHARED_LIBS}" != "OFF" ]]; then
  echo "Production llama.cpp must be locked as a self-contained static build." >&2
  exit 2
fi

# The catalog lock is authoritative. An unlocked build is useful for
# experiments, but it cannot satisfy a locked production deployment.
ALLOW_UNLOCKED="${LLM_LAB_ALLOW_UNLOCKED_BUILD:-0}"
LLAMA_CPP_SOURCE="${LLAMA_CPP_SOURCE:-${LOCK_SOURCE}}"
LLAMA_CPP_REF="${LLAMA_CPP_REF:-${LOCK_COMMIT}}"
CUDA_ARCH="${LLAMA_CPP_CUDA_ARCH:-${LOCK_CUDA_ARCH}}"
if [[ "${ALLOW_UNLOCKED}" != "1" ]] && {
  [[ "${LLAMA_CPP_SOURCE}" != "${LOCK_SOURCE}" ]] ||
  [[ "${LLAMA_CPP_REF}" != "${LOCK_COMMIT}" ]] ||
  [[ "${CUDA_ARCH}" != "${LOCK_CUDA_ARCH}" ]];
}; then
  echo "Refusing overrides that disagree with ${LOCK_FILE}; set LLM_LAB_ALLOW_UNLOCKED_BUILD=1 for an explicitly unlocked experiment." >&2
  exit 2
fi

LLM_LAB_DATA_DIR="${LLM_LAB_DATA:-${REPO_DIR}/.llm-lab-data}"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-${LLM_LAB_DATA_DIR}/cache/llama.cpp}"
BUILD_JOBS="${LLAMA_CPP_BUILD_JOBS:-$(nproc)}"
BUILD_DIR="${LLAMA_CPP_BUILD_DIR:-${LLM_LAB_DATA_DIR}/work/verify/llama.cpp-candidate}"
CANDIDATE_BINARY="${BUILD_DIR}/bin/llama-server"
LOCKED_BINARY="${LLM_LAB_DATA_DIR}/${LOCK_BINARY}"

"${PYTHON_BIN}" - "${REPO_DIR}" "${LLM_LAB_DATA_DIR}" "${LLAMA_CPP_DIR}" "${BUILD_DIR}" <<'PY'
import os
from pathlib import Path
import sys

from llm_lab.paths import LabPaths, ensure_safe_parent_directory, open_safe_directory

repo, data, checkout, build = map(Path, sys.argv[1:])
paths = LabPaths.discover(repo, data)
paths.initialize()
checkout_lexical = Path(os.path.abspath(checkout.expanduser()))
build_lexical = Path(os.path.abspath(build.expanduser()))
try:
    checkout_lexical.relative_to(paths.data_root / "cache")
except ValueError as exc:
    raise SystemExit("llama.cpp checkout must remain below the managed cache") from exc
try:
    build_lexical.relative_to(paths.data_root / "work/verify")
except ValueError as exc:
    raise SystemExit("build candidate must remain below managed work/verify") from exc
for value, purpose in ((checkout, "llama.cpp checkout"), (build, "build candidate")):
    lexical = Path(os.path.abspath(value.expanduser()))
    try:
        lexical.relative_to(paths.data_root)
    except ValueError as exc:
        raise SystemExit(f"{purpose} must remain below {paths.data_root}") from exc
    if os.path.lexists(lexical):
        _, descriptor = open_safe_directory(lexical, purpose=purpose)
        os.close(descriptor)
    else:
        ensure_safe_parent_directory(lexical, purpose=purpose)
PY

if [[ ! -d "${LLAMA_CPP_DIR}/.git" ]]; then
  git clone -- "${LLAMA_CPP_SOURCE}" "${LLAMA_CPP_DIR}"
fi

if ! git -C "${LLAMA_CPP_DIR}" diff --quiet || \
   ! git -C "${LLAMA_CPP_DIR}" diff --cached --quiet; then
  echo "Refusing to switch a modified llama.cpp checkout: ${LLAMA_CPP_DIR}" >&2
  exit 2
fi

ORIGIN_URL="$(git -C "${LLAMA_CPP_DIR}" remote get-url origin)"
if [[ "${ALLOW_UNLOCKED}" != "1" ]] && [[ "${ORIGIN_URL}" != "${LOCK_SOURCE}" ]]; then
  echo "Checkout origin ${ORIGIN_URL} does not match locked source ${LOCK_SOURCE}." >&2
  exit 2
fi

git -C "${LLAMA_CPP_DIR}" fetch --depth 1 origin "${LLAMA_CPP_REF}"
git -C "${LLAMA_CPP_DIR}" checkout --detach "${LLAMA_CPP_REF}"
if [[ "$(git -C "${LLAMA_CPP_DIR}" rev-parse HEAD)" != "${LLAMA_CPP_REF}" ]]; then
  echo "Checkout does not match locked commit ${LLAMA_CPP_REF}." >&2
  exit 2
fi

cmake \
  -S "${LLAMA_CPP_DIR}" \
  -B "${BUILD_DIR}" \
  -DGGML_CUDA=ON \
  -DLLAMA_CURL=ON \
  -DBUILD_SHARED_LIBS="${LOCK_SHARED_LIBS}" \
  -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}" \
  -DCMAKE_BUILD_TYPE=Release

cmake --build "${BUILD_DIR}" --target llama-server -j "${BUILD_JOBS}"

CANDIDATE_SHA256="$(sha256sum "${CANDIDATE_BINARY}" | awk '{print $1}')"
echo "candidate llama-server sha256: ${CANDIDATE_SHA256}"
if [[ "${ALLOW_UNLOCKED}" != "1" ]] && [[ "${CANDIDATE_SHA256}" != "${LOCK_BINARY_SHA256}" ]]; then
  echo "Candidate binary differs from the reviewed lock (expected ${LOCK_BINARY_SHA256}). The production binary was not touched; review the toolchain/output and update the lock deliberately." >&2
  exit 3
fi
chmod 0555 "${CANDIDATE_BINARY}"
if [[ "$(sha256sum "${CANDIDATE_BINARY}" | awk '{print $1}')" != "${CANDIDATE_SHA256}" ]]; then
  echo "Candidate bytes changed while sealing; production was not touched." >&2
  exit 3
fi
if ldd "${CANDIDATE_BINARY}" | grep -F -- "${LLM_LAB_DATA_DIR}" >/dev/null; then
  echo "Candidate has mutable data-root shared-library dependencies; production was not touched." >&2
  exit 3
fi

# Execute a candidate only after its bytes match the reviewed digest. Unlocked
# experiments stay in the candidate directory and never replace production.
CANDIDATE_VERSION="$("${CANDIDATE_BINARY}" --version 2>&1)"
echo "${CANDIDATE_VERSION}"
if [[ "${ALLOW_UNLOCKED}" != "1" ]] && [[ "${CANDIDATE_VERSION}" != *"${LOCK_VERSION_CONTAINS}"* ]]; then
  echo "Candidate version output does not contain locked evidence: ${LOCK_VERSION_CONTAINS}" >&2
  exit 3
fi
if [[ "${ALLOW_UNLOCKED}" == "1" ]]; then
  echo "Unlocked candidate retained at ${CANDIDATE_BINARY}; production was not changed."
  exit 0
fi

"${PYTHON_BIN}" - "${REPO_DIR}" "${LLM_LAB_DATA_DIR}" "${LOCK_BINARY}" "${LOCK_BINARY_SHA256}" "${CANDIDATE_BINARY}" <<'PY'
from pathlib import Path
import sys

from llm_lab.paths import LabPaths, install_immutable_file_beneath

repo = Path(sys.argv[1])
data = Path(sys.argv[2])
locked_binary = sys.argv[3]
locked_sha256 = sys.argv[4]
candidate = Path(sys.argv[5])
paths = LabPaths.discover(repo, data)
paths.initialize()
installed = install_immutable_file_beneath(
    paths.data_root,
    locked_binary,
    candidate,
    expected_sha256=locked_sha256,
    purpose="locked runtime binary",
)
print(installed)
PY

echo "locked llama-server: ${LOCKED_BINARY}"
"${LOCKED_BINARY}" --version
