#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="${CUDA_PROG_CLANGD_CONTAINER:-ubuntu_dev}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
HOST_WORKSPACE="${CUDA_PROG_HOST_WORKSPACE:-$(cd -- "${REPO_ROOT}/.." && pwd)}"
CONTAINER_WORKSPACE="${CUDA_PROG_CONTAINER_WORKSPACE:-/workspace}"
CONTAINER_REPO="${CUDA_PROG_CONTAINER_REPO:-${CONTAINER_WORKSPACE}/cuda_programming}"
COMPILE_COMMANDS_DIR="${CUDA_PROG_COMPILE_COMMANDS_DIR:-${CONTAINER_REPO}}"
QUERY_DRIVER="${CUDA_PROG_CLANGD_QUERY_DRIVER:-/usr/bin/c++,/usr/bin/g++,/usr/bin/g++-11,/usr/bin/gcc,/usr/bin/gcc-11,/usr/local/cuda/bin/nvcc}"

function fail() {
  printf 'clangd-docker: %s\n' "$*" >&2
  exit 1
}

if ! command -v docker >/dev/null 2>&1; then
  fail "docker command not found on host"
fi

if ! docker container inspect "${CONTAINER_NAME}" >/dev/null 2>&1; then
  fail "container '${CONTAINER_NAME}' does not exist; run './docker/build_image.sh' and './docker/start_container.sh' first"
fi

if [[ "$(docker container inspect -f '{{.State.Running}}' "${CONTAINER_NAME}")" != "true" ]]; then
  docker start "${CONTAINER_NAME}" >/dev/null
fi

if ! docker exec "${CONTAINER_NAME}" bash -lc 'command -v clangd >/dev/null 2>&1'; then
  fail "clangd not found in container '${CONTAINER_NAME}'; rebuild the image and recreate the container after the Dockerfile clangd change"
fi

exec docker exec -i "${CONTAINER_NAME}" clangd \
  "--compile-commands-dir=${COMPILE_COMMANDS_DIR}" \
  "--path-mappings=${HOST_WORKSPACE}=${CONTAINER_WORKSPACE}" \
  "--query-driver=${QUERY_DRIVER}" \
  "$@"
