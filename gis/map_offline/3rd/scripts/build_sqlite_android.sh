#!/usr/bin/env bash
# Build SQLite amalgamation (maplibre-native/vendor/sqlite) for Android arm64-v8a.
#
# Usage:
#   bash scripts/build_sqlite_android.sh           # clean sqlite artifacts then build
#   bash scripts/build_sqlite_android.sh clean     # only remove sqlite artifacts
#   bash scripts/build_sqlite_android.sh -j 8      # build with up to 8 parallel jobs
#   JOBS=8 bash scripts/build_sqlite_android.sh
#
# Output (under ./output; clean does NOT remove proj or other libs):
#   output/include/sqlite3.h
#   output/arm64-v8a/libsqlite3.a          (stripped)
#   output/arm64-v8a/libsqlite3.so         (stripped + debuglink)
#   output/arm64-v8a/libsqlite3.a.debug    (debug symbols)
#   output/arm64-v8a/libsqlite3.so.debug   (debug symbols)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

NDK="${NDK:-/mnt/d/devd/android/sdk-linux/ndk/30.0.15729638}"
API="${API:-23}"
ABI="arm64-v8a"
HOST_TAG="linux-x86_64"

if command -v nproc >/dev/null 2>&1; then
  DEFAULT_JOBS="$(nproc)"
else
  DEFAULT_JOBS=4
fi
JOBS="${JOBS:-${DEFAULT_JOBS}}"
SKIP_CLEAN=0
SKIP_CONFIGURE=0

SQLITE_SRC="${ROOT}/maplibre-native/vendor/sqlite/src/sqlite3.c"
SQLITE_INC="${ROOT}/maplibre-native/vendor/sqlite/include"
SQLITE_HDR="${SQLITE_INC}/sqlite3.h"

OUT_DIR="./output"
OBJ_DIR="${OUT_DIR}/obj"
LIB_DIR="${OUT_DIR}/${ABI}"
INC_OUT="${OUT_DIR}/include"

NDK_BIN="${NDK}/toolchains/llvm/prebuilt/${HOST_TAG}/bin"
CC="${NDK_BIN}/aarch64-linux-android${API}-clang"
AR="${NDK_BIN}/llvm-ar"
STRIP="${NDK_BIN}/llvm-strip"
OBJCOPY="${NDK_BIN}/llvm-objcopy"

CFLAGS=(
  -O2
  -g
  -fPIC
  -pipe
  -I"${SQLITE_INC}"
  -DSQLITE_OMIT_LOAD_EXTENSION
  -DSQLITE_THREADSAFE=1
  -DSQLITE_ENABLE_RTREE
  -DHAVE_STRERROR_R
)

OBJ="${OBJ_DIR}/sqlite3.o"
STATIC_LIB="${LIB_DIR}/libsqlite3.a"
STATIC_DBG="${LIB_DIR}/libsqlite3.a.debug"
SHARED_LIB="${LIB_DIR}/libsqlite3.so"
SHARED_DBG="${LIB_DIR}/libsqlite3.so.debug"

die() {
  echo "error: $*" >&2
  exit 1
}

usage() {
  cat <<EOF
Usage: $(basename "$0") [clean] [-j N|--jobs N] [--skip-clean] [--skip-configure] [--build-only] [-h|--help]

  clean             Remove SQLite artifacts under ${OUT_DIR} (keeps proj etc.) and exit
  -j N, --jobs N    Max parallel jobs (default: ${DEFAULT_JOBS}, env JOBS)
  --skip-clean      Skip do_clean (preserve build artifacts for incremental build)
  --skip-configure  No-op for SQLite (direct gcc, no configure step)
  --build-only      Shorthand for --skip-clean --skip-configure
  (default)         Clean SQLite artifacts, then build
EOF
}

do_clean() {
  echo "==> Cleaning SQLite artifacts under ${OUT_DIR}"
  rm -rf "${OBJ_DIR}"
  rm -f \
    "${STATIC_LIB}" \
    "${STATIC_DBG}" \
    "${SHARED_LIB}" \
    "${SHARED_DBG}" \
    "${INC_OUT}/sqlite3.h"
  # Drop empty dirs left behind (do not remove shared output/include or arm64-v8a)
  if [[ -d "${OBJ_DIR}" ]] && [[ -z "$(ls -A "${OBJ_DIR}" 2>/dev/null || true)" ]]; then
    rmdir "${OBJ_DIR}" 2>/dev/null || true
  fi
}

# Extract debug info, strip binary, and add gnu-debuglink back to the stripped file.
# $1 = binary path, $2 = debug path (*.debug)
separate_debug_and_strip() {
  local bin="$1"
  local dbg="$2"
  "${OBJCOPY}" --only-keep-debug "${bin}" "${dbg}"
  "${STRIP}" --strip-unneeded "${bin}"
  (
    cd "$(dirname "${bin}")"
    "${OBJCOPY}" --add-gnu-debuglink="$(basename "${dbg}")" "$(basename "${bin}")"
  )
  chmod -x "${dbg}" 2>/dev/null || true
}

# Run up to JOBS background tasks; wait for a slot when full.
# Usage: run_parallel 'command ...'
_pids=()
run_parallel() {
  while ((${#_pids[@]} >= JOBS)); do
    local pid="${_pids[0]}"
    wait "${pid}" || die "parallel job ${pid} failed"
    _pids=("${_pids[@]:1}")
  done
  "$@" &
  _pids+=("$!")
}

wait_parallel() {
  local pid status=0
  for pid in "${_pids[@]+"${_pids[@]}"}"; do
    wait "${pid}" || status=1
  done
  _pids=()
  ((status == 0)) || die "one or more parallel jobs failed"
}

ONLY_CLEAN=0
while (($# > 0)); do
  case "$1" in
    clean)
      ONLY_CLEAN=1
      shift
      ;;
    -j|--jobs)
      (($# >= 2)) || die "$1 requires a number"
      JOBS="$2"
      shift 2
      ;;
    -j*)
      JOBS="${1#-j}"
      shift
      ;;
    --skip-clean)
      SKIP_CLEAN=1
      shift
      ;;
    --skip-configure)
      SKIP_CONFIGURE=1
      shift
      ;;
    --build-only)
      SKIP_CLEAN=1
      SKIP_CONFIGURE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown argument: $1 (try --help)"
      ;;
  esac
done

[[ "${JOBS}" =~ ^[1-9][0-9]*$ ]] || die "JOBS must be a positive integer, got: ${JOBS}"

if ((ONLY_CLEAN)); then
  do_clean
  echo "==> Clean done"
  exit 0
fi

[[ -d "${NDK}" ]] || die "NDK not found: ${NDK}"
[[ -x "${CC}" ]] || die "clang not found: ${CC}"
[[ -x "${AR}" ]] || die "llvm-ar not found: ${AR}"
[[ -x "${STRIP}" ]] || die "llvm-strip not found: ${STRIP}"
[[ -x "${OBJCOPY}" ]] || die "llvm-objcopy not found: ${OBJCOPY}"
[[ -f "${SQLITE_SRC}" ]] || die "sqlite3.c not found: ${SQLITE_SRC}"
[[ -f "${SQLITE_HDR}" ]] || die "sqlite3.h not found: ${SQLITE_HDR}"

if (( ! SKIP_CLEAN )); then
  do_clean
fi
mkdir -p "${OBJ_DIR}" "${LIB_DIR}" "${INC_OUT}"

echo "==> NDK: ${NDK}"
echo "==> API: ${API}  ABI: ${ABI}  JOBS: ${JOBS}"
echo "==> CC:  ${CC}"
echo "==> SRC: ${SQLITE_SRC}"

echo "==> Compiling sqlite3.o (-fPIC -g)"
"${CC}" "${CFLAGS[@]}" -c "${SQLITE_SRC}" -o "${OBJ}"

echo "==> Creating .a / .so in parallel"
run_parallel "${AR}" rcs "${STATIC_LIB}" "${OBJ}"
run_parallel "${CC}" -shared -fPIC -g -Wl,-soname,libsqlite3.so -o "${SHARED_LIB}" "${OBJ}"
wait_parallel

echo "==> Separating debug symbols and stripping (parallel)"
run_parallel bash -c '
  set -euo pipefail
  cp -f "$1" "$2"
  "$3" --strip-unneeded "$1"
  chmod -x "$2" 2>/dev/null || true
' _ "${STATIC_LIB}" "${STATIC_DBG}" "${STRIP}"
run_parallel separate_debug_and_strip "${SHARED_LIB}" "${SHARED_DBG}"
wait_parallel

cp -f "${SQLITE_HDR}" "${INC_OUT}/sqlite3.h"

# Extract sqlite3ext.h from the amalgamation (it's embedded inside sqlite3.c)
SQLITE3EXT_HDR="${INC_OUT}/sqlite3ext.h"
if grep -q 'Begin file sqlite3ext.h' "${SQLITE_SRC}"; then
  echo "==> Extracting sqlite3ext.h from amalgamation"
  # Extract the sqlite3ext.h block, strip begin/end markers, and fix the include
  awk '/^\/\*\*\*\*\*\*\*\*\*\*\*\*\*\* Begin file sqlite3ext.h/{found=1; next}
       /^\/\*\*\*\*\*\*\*\*\*\*\*\*\*\* End of sqlite3ext.h/{exit}
       found' "${SQLITE_SRC}" \
    | sed 's|^/\* #include "sqlite3.h" \*/$|#include "sqlite3.h"|' \
    > "${SQLITE3EXT_HDR}"
  chmod 644 "${SQLITE3EXT_HDR}"
  echo "==> Created ${SQLITE3EXT_HDR}"
else
  echo "warning: sqlite3ext.h section not found in amalgamation" >&2
fi

echo "==> Done"
ls -la "${STATIC_LIB}" "${STATIC_DBG}" "${SHARED_LIB}" "${SHARED_DBG}" "${INC_OUT}/sqlite3.h"
file "${STATIC_LIB}" "${SHARED_LIB}" "${SHARED_DBG}"
