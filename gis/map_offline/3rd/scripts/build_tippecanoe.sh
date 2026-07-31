#!/usr/bin/env bash
# Build tippecanoe for Linux x86_64 (selected CLIs + libs, -fPIC).
#
# Prerequisites:
#   - SQLite already built (scripts/build_sqlite.sh):
#       ./output/include/sqlite3.h
#       ./output/x86_64/libsqlite3.so
#
# Usage:
#   cd scripts
#   bash build_tippecanoe.sh
#   bash build_tippecanoe.sh clean
#   bash build_tippecanoe.sh -j 8
#
# Output (clean does NOT remove sqlite/proj/gdal):
#   bin/{tippecanoe,tile-join,tile-join-ext}
#   x86_64/libtippecanoe|libtile-join|libtile-join-ext .{a,so}(+.debug)
#   include/{progress_hook,log_hook,exit_util,tile-join-ext}.h
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
ABI="x86_64"

if command -v nproc >/dev/null 2>&1; then
  DEFAULT_JOBS="$(nproc)"
else
  DEFAULT_JOBS=4
fi
JOBS="${JOBS:-${DEFAULT_JOBS}}"
SKIP_CLEAN=0
SKIP_CONFIGURE=0

TIP_SRC="${ROOT}/tippecanoe-2.80.0-base"
OUT_DIR="./output"
LIB_DIR="${OUT_DIR}/${ABI}"
INC_OUT="${OUT_DIR}/include"
BIN_OUT="${OUT_DIR}/bin"

SQLITE_HDR="${INC_OUT}/sqlite3.h"
SQLITE_SHARED="${LIB_DIR}/libsqlite3.so"

if command -v clang >/dev/null 2>&1; then
  CC="${CC:-clang}"
  CXX="${CXX:-clang++}"
else
  CC="${CC:-gcc}"
  CXX="${CXX:-g++}"
fi
AR="${AR:-$(command -v llvm-ar 2>/dev/null || command -v ar)}"
STRIP="${STRIP:-$(command -v llvm-strip 2>/dev/null || command -v strip)}"
OBJCOPY="${OBJCOPY:-$(command -v llvm-objcopy 2>/dev/null || command -v objcopy)}"

LIBS=(
  libtippecanoe.a
  libtippecanoe.so
  libtile-join.a
  libtile-join.so
  libtile-join-ext.a
  libtile-join-ext.so
)

BINS=(
  tippecanoe
  tile-join
  tile-join-ext
)

HEADERS=(
  progress_hook.h
  log_hook.h
  exit_util.h
  tile-join-ext.h
)

die() {
  echo "error: $*" >&2
  exit 1
}

abs_path() {
  local p="$1"
  if [[ -d "${p}" ]]; then
    (cd "${p}" && pwd)
  else
    echo "$(cd "$(dirname "${p}")" && pwd)/$(basename "${p}")"
  fi
}

usage() {
  cat <<EOF
Usage: $(basename "$0") [clean] [-j N|--jobs N] [--skip-clean] [--skip-configure] [--build-only] [-h|--help]

  clean             Remove tippecanoe host artifacts under ${OUT_DIR} and exit
  -j N, --jobs N    Max parallel jobs (default: ${DEFAULT_JOBS}, env JOBS)
  --skip-clean      Skip do_clean (preserve build artifacts for incremental build)
  --skip-configure  No-op for tippecanoe (uses Make, not CMake)
  --build-only      Shorthand for --skip-clean --skip-configure
  (default)         Clean tippecanoe artifacts, then build tippecanoe/tile-join/tile-join-ext + libs
EOF
}

do_clean() {
  echo "==> Cleaning tippecanoe host artifacts under ${OUT_DIR}"
  local name hdr
  for name in "${LIBS[@]}"; do
    rm -f "${LIB_DIR}/${name}" "${LIB_DIR}/${name}.debug"
  done
  for name in "${BINS[@]}"; do
    rm -f "${BIN_OUT}/${name}"
  done
  for hdr in "${HEADERS[@]}"; do
    rm -f "${INC_OUT}/${hdr}"
  done
  if [[ -d "${TIP_SRC}" ]]; then
    echo "==> make clean in ${TIP_SRC}"
    make -C "${TIP_SRC}" clean >/dev/null 2>&1 || true
  fi
}

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

command -v "${CC}" >/dev/null || die "C compiler not found: ${CC}"
command -v "${CXX}" >/dev/null || die "C++ compiler not found: ${CXX}"
[[ -x "${AR}" ]] || die "ar not found: ${AR}"
[[ -x "${STRIP}" ]] || die "strip not found"
[[ -x "${OBJCOPY}" ]] || die "objcopy not found"
[[ -d "${TIP_SRC}" ]] || die "tippecanoe source not found: ${TIP_SRC}"
[[ -f "${TIP_SRC}/Makefile" ]] || die "Makefile missing in ${TIP_SRC}"
[[ -f "${SQLITE_HDR}" ]] || die "sqlite header missing: ${SQLITE_HDR} (run build_sqlite.sh first)"
[[ -f "${SQLITE_SHARED}" ]] || die "sqlite shared lib missing: ${SQLITE_SHARED}"

if (( ! SKIP_CLEAN )); then
  do_clean
fi
mkdir -p "${LIB_DIR}" "${INC_OUT}" "${BIN_OUT}"

INC_ABS="$(abs_path "${INC_OUT}")"
LIB_ABS="$(abs_path "${LIB_DIR}")"
SQLITE_SO_ABS="$(abs_path "${SQLITE_SHARED}")"

echo "==> ABI: ${ABI}  JOBS: ${JOBS}"
echo "==> SRC: ${TIP_SRC}"
echo "==> SQLite: ${SQLITE_SO_ABS}"
echo "==> CXX: ${CXX}"

echo "==> Building tippecanoe CLIs + libs (-fPIC, host)"
make -C "${TIP_SRC}" -j"${JOBS}" \
  CC="${CC}" \
  CXX="${CXX}" \
  AR="${AR}" \
  INCLUDES="-I${INC_ABS} -I. -Iclipper2/include" \
  LIBS="-L${LIB_ABS} -Wl,-rpath,${LIB_ABS}" \
  LDFLAGS="-L${LIB_ABS} -Wl,-rpath,${LIB_ABS}" \
  SQLITE_LIB="${SQLITE_SO_ABS}" \
  PTHREAD_LIB=-lpthread \
  LOG_LIB= \
  "${BINS[@]}" \
  libtippecanoe.a libtippecanoe.so \
  libtile-join.a libtile-join.so \
  libtile-join-ext.a libtile-join-ext.so

echo "==> Collecting libraries into ${LIB_DIR}"
local_name=""
for local_name in "${LIBS[@]}"; do
  [[ -f "${TIP_SRC}/${local_name}" ]] || die "missing build product: ${TIP_SRC}/${local_name}"
  cp -f "${TIP_SRC}/${local_name}" "${LIB_DIR}/${local_name}"
done

echo "==> Collecting binaries into ${BIN_OUT}"
for local_name in "${BINS[@]}"; do
  [[ -f "${TIP_SRC}/${local_name}" ]] || die "missing binary: ${TIP_SRC}/${local_name}"
  cp -f "${TIP_SRC}/${local_name}" "${BIN_OUT}/${local_name}"
  "${STRIP}" --strip-unneeded "${BIN_OUT}/${local_name}" 2>/dev/null || true
done

echo "==> Collecting headers into ${INC_OUT}"
for local_name in "${HEADERS[@]}"; do
  [[ -f "${TIP_SRC}/${local_name}" ]] || die "missing header: ${TIP_SRC}/${local_name}"
  cp -f "${TIP_SRC}/${local_name}" "${INC_OUT}/${local_name}"
done

echo "==> Separating debug symbols and stripping"
for local_name in libtippecanoe.a libtile-join.a libtile-join-ext.a; do
  cp -f "${LIB_DIR}/${local_name}" "${LIB_DIR}/${local_name}.debug"
  "${STRIP}" --strip-unneeded "${LIB_DIR}/${local_name}"
  chmod -x "${LIB_DIR}/${local_name}.debug" 2>/dev/null || true
done
for local_name in libtippecanoe.so libtile-join.so libtile-join-ext.so; do
  separate_debug_and_strip "${LIB_DIR}/${local_name}" "${LIB_DIR}/${local_name}.debug"
done

echo "==> Done"
ls -la \
  "${LIB_DIR}/libtippecanoe.so" \
  "${LIB_DIR}/libtile-join.so" \
  "${LIB_DIR}/libtile-join-ext.so" \
  "${BIN_OUT}/tippecanoe" \
  "${BIN_OUT}/tile-join" \
  "${BIN_OUT}/tile-join-ext"
file \
  "${LIB_DIR}/libtippecanoe.so" \
  "${BIN_OUT}/tippecanoe"
