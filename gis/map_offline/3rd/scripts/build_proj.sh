#!/usr/bin/env bash
# Build PROJ 9.5.1 for Linux x86_64 (static + shared libs only, -fPIC).
#
# Prerequisites:
#   - Host: cmake, ninja, sqlite3 CLI (for generating proj.db only)
#   - SQLite already built (scripts/build_sqlite.sh):
#       ./output/include/sqlite3.h
#       ./output/x86_64/libsqlite3.{a,so}
#
# Usage:
#   cd scripts
#   bash build_proj.sh
#   bash build_proj.sh clean
#   bash build_proj.sh -j 8
#
# Output (no PROJ CLIs; clean does NOT remove sqlite):
#   x86_64/libproj.{a,so}(+.debug)
#   include/proj.h, proj/...
#   share/proj/
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

PROJ_SRC="${ROOT}/PROJ-9.5.1"
OUT_DIR="./output"
LIB_DIR="${OUT_DIR}/${ABI}"
INC_OUT="${OUT_DIR}/include"
SHARE_OUT="${OUT_DIR}/share/proj"
BUILD_ROOT="${OUT_DIR}/build"
BUILD_STATIC="${BUILD_ROOT}/proj-host-static"
BUILD_SHARED="${BUILD_ROOT}/proj-host-shared"
STAGE_STATIC="${BUILD_ROOT}/stage-proj-host-static"
STAGE_SHARED="${BUILD_ROOT}/stage-proj-host-shared"

SQLITE_INC="${OUT_DIR}/include"
SQLITE_HDR="${SQLITE_INC}/sqlite3.h"
SQLITE_STATIC="${LIB_DIR}/libsqlite3.a"
SQLITE_SHARED="${LIB_DIR}/libsqlite3.so"

if command -v clang >/dev/null 2>&1; then
  CC="${CC:-clang}"
  CXX="${CXX:-clang++}"
else
  CC="${CC:-gcc}"
  CXX="${CXX:-g++}"
fi
STRIP="${STRIP:-$(command -v llvm-strip 2>/dev/null || command -v strip)}"
OBJCOPY="${OBJCOPY:-$(command -v llvm-objcopy 2>/dev/null || command -v objcopy)}"
CMAKE_BIN="${CMAKE_BIN:-$(command -v cmake || true)}"
NINJA_BIN="${NINJA_BIN:-$(command -v ninja || true)}"
EXE_SQLITE3="${EXE_SQLITE3:-$(command -v sqlite3 || true)}"

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

  clean             Remove PROJ host artifacts under ${OUT_DIR} (keeps sqlite) and exit
  -j N, --jobs N    Max parallel jobs (default: ${DEFAULT_JOBS}, env JOBS)
  --skip-clean      Skip do_clean (preserve build dirs for incremental build)
  --skip-configure  Skip cmake configure, build + install from existing build dir
  --build-only      Shorthand for --skip-clean --skip-configure
  (default)         Clean PROJ artifacts, then build static + shared libs
EOF
}

do_clean() {
  echo "==> Cleaning PROJ host artifacts under ${OUT_DIR}"
  rm -rf \
    "${BUILD_STATIC}" \
    "${BUILD_SHARED}" \
    "${STAGE_STATIC}" \
    "${STAGE_SHARED}" \
    "${SHARE_OUT}"
  rm -f \
    "${LIB_DIR}/libproj.a" \
    "${LIB_DIR}/libproj.a.debug" \
    "${LIB_DIR}/libproj.so" \
    "${LIB_DIR}/libproj.so.debug"
  rm -f "${LIB_DIR}"/libproj.so.*
  rm -rf "${INC_OUT}/proj"
  rm -f \
    "${INC_OUT}/proj.h" \
    "${INC_OUT}/proj_experimental.h" \
    "${INC_OUT}/proj_constants.h" \
    "${INC_OUT}/proj_symbol_rename.h" \
    "${INC_OUT}/geodesic.h"
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

[[ -x "${STRIP}" ]] || die "strip not found"
[[ -x "${OBJCOPY}" ]] || die "objcopy not found"
[[ -x "${CMAKE_BIN}" ]] || die "cmake not found (set CMAKE_BIN)"
[[ -x "${NINJA_BIN}" ]] || die "ninja not found (set NINJA_BIN)"
[[ -x "${EXE_SQLITE3}" ]] || die "host sqlite3 CLI not found (set EXE_SQLITE3=/usr/bin/sqlite3)"
command -v "${CC}" >/dev/null || die "C compiler not found: ${CC}"
command -v "${CXX}" >/dev/null || die "C++ compiler not found: ${CXX}"
[[ -d "${PROJ_SRC}" ]] || die "PROJ source not found: ${PROJ_SRC}"
[[ -f "${PROJ_SRC}/CMakeLists.txt" ]] || die "CMakeLists.txt missing in ${PROJ_SRC}"
[[ -f "${SQLITE_HDR}" ]] || die "sqlite header missing: ${SQLITE_HDR} (run build_sqlite.sh first)"
[[ -f "${SQLITE_STATIC}" ]] || die "missing ${SQLITE_STATIC}"
[[ -f "${SQLITE_SHARED}" ]] || die "missing ${SQLITE_SHARED}"

if (( ! SKIP_CLEAN )); then
  do_clean
fi
mkdir -p "${LIB_DIR}" "${INC_OUT}" "${SHARE_OUT}" "${BUILD_STATIC}" "${BUILD_SHARED}" "${BUILD_ROOT}"

echo "==> ABI:    ${ABI}  JOBS: ${JOBS}"
echo "==> PROJ:   ${PROJ_SRC}"
echo "==> SQLite: ${SQLITE_INC} + ${LIB_DIR}/libsqlite3.*"
echo "==> cmake:  ${CMAKE_BIN} ($("${CMAKE_BIN}" --version | head -1))"
echo "==> CC/CXX: ${CC} / ${CXX}"
echo "==> Host sqlite3 (proj.db only): ${EXE_SQLITE3}"

COMMON_CMAKE_ARGS=(
  -G Ninja
  -DCMAKE_BUILD_TYPE=RelWithDebInfo
  -DCMAKE_C_COMPILER="${CC}"
  -DCMAKE_CXX_COMPILER="${CXX}"
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON
  -DCMAKE_INSTALL_LIBDIR=lib
  -DSQLite3_INCLUDE_DIR="$(abs_path "${SQLITE_INC}")"
  -DEXE_SQLITE3="${EXE_SQLITE3}"
  -DBUILD_APPS=OFF
  -DBUILD_TESTING=OFF
  -DBUILD_EXAMPLES=OFF
  -DBUILD_PROJSYNC=OFF
  -DENABLE_TIFF=OFF
  -DENABLE_CURL=OFF
  -DNLOHMANN_JSON_ORIGIN=internal
  -DEMBED_PROJ_DATA_PATH=OFF
  -DCMAKE_FIND_USE_PACKAGE_REGISTRY=OFF
  -DCMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY=OFF
)

configure_and_build() {
  local kind="$1"
  local build_dir="$2"
  local stage_dir="$3"
  local sqlite_lib="$4"
  local shared_flag="$5"

  rm -rf "${stage_dir}"
  mkdir -p "${build_dir}" "${stage_dir}"

  if (( SKIP_CONFIGURE )); then
    echo "==> Skipping configure PROJ (${kind}) (--skip-configure)"
    [[ -f "${build_dir}/CMakeCache.txt" ]] || \
      die "build dir not configured: ${build_dir} (run without --skip-configure first)"
  else
    echo "==> Configure PROJ (${kind})"

    env PKG_CONFIG_LIBDIR=/dev/null PKG_CONFIG_PATH= \
    "${CMAKE_BIN}" -S "${PROJ_SRC}" -B "${build_dir}" \
      "${COMMON_CMAKE_ARGS[@]}" \
      -DBUILD_SHARED_LIBS="${shared_flag}" \
      -DSQLite3_LIBRARY="$(abs_path "${sqlite_lib}")" \
      -DCMAKE_INSTALL_PREFIX="$(abs_path "${stage_dir}")"
  fi

  echo "==> Build PROJ (${kind}) -j${JOBS}"
  "${CMAKE_BIN}" --build "${build_dir}" --parallel "${JOBS}"

  echo "==> Install PROJ (${kind}) -> ${stage_dir}"
  "${CMAKE_BIN}" --install "${build_dir}"
}

configure_and_build static "${BUILD_STATIC}" "${STAGE_STATIC}" "${SQLITE_STATIC}" OFF
configure_and_build shared "${BUILD_SHARED}" "${STAGE_SHARED}" "${SQLITE_SHARED}" ON

echo "==> Collecting libraries into ${LIB_DIR}"
STATIC_SRC="$(find "${STAGE_STATIC}" -name 'libproj.a' | head -n1)"
[[ -n "${STATIC_SRC}" && -f "${STATIC_SRC}" ]] || die "libproj.a not found after static install"

SHARED_SRC=""
if [[ -f "${STAGE_SHARED}/lib/libproj.so" ]]; then
  SHARED_SRC="${STAGE_SHARED}/lib/libproj.so"
elif [[ -f "${STAGE_SHARED}/lib64/libproj.so" ]]; then
  SHARED_SRC="${STAGE_SHARED}/lib64/libproj.so"
else
  SHARED_SRC="$(find "${STAGE_SHARED}" \( -name 'libproj.so' -o -name 'libproj.so.*' \) | head -n1 || true)"
fi
[[ -n "${SHARED_SRC}" && -f "${SHARED_SRC}" ]] || die "libproj.so not found after shared install"
SHARED_REAL="$(readlink -f "${SHARED_SRC}")"

cp -f "${STATIC_SRC}" "${LIB_DIR}/libproj.a"
cp -f "${SHARED_REAL}" "${LIB_DIR}/libproj.so"
# GDAL (and the dynamic linker) look up the ELF SONAME (libproj.so.25), not
# just the unversioned libproj.so we install as the public name.
PROJ_SONAME="$(readelf -d "${LIB_DIR}/libproj.so" | awk '/SONAME/ {gsub(/[\[\]]/,"",$5); print $5; exit}')"
if [[ -n "${PROJ_SONAME}" && "${PROJ_SONAME}" != "libproj.so" ]]; then
  ln -sfn libproj.so "${LIB_DIR}/${PROJ_SONAME}"
fi

echo "==> Collecting headers into ${INC_OUT}"
if [[ -d "${STAGE_SHARED}/include" ]]; then
  cp -a "${STAGE_SHARED}/include/." "${INC_OUT}/"
elif [[ -d "${STAGE_STATIC}/include" ]]; then
  cp -a "${STAGE_STATIC}/include/." "${INC_OUT}/"
else
  die "PROJ headers not found in install prefix"
fi

echo "==> Collecting proj data into ${SHARE_OUT}"
DATA_SRC=""
if [[ -d "${STAGE_SHARED}/share/proj" ]]; then
  DATA_SRC="${STAGE_SHARED}/share/proj"
elif [[ -d "${STAGE_STATIC}/share/proj" ]]; then
  DATA_SRC="${STAGE_STATIC}/share/proj"
fi
[[ -n "${DATA_SRC}" ]] || die "share/proj not found after install"
mkdir -p "${SHARE_OUT}"
cp -a "${DATA_SRC}/." "${SHARE_OUT}/"

echo "==> Separating debug symbols and stripping"
cp -f "${LIB_DIR}/libproj.a" "${LIB_DIR}/libproj.a.debug"
"${STRIP}" --strip-unneeded "${LIB_DIR}/libproj.a"
chmod -x "${LIB_DIR}/libproj.a.debug" 2>/dev/null || true
separate_debug_and_strip "${LIB_DIR}/libproj.so" "${LIB_DIR}/libproj.so.debug"

echo "==> Done"
ls -la "${LIB_DIR}/libproj.a" "${LIB_DIR}/libproj.so" "${SHARE_OUT}/proj.db"
ls -la "${INC_OUT}/proj.h" 2>/dev/null || true
file "${LIB_DIR}/libproj.a" "${LIB_DIR}/libproj.so"
