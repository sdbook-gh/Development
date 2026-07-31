#!/usr/bin/env bash
# Build PROJ 9.5.1 for Android arm64-v8a (static + shared, -fPIC).
#
# Prerequisites:
#   - Host: cmake, ninja, sqlite3 CLI
#   - Target SQLite already built (scripts/build_sqlite_android.sh):
#       ./output/include/sqlite3.h
#       ./output/arm64-v8a/libsqlite3.a
#       ./output/arm64-v8a/libsqlite3.so
#
# Usage:
#   cd scripts   # recommended so ./output matches sqlite
#   bash build_proj_android.sh
#   bash build_proj_android.sh clean
#   bash build_proj_android.sh -j 8
#
# Output (under ./output, alongside sqlite; clean does NOT remove sqlite):
#   include/proj.h, proj_*.h, proj/...
#   arm64-v8a/libproj.a / libproj.so (+ .debug)
#   share/proj/proj.db
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

PROJ_SRC="${ROOT}/PROJ-9.5.1"
OUT_DIR="./output"
LIB_DIR="${OUT_DIR}/${ABI}"
INC_OUT="${OUT_DIR}/include"
SHARE_OUT="${OUT_DIR}/share/proj"
BUILD_ROOT="${OUT_DIR}/build"
BUILD_STATIC="${BUILD_ROOT}/proj-static"
BUILD_SHARED="${BUILD_ROOT}/proj-shared"
STAGE_STATIC="${BUILD_ROOT}/stage-static"
STAGE_SHARED="${BUILD_ROOT}/stage-shared"

SQLITE_INC="${OUT_DIR}/include"
SQLITE_HDR="${SQLITE_INC}/sqlite3.h"
SQLITE_STATIC="${LIB_DIR}/libsqlite3.a"
SQLITE_SHARED="${LIB_DIR}/libsqlite3.so"

NDK_BIN="${NDK}/toolchains/llvm/prebuilt/${HOST_TAG}/bin"
TOOLCHAIN="${NDK}/build/cmake/android.toolchain.cmake"
STRIP="${NDK_BIN}/llvm-strip"
OBJCOPY="${NDK_BIN}/llvm-objcopy"

CMAKE_BIN="${CMAKE_BIN:-$(command -v cmake || true)}"
NINJA_BIN="${NINJA_BIN:-$(command -v ninja || true)}"
EXE_SQLITE3="${EXE_SQLITE3:-$(command -v sqlite3 || true)}"

die() {
  echo "error: $*" >&2
  exit 1
}

usage() {
  cat <<EOF
Usage: $(basename "$0") [clean] [-j N|--jobs N] [--skip-clean] [--skip-configure] [--build-only] [-h|--help]

  clean             Remove PROJ artifacts under ${OUT_DIR} (keeps sqlite) and exit
  -j N, --jobs N    Max parallel jobs (default: ${DEFAULT_JOBS}, env JOBS)
  --skip-clean      Skip do_clean (preserve build dirs for incremental build)
  --skip-configure  Skip cmake configure, build + install from existing build dir
  --build-only      Shorthand for --skip-clean --skip-configure
  (default)         Clean PROJ artifacts, then build static + shared
EOF
}

do_clean() {
  echo "==> Cleaning PROJ artifacts under ${OUT_DIR}"
  rm -rf \
    "${BUILD_ROOT}/proj-static" \
    "${BUILD_ROOT}/proj-shared" \
    "${BUILD_ROOT}/stage-static" \
    "${BUILD_ROOT}/stage-shared" \
    "${LIB_DIR}/libproj.a" \
    "${LIB_DIR}/libproj.a.debug" \
    "${LIB_DIR}/libproj.so" \
    "${LIB_DIR}/libproj.so.debug"
  # Versioned soname copies if any
  rm -f "${LIB_DIR}"/libproj.so.*
  rm -rf "${INC_OUT}/proj"
  rm -f \
    "${INC_OUT}/proj.h" \
    "${INC_OUT}/proj_experimental.h" \
    "${INC_OUT}/proj_constants.h" \
    "${INC_OUT}/geodesic.h"
  rm -rf "${SHARE_OUT}"
  # Drop empty build root if nothing left
  if [[ -d "${BUILD_ROOT}" ]] && [[ -z "$(ls -A "${BUILD_ROOT}" 2>/dev/null || true)" ]]; then
    rmdir "${BUILD_ROOT}" 2>/dev/null || true
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

[[ -d "${NDK}" ]] || die "NDK not found: ${NDK}"
[[ -f "${TOOLCHAIN}" ]] || die "Android toolchain not found: ${TOOLCHAIN}"
[[ -x "${STRIP}" ]] || die "llvm-strip not found: ${STRIP}"
[[ -x "${OBJCOPY}" ]] || die "llvm-objcopy not found: ${OBJCOPY}"
[[ -x "${CMAKE_BIN}" ]] || die "cmake not found (set CMAKE_BIN)"
[[ -x "${NINJA_BIN}" ]] || die "ninja not found (set NINJA_BIN)"
[[ -x "${EXE_SQLITE3}" ]] || die "host sqlite3 CLI not found (set EXE_SQLITE3)"
[[ -d "${PROJ_SRC}" ]] || die "PROJ source not found: ${PROJ_SRC}"
[[ -f "${PROJ_SRC}/CMakeLists.txt" ]] || die "CMakeLists.txt missing in ${PROJ_SRC}"
[[ -f "${SQLITE_HDR}" ]] || die "sqlite header missing: ${SQLITE_HDR} (run build_sqlite_android.sh first)"
[[ -f "${SQLITE_STATIC}" ]] || die "sqlite static lib missing: ${SQLITE_STATIC}"
[[ -f "${SQLITE_SHARED}" ]] || die "sqlite shared lib missing: ${SQLITE_SHARED}"

if (( ! SKIP_CLEAN )); then
  do_clean
fi
mkdir -p "${LIB_DIR}" "${INC_OUT}" "${SHARE_OUT}" "${BUILD_STATIC}" "${BUILD_SHARED}"

echo "==> NDK:    ${NDK}"
echo "==> API:    ${API}  ABI: ${ABI}  JOBS: ${JOBS}"
echo "==> PROJ:   ${PROJ_SRC}"
echo "==> SQLite: ${SQLITE_INC} + ${LIB_DIR}/libsqlite3.*"
echo "==> cmake:  ${CMAKE_BIN} ($("${CMAKE_BIN}" --version | head -1))"
echo "==> Host sqlite3: ${EXE_SQLITE3}"

COMMON_CMAKE_ARGS=(
  -G Ninja
  -DCMAKE_TOOLCHAIN_FILE="${TOOLCHAIN}"
  -DANDROID_ABI="${ABI}"
  -DANDROID_PLATFORM="android-${API}"
  -DANDROID_STL=c++_static
  -DANDROID_TOOLCHAIN=clang
  -DANDROID_CPP_FEATURES=exceptions
  -DCMAKE_BUILD_TYPE=RelWithDebInfo
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON
  -DCMAKE_FIND_ROOT_PATH_MODE_PROGRAM=NEVER
  -DCMAKE_FIND_ROOT_PATH_MODE_LIBRARY=BOTH
  -DCMAKE_FIND_ROOT_PATH_MODE_INCLUDE=BOTH
  -DCMAKE_FIND_ROOT_PATH_MODE_PACKAGE=BOTH
  -DSQLite3_INCLUDE_DIR="$(cd "${SQLITE_INC}" && pwd)"
  -DEXE_SQLITE3="${EXE_SQLITE3}"
  -DBUILD_APPS=OFF
  -DBUILD_TESTING=OFF
  -DBUILD_EXAMPLES=OFF
  -DENABLE_TIFF=OFF
  -DENABLE_CURL=OFF
  -DNLOHMANN_JSON_ORIGIN=internal
  -DEMBED_PROJ_DATA_PATH=OFF
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

    "${CMAKE_BIN}" -S "${PROJ_SRC}" -B "${build_dir}" \
      "${COMMON_CMAKE_ARGS[@]}" \
      -DBUILD_SHARED_LIBS="${shared_flag}" \
      -DSQLite3_LIBRARY="$(cd "$(dirname "${sqlite_lib}")" && pwd)/$(basename "${sqlite_lib}")" \
      -DCMAKE_INSTALL_PREFIX="$(cd "${stage_dir}" && pwd)"
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

# Prefer unversioned libproj.so; fall back to any libproj.so*
SHARED_SRC=""
if [[ -f "${STAGE_SHARED}/lib/libproj.so" ]]; then
  SHARED_SRC="${STAGE_SHARED}/lib/libproj.so"
elif [[ -f "${STAGE_SHARED}/lib64/libproj.so" ]]; then
  SHARED_SRC="${STAGE_SHARED}/lib64/libproj.so"
else
  SHARED_SRC="$(find "${STAGE_SHARED}" -name 'libproj.so' -o -name 'libproj.so.*' | head -n1 || true)"
fi
[[ -n "${SHARED_SRC}" && -f "${SHARED_SRC}" ]] || die "libproj.so not found after shared install"

# Resolve symlink to real file for strip, then install as libproj.so
SHARED_REAL="$(readlink -f "${SHARED_SRC}")"

cp -f "${STATIC_SRC}" "${LIB_DIR}/libproj.a"
cp -f "${SHARED_REAL}" "${LIB_DIR}/libproj.so"
PROJ_SONAME="$(readelf -d "${LIB_DIR}/libproj.so" | awk '/SONAME/ {gsub(/[\[\]]/,"",$5); print $5; exit}')"
if [[ -n "${PROJ_SONAME}" && "${PROJ_SONAME}" != "libproj.so" ]]; then
  ln -sfn libproj.so "${LIB_DIR}/${PROJ_SONAME}"
fi

echo "==> Collecting headers into ${INC_OUT}"
# Headers are identical from either install; use shared stage
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
ls -la \
  "${LIB_DIR}/libproj.a" \
  "${LIB_DIR}/libproj.a.debug" \
  "${LIB_DIR}/libproj.so" \
  "${LIB_DIR}/libproj.so.debug" \
  "${SHARE_OUT}/proj.db"
ls -la "${INC_OUT}/proj.h" "${INC_OUT}/proj" 2>/dev/null || ls -la "${INC_OUT}/proj.h"
file "${LIB_DIR}/libproj.a" "${LIB_DIR}/libproj.so" "${LIB_DIR}/libproj.so.debug"
