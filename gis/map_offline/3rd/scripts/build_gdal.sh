#!/usr/bin/env bash
# Build GDAL 3.10.3 for Linux x86_64 (static + shared + GDAL apps, -fPIC).
#
# Prerequisites:
#   - Host: cmake, ninja
#   - SQLite + PROJ already built under ./output (build_sqlite.sh, build_proj.sh):
#       ./output/include/{sqlite3.h,proj.h}
#       ./output/x86_64/lib{sqlite3,proj}.{a,so}
#
# Usage:
#   cd scripts
#   bash build_gdal.sh
#   bash build_gdal.sh clean
#   bash build_gdal.sh -j 8
#
# Output:
#   bin/{gdalinfo,ogrinfo,ogr2ogr,...}   # GDAL CLIs only among host packages
#   x86_64/libgdal.{a,so}(+.debug)
#   include/gdal*.h ...
#   share/gdal/
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

GDAL_SRC="${ROOT}/gdal-3.10.3"
OUT_DIR="./output"
LIB_DIR="${OUT_DIR}/${ABI}"
INC_OUT="${OUT_DIR}/include"
BIN_OUT="${OUT_DIR}/bin"
SHARE_OUT="${OUT_DIR}/share/gdal"
BUILD_ROOT="${OUT_DIR}/build"
BUILD_STATIC="${BUILD_ROOT}/gdal-host-static"
BUILD_SHARED="${BUILD_ROOT}/gdal-host-shared"
STAGE_STATIC="${BUILD_ROOT}/stage-gdal-host-static"
STAGE_SHARED="${BUILD_ROOT}/stage-gdal-host-shared"
HEADER_MANIFEST="${BUILD_ROOT}/gdal-host-installed-headers.txt"
BIN_MANIFEST="${BUILD_ROOT}/gdal-host-installed-bins.txt"

SQLITE_HDR="${INC_OUT}/sqlite3.h"
PROJ_HDR="${INC_OUT}/proj.h"
SQLITE_STATIC="${LIB_DIR}/libsqlite3.a"
SQLITE_SHARED="${LIB_DIR}/libsqlite3.so"
PROJ_STATIC="${LIB_DIR}/libproj.a"
PROJ_SHARED="${LIB_DIR}/libproj.so"

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

die() {
  echo "error: $*" >&2
  exit 1
}

usage() {
  cat <<EOF
Usage: $(basename "$0") [clean] [-j N|--jobs N] [--skip-clean] [--skip-configure] [--build-only] [-h|--help]

  clean             Remove GDAL host artifacts under ${OUT_DIR} (keeps sqlite/proj) and exit
  -j N, --jobs N    Max parallel jobs (default: ${DEFAULT_JOBS}, env JOBS)
  --skip-clean      Skip do_clean (preserve build dirs for incremental build)
  --skip-configure  Skip cmake configure, build + install from existing build dir
  --build-only      Shorthand for --skip-clean --skip-configure
  (default)         Clean GDAL artifacts, then build static + shared + apps
EOF
}

abs_path() {
  local p="$1"
  if [[ -d "${p}" ]]; then
    (cd "${p}" && pwd)
  else
    echo "$(cd "$(dirname "${p}")" && pwd)/$(basename "${p}")"
  fi
}

remove_gdal_headers() {
  if [[ -f "${HEADER_MANIFEST}" ]]; then
    local rel
    while IFS= read -r rel || [[ -n "${rel}" ]]; do
      [[ -z "${rel}" ]] && continue
      rm -rf "${INC_OUT}/${rel}"
    done < "${HEADER_MANIFEST}"
    return
  fi
  local f
  for f in "${INC_OUT}"/gdal*.h "${INC_OUT}"/ogr*.h "${INC_OUT}"/cpl*.h \
           "${INC_OUT}"/gdal_*.h "${INC_OUT}"/GDAL*.h \
           "${INC_OUT}"/rawdataset.h "${INC_OUT}"/vrtdataset.h \
           "${INC_OUT}"/memdataset.h "${INC_OUT}"/gdalwarper.h \
           "${INC_OUT}"/gdalgrid.h "${INC_OUT}"/gdal_alg.h \
           "${INC_OUT}"/gdal_priv.h "${INC_OUT}"/gdal_frmts.h \
           "${INC_OUT}"/gdal_proxy.h "${INC_OUT}"/gdal_rat.h \
           "${INC_OUT}"/gdal_mdreader.h "${INC_OUT}"/gdal_utils.h \
           "${INC_OUT}"/gdal_pam.h "${INC_OUT}"/gdal_version.h \
           "${INC_OUT}"/gdalcachedpixelaccessor.h \
           "${INC_OUT}"/gdalgeorefpamdataset.h \
           "${INC_OUT}"/gdaljp2*.h "${INC_OUT}"/gdal_avif.h \
           "${INC_OUT}"/ogr_*.h "${INC_OUT}"/ogrsf_frmts.h; do
    rm -f "${f}" 2>/dev/null || true
  done
}

do_clean() {
  echo "==> Cleaning GDAL host artifacts under ${OUT_DIR}"
  remove_gdal_headers
  rm -rf \
    "${BUILD_STATIC}" \
    "${BUILD_SHARED}" \
    "${STAGE_STATIC}" \
    "${STAGE_SHARED}" \
    "${SHARE_OUT}"
  rm -f \
    "${LIB_DIR}/libgdal.a" \
    "${LIB_DIR}/libgdal.a.debug" \
    "${LIB_DIR}/libgdal.so" \
    "${LIB_DIR}/libgdal.so.debug" \
    "${HEADER_MANIFEST}"
  rm -f "${LIB_DIR}"/libgdal.so.*
  if [[ -f "${BIN_MANIFEST}" ]]; then
    local b
    while IFS= read -r b || [[ -n "${b}" ]]; do
      [[ -z "${b}" ]] && continue
      rm -f "${BIN_OUT}/${b}"
    done < "${BIN_MANIFEST}"
    rm -f "${BIN_MANIFEST}"
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

# Rewrite ELF SONAME / DT_NEEDED string in-place (same dynstr slot, shortened).
# Avoids requiring patchelf when only changing libgdal.so.N -> libgdal.so.
elf_shorten_dynstr() {
  local file="$1"
  local old="$2"
  local new="$3"
  python3 - "$file" "$old" "$new" <<'PY'
import sys
path, old, new = sys.argv[1], sys.argv[2].encode(), sys.argv[3].encode()
if len(new) >= len(old):
    raise SystemExit(f"refusing non-shortening rewrite: {old!r} -> {new!r}")
data = bytearray(open(path, "rb").read())
needle = old + b"\0"
repl = new + b"\0" + b"\0" * (len(old) - len(new))
count = 0
start = 0
while True:
    i = data.find(needle, start)
    if i < 0:
        break
    data[i:i + len(needle)] = repl
    count += 1
    start = i + len(needle)
if count == 0:
    raise SystemExit(f"string not found in {path}: {old.decode()}")
open(path, "wb").write(data)
print(f"rewrote {count} occurrence(s) of {old.decode()} -> {new.decode()} in {path}")
PY
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
command -v "${CC}" >/dev/null || die "C compiler not found: ${CC}"
command -v "${CXX}" >/dev/null || die "C++ compiler not found: ${CXX}"
[[ -d "${GDAL_SRC}" ]] || die "GDAL source not found: ${GDAL_SRC}"
[[ -f "${GDAL_SRC}/CMakeLists.txt" ]] || die "CMakeLists.txt missing in ${GDAL_SRC}"
[[ -f "${SQLITE_HDR}" ]] || die "sqlite header missing: ${SQLITE_HDR} (run build_sqlite.sh first)"
[[ -f "${PROJ_HDR}" ]] || die "proj header missing: ${PROJ_HDR} (run build_proj.sh first)"
[[ -f "${SQLITE_STATIC}" ]] || die "missing ${SQLITE_STATIC}"
[[ -f "${SQLITE_SHARED}" ]] || die "missing ${SQLITE_SHARED}"
[[ -f "${PROJ_STATIC}" ]] || die "missing ${PROJ_STATIC}"
[[ -f "${PROJ_SHARED}" ]] || die "missing ${PROJ_SHARED}"

if (( ! SKIP_CLEAN )); then
  do_clean
fi
mkdir -p "${LIB_DIR}" "${INC_OUT}" "${BIN_OUT}" "${SHARE_OUT}" "${BUILD_STATIC}" "${BUILD_SHARED}" "${BUILD_ROOT}"

INC_ABS="$(abs_path "${INC_OUT}")"
SQLITE_STATIC_ABS="$(abs_path "${SQLITE_STATIC}")"
SQLITE_SHARED_ABS="$(abs_path "${SQLITE_SHARED}")"
PROJ_STATIC_ABS="$(abs_path "${PROJ_STATIC}")"
PROJ_SHARED_ABS="$(abs_path "${PROJ_SHARED}")"

echo "==> ABI:    ${ABI}  JOBS: ${JOBS}"
echo "==> GDAL:   ${GDAL_SRC}"
echo "==> PROJ:   ${PROJ_HDR} + ${LIB_DIR}/libproj.*"
echo "==> SQLite: ${SQLITE_HDR} + ${LIB_DIR}/libsqlite3.*"
echo "==> cmake:  ${CMAKE_BIN} ($("${CMAKE_BIN}" --version | head -1))"
echo "==> CC/CXX: ${CC} / ${CXX}"

COMMON_CMAKE_ARGS=(
  -G Ninja
  -DCMAKE_BUILD_TYPE=RelWithDebInfo
  -DCMAKE_C_COMPILER="${CC}"
  -DCMAKE_CXX_COMPILER="${CXX}"
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON
  -DCMAKE_INSTALL_LIBDIR=lib
  # Apps live in bin/; libs in x86_64/. Build-tree apps still need an absolute
  # rpath so the linker can resolve libproj.so.25 while linking against libgdal.
  -DCMAKE_INSTALL_RPATH="\$ORIGIN/../${ABI}"
  -DCMAKE_BUILD_RPATH="$(abs_path "${LIB_DIR}")"
  -DCMAKE_BUILD_WITH_INSTALL_RPATH=OFF
  # Ship only libgdal.so (no libgdal.so.36); force apps to DT_NEEDED that name.
  # Handled by the GDAL_UNVERSIONED_SONAME option in gdal.cmake.
  -DGDAL_UNVERSIONED_SONAME=ON
  -DBUILD_APPS=ON
  -DBUILD_TESTING=OFF
  -DBUILD_PYTHON_BINDINGS=OFF
  -DBUILD_JAVA_BINDINGS=OFF
  -DBUILD_CSHARP_BINDINGS=OFF
  -DENABLE_GNM=OFF
  -DGDAL_BUILD_OPTIONAL_DRIVERS=OFF
  -DOGR_BUILD_OPTIONAL_DRIVERS=OFF
  -DGDAL_USE_EXTERNAL_LIBS=OFF
  -DGDAL_USE_ZLIB_INTERNAL=ON
  -DGDAL_USE_JSONC_INTERNAL=ON
  -DGDAL_USE_QHULL=OFF
  -DGDAL_USE_QHULL_INTERNAL=ON
  -DGDAL_USE_CURL=OFF
  -DGDAL_USE_GEOS=OFF
  -DGDAL_USE_EXPAT=OFF
  -DGDAL_USE_ICONV=OFF
  -DGDAL_USE_LIBXML2=OFF
  -DGDAL_USE_OPENSSL=OFF
  -DGDAL_USE_PNG=OFF
  -DGDAL_USE_JPEG=OFF
  -DGDAL_USE_GIF=OFF
  -DGDAL_USE_TIFF=OFF
  -DGDAL_USE_TIFF_INTERNAL=OFF
  -DGDAL_USE_GEOTIFF=OFF
  -DGDAL_USE_GEOTIFF_INTERNAL=OFF
  -DGDAL_ENABLE_DRIVER_GTIFF=OFF
  -DGDAL_USE_SPATIALITE=OFF
  -DGDAL_USE_RASTERLITE2=OFF
  -DGDAL_USE_PCRE=OFF
  -DGDAL_USE_PCRE2=OFF
  -DGDAL_USE_FREEXL=OFF
  -DGDAL_USE_LIBKML=OFF
  -DGDAL_USE_XERCESC=OFF
  -DGDAL_USE_ZSTD=OFF
  -DGDAL_USE_LIBLZMA=OFF
  -DGDAL_USE_LZ4=OFF
  -DGDAL_USE_DEFLATE=OFF
  -DGDAL_USE_BLOSC=OFF
  -DGDAL_USE_WEBP=OFF
  -DGDAL_USE_OPENJPEG=OFF
  -DGDAL_USE_ARMADILLO=OFF
  -DGDAL_USE_POPPLER=OFF
  -DGDAL_USE_CFITSIO=OFF
  -DGDAL_USE_NETCDF=OFF
  -DGDAL_USE_HDF5=OFF
  -DGDAL_USE_HDF4=OFF
  -DGDAL_USE_POSTGRESQL=OFF
  -DGDAL_USE_MYSQL=OFF
  -DGDAL_USE_ODBC=OFF
  -DGDAL_USE_SQLITE3=ON
  -DOGR_ENABLE_DRIVER_SQLITE=ON
  -DOGR_ENABLE_DRIVER_GPKG=ON
  -DGDAL_FIND_PACKAGE_PROJ_MODE=MODULE
  -DPROJ_INCLUDE_DIR="${INC_ABS}"
  -DSQLite3_INCLUDE_DIR="${INC_ABS}"
  -DCMAKE_FIND_USE_PACKAGE_REGISTRY=OFF
  -DCMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY=OFF
)

configure_and_build() {
  local kind="$1"
  local build_dir="$2"
  local stage_dir="$3"
  local proj_lib="$4"
  local sqlite_lib="$5"
  local shared_flag="$6"

  rm -rf "${stage_dir}"
  mkdir -p "${build_dir}" "${stage_dir}"

  if (( SKIP_CONFIGURE )); then
    echo "==> Skipping configure GDAL (${kind}) (--skip-configure)"
    [[ -f "${build_dir}/CMakeCache.txt" ]] || \
      die "build dir not configured: ${build_dir} (run without --skip-configure first)"
  else
    echo "==> Configure GDAL (${kind})"

    local extra_pic=()
    if [[ "${shared_flag}" == "OFF" ]]; then
      extra_pic+=(-DGDAL_OBJECT_LIBRARIES_POSITION_INDEPENDENT_CODE=ON)
    fi

    env PKG_CONFIG_LIBDIR=/dev/null PKG_CONFIG_PATH= \
    "${CMAKE_BIN}" -S "${GDAL_SRC}" -B "${build_dir}" \
      "${COMMON_CMAKE_ARGS[@]}" \
      "${extra_pic[@]}" \
      -DBUILD_SHARED_LIBS="${shared_flag}" \
      -DPROJ_LIBRARY="${proj_lib}" \
      -DSQLite3_LIBRARY="${sqlite_lib}" \
      -DCMAKE_INSTALL_PREFIX="$(abs_path "${stage_dir}")"
  fi

  echo "==> Build GDAL (${kind}) -j${JOBS}"
  "${CMAKE_BIN}" --build "${build_dir}" --parallel "${JOBS}"

  echo "==> Install GDAL (${kind}) -> ${stage_dir}"
  "${CMAKE_BIN}" --install "${build_dir}"
}

configure_and_build static \
  "${BUILD_STATIC}" "${STAGE_STATIC}" \
  "${PROJ_STATIC_ABS}" "${SQLITE_STATIC_ABS}" OFF

configure_and_build shared \
  "${BUILD_SHARED}" "${STAGE_SHARED}" \
  "${PROJ_SHARED_ABS}" "${SQLITE_SHARED_ABS}" ON

echo "==> Collecting libraries into ${LIB_DIR}"
STATIC_SRC="$(find "${STAGE_STATIC}" -name 'libgdal.a' | head -n1)"
[[ -n "${STATIC_SRC}" && -f "${STATIC_SRC}" ]] || die "libgdal.a not found after static install"

SHARED_SRC=""
if [[ -f "${STAGE_SHARED}/lib/libgdal.so" ]]; then
  SHARED_SRC="${STAGE_SHARED}/lib/libgdal.so"
elif [[ -f "${STAGE_SHARED}/lib64/libgdal.so" ]]; then
  SHARED_SRC="${STAGE_SHARED}/lib64/libgdal.so"
else
  SHARED_SRC="$(find "${STAGE_SHARED}" \( -name 'libgdal.so' -o -name 'libgdal.so.*' \) | head -n1 || true)"
fi
[[ -n "${SHARED_SRC}" && -f "${SHARED_SRC}" ]] || die "libgdal.so not found after shared install"
SHARED_REAL="$(readlink -f "${SHARED_SRC}")"

cp -f "${STATIC_SRC}" "${LIB_DIR}/libgdal.a"
cp -f "${SHARED_REAL}" "${LIB_DIR}/libgdal.so"

echo "==> Collecting headers into ${INC_OUT}"
STAGE_INC=""
if [[ -d "${STAGE_SHARED}/include" ]]; then
  STAGE_INC="${STAGE_SHARED}/include"
elif [[ -d "${STAGE_STATIC}/include" ]]; then
  STAGE_INC="${STAGE_STATIC}/include"
else
  die "GDAL headers not found in install prefix"
fi

: > "${HEADER_MANIFEST}"
(
  cd "${STAGE_INC}"
  find . -mindepth 1 \( -type f -o -type l -o -type d \) | sed 's|^\./||' | sort
) > "${HEADER_MANIFEST}"
cp -a "${STAGE_INC}/." "${INC_OUT}/"

echo "==> Collecting gdal data into ${SHARE_OUT}"
DATA_SRC=""
if [[ -d "${STAGE_SHARED}/share/gdal" ]]; then
  DATA_SRC="${STAGE_SHARED}/share/gdal"
elif [[ -d "${STAGE_STATIC}/share/gdal" ]]; then
  DATA_SRC="${STAGE_STATIC}/share/gdal"
fi
[[ -n "${DATA_SRC}" ]] || die "share/gdal not found after install"
mkdir -p "${SHARE_OUT}"
cp -a "${DATA_SRC}/." "${SHARE_OUT}/"

echo "==> Collecting apps into ${BIN_OUT}"
: > "${BIN_MANIFEST}"
STAGE_BIN=""
if [[ -d "${STAGE_SHARED}/bin" ]]; then
  STAGE_BIN="${STAGE_SHARED}/bin"
elif [[ -d "${STAGE_STATIC}/bin" ]]; then
  STAGE_BIN="${STAGE_STATIC}/bin"
fi
if [[ -n "${STAGE_BIN}" ]]; then
  local_name=""
  for local_name in "${STAGE_BIN}"/*; do
    [[ -f "${local_name}" && -x "${local_name}" ]] || continue
    base="$(basename "${local_name}")"
    # Avoid overwriting sqlite3 / proj tools if names collide (unlikely for gdal_*)
    cp -f "${local_name}" "${BIN_OUT}/${base}"
    echo "${base}" >> "${BIN_MANIFEST}"
    "${STRIP}" --strip-unneeded "${BIN_OUT}/${base}" 2>/dev/null || true
  done
fi

echo "==> Separating debug symbols and stripping"
cp -f "${LIB_DIR}/libgdal.a" "${LIB_DIR}/libgdal.a.debug"
"${STRIP}" --strip-unneeded "${LIB_DIR}/libgdal.a"
chmod -x "${LIB_DIR}/libgdal.a.debug" 2>/dev/null || true
separate_debug_and_strip "${LIB_DIR}/libgdal.so" "${LIB_DIR}/libgdal.so.debug"

echo "==> Done"
ls -la "${LIB_DIR}/libgdal.a" "${LIB_DIR}/libgdal.so"
ls -la "${INC_OUT}/gdal.h" 2>/dev/null || ls "${INC_OUT}"/gdal*.h | head
ls -la "${BIN_OUT}" | head
file "${LIB_DIR}/libgdal.a" "${LIB_DIR}/libgdal.so"
