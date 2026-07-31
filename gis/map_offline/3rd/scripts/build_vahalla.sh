#!/usr/bin/env bash
# Build Valhalla 3.8.3 for Linux x86_64 (data tools + shared library, -fPIC,
# stripped with separate debug info).
#
# Produces valhalla_build_tiles and other data tools for generating .gph tiles.
#
# Prerequisites (Ubuntu 22.04):
#   sudo apt install -y build-essential cmake ninja-build pkg-config python3 \
#     zlib1g-dev libboost-all-dev libprotobuf-dev protobuf-compiler \
#     libsqlite3-dev libspatialite-dev spatialite-bin \
#     libgeos-dev libluajit-5.1-dev libssl-dev
#
# Usage:
#   cd scripts
#   bash build_vahalla.sh
#   bash build_vahalla.sh clean
#   bash build_vahalla.sh -j 8
#
# Output (under ./output):
#   valhalla/bin/valhalla_build_tiles   (+ other data tools, .debug alongside)
#   valhalla/lib/libvalhalla.so*        (+.debug)
#   valhalla/include/valhalla/
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

VALHALLA_SRC="${ROOT}/valhalla-3.8.3"
OUT_DIR="${SCRIPT_DIR}/output"
VALHALLA_OUT="${OUT_DIR}/valhalla"
BUILD_DIR="${OUT_DIR}/build/valhalla-host"
STAGE_DIR="${OUT_DIR}/build/stage-valhalla-host"

# We build protobuf 21.12 from source because the system protobuf (3.12.4 on
# Ubuntu 22.04) is too old: Valhalla 3.8.3 uses has_*() APIs on oneof fields
# that older protoc doesn't generate.
PROTOBUF_VER="21.12"
PROTOBUF_SRC="${OUT_DIR}/build/protobuf-${PROTOBUF_VER}"
PROTOBUF_STAGE="${OUT_DIR}/build/stage-protobuf"
PROTOBUF_INSTALL="${OUT_DIR}/protobuf"

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

  clean             Remove Valhalla host artifacts under ${OUT_DIR} and exit
  -j N, --jobs N    Max parallel jobs (default: ${DEFAULT_JOBS}, env JOBS)
  --skip-clean      Skip do_clean (preserve build dirs for incremental build)
  --skip-configure  Skip cmake configure, build + install from existing build dir
  --build-only      Shorthand for --skip-clean --skip-configure
  (default)         Build Valhalla data tools + shared library

Prerequisites (Ubuntu 22.04):
  sudo apt install -y build-essential cmake ninja-build pkg-config python3 \\
    zlib1g-dev libboost-all-dev libprotobuf-dev protobuf-compiler \\
    libsqlite3-dev libspatialite-dev spatialite-bin \\
    libgeos-dev libluajit-5.1-dev libssl-dev
EOF
}

do_clean() {
  echo "==> Cleaning Valhalla host artifacts under ${OUT_DIR}"
  rm -rf "${BUILD_DIR}" "${STAGE_DIR}" "${VALHALLA_OUT}"
  # Note: PROTOBUF_INSTALL and PROTOBUF_SRC are kept (protobuf is built before clean)
}

# ---------------------------------------------------------------------------
# Strip an ELF file and save its debug symbols to a .debug sidecar.
# ---------------------------------------------------------------------------
strip_elf() {
  local bin="$1"
  local dbg="${bin}.debug"
  [[ -f "${bin}" ]] || return 0
  file "${bin}" | grep -q "ELF" || return 0
  if [[ -f "${dbg}" ]]; then
    return 0
  fi
  echo "  strip: $(basename "${bin}")"
  "${OBJCOPY}" --only-keep-debug "${bin}" "${dbg}"
  "${STRIP}" --strip-unneeded "${bin}"
  (
    cd "$(dirname "${bin}")"
    "${OBJCOPY}" --add-gnu-debuglink="$(basename "${dbg}")" "$(basename "${bin}")"
  )
  chmod -x "${dbg}" 2>/dev/null || true
}

# ---------------------------------------------------------------------------
# Verify all system-level dependencies are present.
# ---------------------------------------------------------------------------
check_deps() {
  echo "==> Checking system dependencies"

  local missing=()

  # Build tools
  for tool in "${CC}" "${CXX}" cmake ninja strip objcopy pkg-config \
              spatialite_tool spatialite python3; do
    if ! command -v "${tool}" >/dev/null 2>&1; then
      missing+=("${tool}")
    fi
  done

  # pkg-config modules
  for mod in zlib sqlite3 spatialite geos luajit openssl; do
    if ! pkg-config --exists "${mod}" 2>/dev/null; then
      missing+=("pkg-config:${mod}")
    fi
  done

  # Boost headers
  if [[ ! -d "/usr/include/boost" ]]; then
    missing+=("boost-headers")
  fi

  if (( ${#missing[@]} > 0 )); then
    echo "Missing dependencies:" >&2
    for m in "${missing[@]}"; do
      echo "  - ${m}" >&2
    done
    echo "" >&2
    echo "Install with:" >&2
    echo "  sudo apt install -y build-essential cmake ninja-build pkg-config python3 \\" >&2
    echo "    zlib1g-dev libboost-all-dev libprotobuf-dev protobuf-compiler \\" >&2
    echo "    libsqlite3-dev libspatialite-dev spatialite-bin \\" >&2
    echo "    libgeos-dev libluajit-5.1-dev libssl-dev" >&2
    exit 1
  fi

  echo "  All dependencies satisfied"
}

# ---------------------------------------------------------------------------
# Populate empty third_party submodules (tarball extractions don't include them)
# ---------------------------------------------------------------------------
ensure_third_party() {
  echo "==> Checking third_party submodules"
  local tp="${VALHALLA_SRC}/third_party"

  # Git submodules (header-only or data)
  declare -A repos=(
    ["date"]="https://github.com/HowardHinnant/date.git"
    ["rapidjson"]="https://github.com/miloyip/rapidjson.git"
    ["unordered_dense"]="https://github.com/martinus/unordered_dense.git"
    ["cxxopts"]="https://github.com/jarro2783/cxxopts.git"
    ["libosmium"]="https://github.com/osmcode/libosmium.git"
    ["protozero"]="https://github.com/mapbox/protozero.git"
    ["vtzero"]="https://github.com/mapbox/vtzero.git"
    ["cpp-statsd-client"]="https://github.com/vthiery/cpp-statsd-client.git"
    ["just_gtfs"]="https://github.com/valhalla/just_gtfs.git"
    ["microtar"]="https://github.com/rxi/microtar.git"
    ["dirent"]="https://github.com/tronkko/dirent.git"
  )

  for name in "${!repos[@]}"; do
    if [[ -d "${tp}/${name}" && -n "$(ls -A "${tp}/${name}" 2>/dev/null | head -1)" ]]; then
      continue
    fi
    echo "  Cloning ${name}..."
    rm -rf "${tp}/${name}"
    git clone --depth 1 "${repos[${name}]}" "${tp}/${name}" 2>&1 | tail -1
  done

  # tz data files (only need the data files, not the full git history)
  local tz_dir="${tp}/tz"
  local tz_files=(leapseconds.awk leap-seconds.list Makefile africa antarctica asia australasia backward etcetera europe northamerica southamerica)
  local need_tz=0
  for f in "${tz_files[@]}"; do
    if [[ ! -f "${tz_dir}/${f}" ]]; then
      need_tz=1
      break
    fi
  done
  if (( need_tz )); then
    mkdir -p "${tz_dir}"
    for f in "${tz_files[@]}"; do
      if [[ ! -f "${tz_dir}/${f}" ]]; then
        echo "  Downloading tz/${f}..."
        wget -q "https://raw.githubusercontent.com/eggert/tz/main/${f}" -O "${tz_dir}/${f}" 2>/dev/null || \
          curl -sL "https://raw.githubusercontent.com/eggert/tz/main/${f}" -o "${tz_dir}/${f}"
      fi
    done
  fi

  # Pre-generate leapseconds (avoids needing make in the tz directory)
  if [[ ! -f "${tz_dir}/leapseconds" && -f "${tz_dir}/leapseconds.awk" && -f "${tz_dir}/leap-seconds.list" ]]; then
    echo "  Generating tz/leapseconds..."
    awk -f "${tz_dir}/leapseconds.awk" "${tz_dir}/leap-seconds.list" > "${tz_dir}/leapseconds"
  fi

  echo "  Third_party submodules ready"
}

# ---------------------------------------------------------------------------
# Build Protobuf 21.12 from source (system 3.12.4 is too old for Valhalla 3.8.3)
# ---------------------------------------------------------------------------
build_protobuf() {
  echo "========== Building Protobuf ${PROTOBUF_VER} =========="

  if [[ -x "${PROTOBUF_INSTALL}/bin/protoc" && -f "${PROTOBUF_INSTALL}/lib/libprotobuf-lite.so" ]]; then
    echo "  Protobuf already built, skipping"
    return 0
  fi

  # Download (tarball is faster than git clone)
  if [[ ! -d "${PROTOBUF_SRC}" ]]; then
    mkdir -p "$(dirname "${PROTOBUF_SRC}")"
    echo "  Downloading protobuf ${PROTOBUF_VER}..."
    local archive="$(dirname "${PROTOBUF_SRC}")/protobuf.tar.gz"
    curl -L -o "${archive}" "https://github.com/protocolbuffers/protobuf/archive/refs/tags/v${PROTOBUF_VER}.tar.gz"
    tar xzf "${archive}" -C "$(dirname "${PROTOBUF_SRC}")"
    rm -f "${archive}"
    # GitHub archive extracts to protobuf-v21.12, normalize to protobuf-21.12
    mv "$(dirname "${PROTOBUF_SRC}")/protobuf-v${PROTOBUF_VER}" "${PROTOBUF_SRC}" 2>/dev/null || true
  fi

  # Build
  local pb_build="${PROTOBUF_SRC}/cmake-build"
  rm -rf "${pb_build}" "${PROTOBUF_INSTALL}"
  mkdir -p "${pb_build}"

  "${CMAKE_BIN}" -G Ninja -S "${PROTOBUF_SRC}" -B "${pb_build}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER="${CC}" \
    -DCMAKE_CXX_COMPILER="${CXX}" \
    -DCMAKE_C_FLAGS="-fPIC -O2" \
    -DCMAKE_CXX_FLAGS="-fPIC -O2" \
    -DCMAKE_INSTALL_PREFIX="${PROTOBUF_INSTALL}" \
    -DCMAKE_INSTALL_LIBDIR=lib \
    -Dprotobuf_BUILD_TESTS=OFF \
    -Dprotobuf_BUILD_SHARED_LIBS=ON \
    -Dprotobuf_BUILD_PROTOC_BINARIES=ON

  "${CMAKE_BIN}" --build "${pb_build}" --parallel "${JOBS}"
  "${CMAKE_BIN}" --install "${pb_build}"

  echo "  Protobuf ${PROTOBUF_VER} built: ${PROTOBUF_INSTALL}"
}

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------
[[ -x "${STRIP}" ]]     || die "strip not found"
[[ -x "${OBJCOPY}" ]]   || die "objcopy not found"
[[ -x "${CMAKE_BIN}" ]] || die "cmake not found (set CMAKE_BIN)"
[[ -x "${NINJA_BIN}" ]] || die "ninja not found (set NINJA_BIN)"
[[ -d "${VALHALLA_SRC}" ]] || die "Valhalla source not found: ${VALHALLA_SRC}"
[[ -f "${VALHALLA_SRC}/CMakeLists.txt" ]] || die "CMakeLists.txt missing in ${VALHALLA_SRC}"

check_deps
ensure_third_party
build_protobuf

# ---------------------------------------------------------------------------
# Prepare directories
# ---------------------------------------------------------------------------
if (( ! SKIP_CLEAN )); then
  do_clean
fi
mkdir -p "${BUILD_DIR}" "${STAGE_DIR}"

echo "==> ABI:      ${ABI}  JOBS: ${JOBS}"
echo "==> Source:   ${VALHALLA_SRC}"
echo "==> Build:    ${BUILD_DIR}"
echo "==> Stage:    ${STAGE_DIR}"
echo "==> Output:   ${VALHALLA_OUT}"
echo "==> CC/CXX:   ${CC} / ${CXX}"
echo "==> cmake:    ${CMAKE_BIN} ($("${CMAKE_BIN}" --version | head -1))"

# ---------------------------------------------------------------------------
# Configure
#
# Key CMake options:
#   ENABLE_DATA_TOOLS=ON   Build valhalla_build_tiles and friends
#   ENABLE_TOOLS=OFF       Skip valhalla_service (not needed for tile building)
#   ENABLE_SERVICES=OFF    Skip prime_server workers
#   ENABLE_HTTP=OFF        No curl needed for offline tile building
#   ENABLE_LZ4=OFF         Optional compression; zlib is sufficient
#   ENABLE_GEOTIFF=OFF     Only for isochrone GeoTIFF output
#   ENABLE_PYTHON_BINDINGS=OFF
#   ENABLE_TESTS=OFF
#   BUILD_SHARED_LIBS=ON   Produce libvalhalla.so
#
# CMAKE_INSTALL_RPATH='$ORIGIN/../lib' lets binaries in bin/ find
# libvalhalla.so in lib/ at runtime without LD_LIBRARY_PATH.
# ---------------------------------------------------------------------------
if (( SKIP_CONFIGURE )); then
  echo "==> Skipping configure Valhalla (--skip-configure)"
  [[ -f "${BUILD_DIR}/CMakeCache.txt" ]] || \
    die "build dir not configured: ${BUILD_DIR} (run without --skip-configure first)"
else
  echo "==> Configuring Valhalla"

  # Point CMake and pkg-config at our custom protobuf 21.12 install
  export PATH="${PROTOBUF_INSTALL}/bin:${PATH}"
  export PKG_CONFIG_PATH="${PROTOBUF_INSTALL}/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
  export CMAKE_PREFIX_PATH="${PROTOBUF_INSTALL}"

  "${CMAKE_BIN}" -G Ninja -S "${VALHALLA_SRC}" -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_C_COMPILER="${CC}" \
    -DCMAKE_CXX_COMPILER="${CXX}" \
    -DCMAKE_C_FLAGS="-fPIC -g" \
    -DCMAKE_CXX_FLAGS="-fPIC -g" \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DCMAKE_INSTALL_PREFIX="$(cd "${STAGE_DIR}" && pwd)" \
    -DCMAKE_INSTALL_RPATH='$ORIGIN/../lib' \
    -DCMAKE_INSTALL_LIBDIR=lib \
    -DBUILD_SHARED_LIBS=ON \
    -DENABLE_DATA_TOOLS=ON \
    -DENABLE_TOOLS=OFF \
    -DENABLE_SERVICES=OFF \
    -DENABLE_HTTP=OFF \
    -DENABLE_LZ4=OFF \
    -DENABLE_GEOTIFF=OFF \
    -DENABLE_PYTHON_BINDINGS=OFF \
    -DENABLE_TESTS=OFF \
    -DENABLE_CCACHE=OFF \
    -DENABLE_COMPILER_WARNINGS=OFF \
    -DENABLE_SINGLE_FILES_WERROR=OFF \
    -DProtobuf_DIR="${PROTOBUF_INSTALL}/lib/cmake/protobuf" \
    -DProtobuf_PROTOC_EXECUTABLE="${PROTOBUF_INSTALL}/bin/protoc" \
    -DPROTOBUF_INCLUDE_DIR="${PROTOBUF_INSTALL}/include" \
    -DProtobuf_INCLUDE_DIRS="${PROTOBUF_INSTALL}/include"
fi

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
echo "==> Building Valhalla (-j${JOBS})"
"${CMAKE_BIN}" --build "${BUILD_DIR}" --parallel "${JOBS}"

# ---------------------------------------------------------------------------
# Install to staging
# ---------------------------------------------------------------------------
echo "==> Installing to staging: ${STAGE_DIR}"
"${CMAKE_BIN}" --install "${BUILD_DIR}"

# Copy protobuf shared libraries so the installation is self-contained
echo "==> Bundling protobuf runtime libraries"
mkdir -p "${STAGE_DIR}/lib"
cp -a "${PROTOBUF_INSTALL}/lib/"libprotobuf-lite.so* "${STAGE_DIR}/lib/" 2>/dev/null || true
cp -a "${PROTOBUF_INSTALL}/lib/"libprotobuf.so* "${STAGE_DIR}/lib/" 2>/dev/null || true

# ---------------------------------------------------------------------------
# Strip & separate debug info for all ELF files
# ---------------------------------------------------------------------------
echo "==> Stripping ELF binaries and separating debug info"

# Shared libraries in lib/
if [[ -d "${STAGE_DIR}/lib" ]]; then
  while IFS= read -r -d '' f; do
    strip_elf "${f}"
  done < <(find "${STAGE_DIR}/lib" -type f -print0)
fi

# Executables in bin/
if [[ -d "${STAGE_DIR}/bin" ]]; then
  while IFS= read -r -d '' f; do
    strip_elf "${f}"
  done < <(find "${STAGE_DIR}/bin" -type f -print0)
fi

# ---------------------------------------------------------------------------
# Copy staging to final output
# ---------------------------------------------------------------------------
echo "==> Copying to output: ${VALHALLA_OUT}"
rm -rf "${VALHALLA_OUT}"
cp -a "${STAGE_DIR}" "${VALHALLA_OUT}"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "================================================"
echo "  Valhalla build complete!"
echo "================================================"
echo ""
echo "  Install prefix: ${VALHALLA_OUT}"
echo ""
echo "  Data tools (bin/):"

# List valhalla binaries (ELF files only, skip Python scripts)
for f in "${VALHALLA_OUT}/bin/"valhalla_*; do
  [[ -f "${f}" ]] || continue
  if file "${f}" | grep -q "ELF"; then
    echo "    $(basename "${f}")  (ELF binary)"
  else
    echo "    $(basename "${f}")  (script)"
  fi
done

echo ""
echo "  Library (lib/):"
ls -la "${VALHALLA_OUT}/lib/"libvalhalla* 2>/dev/null | sed 's/^/    /'
echo ""
echo "  Debug symbols (.debug sidecars alongside each ELF file)"
echo ""
echo "  Usage:"
echo "    export PATH=${VALHALLA_OUT}/bin:\$PATH"
echo "    export LD_LIBRARY_PATH=${VALHALLA_OUT}/lib:\$LD_LIBRARY_PATH"
echo ""
echo "    # 1. Generate config"
echo "    valhalla_build_config --mjolnir-tile-dir /data/tiles --mjolnir-tile-extract /data/tiles.tar > valhalla.json"
echo ""
echo "    # 2. Build admin/timezone databases"
echo "    valhalla_build_admins -c valhalla.json"
echo "    valhalla_build_timezones -c valhalla.json"
echo ""
echo "    # 3. Build tiles from OSM PBF"
echo "    valhalla_build_tiles -c valhalla.json input.osm.pbf"
echo ""
echo "    # 4. Pack tiles into tar (for mmap loading on Android)"
echo "    valhalla_build_extract -c valhalla.json"
echo ""
