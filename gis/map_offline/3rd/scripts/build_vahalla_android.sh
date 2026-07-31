#!/usr/bin/env bash
# Build Valhalla 3.8.3 .so for Android (JNI routing/navigation).
#
# Produces libvalhalla.so and runtime dependencies for Android, suitable
# for loading via System.loadLibrary() from Java/Kotlin JNI code.
#
# Prerequisites:
#   1. Run build_vahalla.sh first (provides host protoc + third_party submodules)
#   2. Android NDK at NDK_PATH (default: /mnt/d/devd/android/sdk-linux/ndk/30.0.15729638)
#
# Usage:
#   cd scripts
#   bash build_vahalla_android.sh                      # arm64-v8a (default)
#   bash build_vahalla_android.sh --abi armeabi-v7a     # 32-bit ARM
#   bash build_vahalla_android.sh --abi x86_64          # x86_64 (emulator)
#   bash build_vahalla_android.sh clean
#   bash build_vahalla_android.sh -j 8
#
# Output:
#   output/valhalla-android/<abi>/
#     libvalhalla.so          (stripped, -fPIC, RPATH=$ORIGIN)
#     libvalhalla.so.debug    (debug symbols)
#     libprotobuf-lite.so     (stripped)
#     libprotobuf-lite.so.debug
#     libc++_shared.so        (from NDK)
set -euo pipefail

# ======================== Config ========================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

NDK_PATH="${NDK_PATH:-/mnt/d/devd/android/sdk-linux/ndk/30.0.15729638}"
ANDROID_ABI="${ANDROID_ABI:-arm64-v8a}"
ANDROID_API="${ANDROID_API:-21}"
ANDROID_STL="c++_shared"

VALHALLA_SRC="${ROOT}/valhalla-3.8.3"
OUT_DIR="${SCRIPT_DIR}/output"
PC_PROTOBUF="${OUT_DIR}/protobuf"

PROTOBUF_VER="21.12"
PROTOBUF_SRC="${OUT_DIR}/build/protobuf-${PROTOBUF_VER}"
BOOST_INCLUDEDIR="${BOOST_INCLUDEDIR:-/usr/local/boost-1.89.0/include}"

if command -v nproc >/dev/null 2>&1; then
  DEFAULT_JOBS="$(nproc)"
else
  DEFAULT_JOBS=4
fi
JOBS="${JOBS:-${DEFAULT_JOBS}}"
SKIP_CLEAN=0
SKIP_CONFIGURE=0

CMAKE_BIN="${CMAKE_BIN:-$(command -v cmake || true)}"
NINJA_BIN="${NINJA_BIN:-$(command -v ninja || true)}"

# NDK tool paths (set in setup_paths after ABI is known)
NDK_TOOLCHAIN=""
NDK_PREBUILT=""
NDK_TOOL_BIN=""
NDK_SYSROOT=""
NDK_LIB_DIR=""
STRIP=""
OBJCOPY=""

# Build paths (set in setup_paths)
BUILD_ROOT=""
STAGE_DIR=""
XCOMPILE_PREFIX=""
ABI_OUT=""

# ======================== Functions ========================

die() {
  echo "error: $*" >&2
  exit 1
}

usage() {
  cat <<EOF
Usage: $(basename "$0") [OPTIONS]

  --abi ABI         Target ABI: arm64-v8a (default), armeabi-v7a, x86_64
  --api LEVEL       Android API level (default: 21)
  -j N, --jobs N    Parallel jobs (default: ${DEFAULT_JOBS})
  --skip-clean      Skip do_clean (preserve build dirs for incremental build)
  --skip-configure  Skip cmake configure, build + install from existing build dir
  --build-only      Shorthand for --skip-clean --skip-configure
  clean             Remove Android build artifacts for current ABI
  -h, --help        Show this help

Prerequisites:
  1. Run build_vahalla.sh first (provides host protoc + third_party)
  2. NDK at: ${NDK_PATH}

Environment:
  NDK_PATH         Override NDK location
  ANDROID_ABI      Override target ABI
  ANDROID_API      Override API level
  BOOST_INCLUDEDIR Override Boost headers path
EOF
}

setup_paths() {
  # ABI -> NDK sysroot lib directory
  case "$ANDROID_ABI" in
    arm64-v8a)    NDK_LIB_DIR="aarch64-linux-android" ;;
    armeabi-v7a)  NDK_LIB_DIR="arm-linux-androideabi" ;;
    x86_64)       NDK_LIB_DIR="x86_64-linux-android" ;;
    *) die "Unsupported ABI: $ANDROID_ABI" ;;
  esac

  NDK_TOOLCHAIN="$NDK_PATH/build/cmake/android.toolchain.cmake"
  NDK_PREBUILT="$NDK_PATH/toolchains/llvm/prebuilt/linux-x86_64"
  NDK_TOOL_BIN="$NDK_PREBUILT/bin"
  NDK_SYSROOT="$NDK_PREBUILT/sysroot"
  STRIP="$NDK_TOOL_BIN/llvm-strip"
  OBJCOPY="$NDK_TOOL_BIN/llvm-objcopy"

  BUILD_ROOT="${OUT_DIR}/build/valhalla-android-${ANDROID_ABI}"
  STAGE_DIR="${BUILD_ROOT}/stage"
  # Cross-compile prefix is separate from BUILD_ROOT so do_clean preserves it
  XCOMPILE_PREFIX="${OUT_DIR}/build/xcompile-${ANDROID_ABI}"
  ABI_OUT="${ANDROID_OUT:-${OUT_DIR}/valhalla-android}/${ANDROID_ABI}"
}

do_clean() {
  echo "==> Cleaning Valhalla Android artifacts ($ANDROID_ABI)"
  rm -rf "$BUILD_ROOT" "$ABI_OUT"
  # Note: XCOMPILE_PREFIX (protobuf) is preserved for incremental rebuilds
}

# ---------------------------------------------------------------------------
# Strip an ELF file and save debug symbols to a .debug sidecar.
# Uses NDK's llvm-strip / llvm-objcopy.
# ---------------------------------------------------------------------------
strip_elf() {
  local bin="$1"
  local dbg="${bin}.debug"
  [[ -f "$bin" ]] || return 0
  file "$bin" | grep -q "ELF" || return 0
  if [[ -f "$dbg" ]]; then
    return 0
  fi
  echo "  strip: $(basename "$bin")"
  "$OBJCOPY" --only-keep-debug "$bin" "$dbg"
  "$STRIP" --strip-unneeded "$bin"
  (
    cd "$(dirname "$bin")"
    "$OBJCOPY" --add-gnu-debuglink="$(basename "$dbg")" "$(basename "$bin")"
  )
  chmod -x "$dbg" 2>/dev/null || true
}

# ======================== Checks ========================

check_prerequisites() {
  echo "==> Checking prerequisites"

  [[ -d "$NDK_PATH" ]] || die "NDK not found: $NDK_PATH"
  [[ -f "$NDK_TOOLCHAIN" ]] || die "NDK toolchain not found: $NDK_TOOLCHAIN"
  [[ -x "$STRIP" ]] || die "llvm-strip not found: $STRIP"
  [[ -x "$OBJCOPY" ]] || die "llvm-objcopy not found: $OBJCOPY"
  [[ -x "$CMAKE_BIN" ]] || die "cmake not found"
  [[ -x "$NINJA_BIN" ]] || die "ninja not found"
  [[ -d "$VALHALLA_SRC" ]] || die "Valhalla source not found: $VALHALLA_SRC"
  [[ -f "$VALHALLA_SRC/CMakeLists.txt" ]] || die "CMakeLists.txt missing"

  # Host protoc from PC build
  [[ -x "$PC_PROTOBUF/bin/protoc" ]] || \
    die "Host protoc not found: $PC_PROTOBUF/bin/protoc\n    Run build_vahalla.sh first"

  # third_party submodules
  [[ -f "$VALHALLA_SRC/third_party/date/include/date/date.h" ]] || \
    die "third_party not populated. Run build_vahalla.sh first"

  # Boost headers
  [[ -f "$BOOST_INCLUDEDIR/boost/version.hpp" ]] || \
    die "Boost headers not found: $BOOST_INCLUDEDIR"

  # Protobuf source (reused from PC build)
  [[ -d "$PROTOBUF_SRC" ]] || \
    die "Protobuf source not found: $PROTOBUF_SRC\n    Run build_vahalla.sh first"

  # libc++_shared.so in NDK
  [[ -f "$NDK_SYSROOT/usr/lib/$NDK_LIB_DIR/libc++_shared.so" ]] || \
    die "libc++_shared.so not found in NDK for $ANDROID_ABI"

  echo "  All prerequisites satisfied"
  echo "  ABI: $ANDROID_ABI  API: android-$ANDROID_API  STL: $ANDROID_STL"
  echo "  NDK: $NDK_PATH"
}

# ======================== Setup zlib.pc ========================
#
# The NDK provides zlib.h and libz.so in the sysroot, but no pkg-config
# file. Valhalla uses pkg_check_modules(ZLIB REQUIRED IMPORTED_TARGET zlib),
# so we create a zlib.pc that points to the NDK sysroot.
# ---------------------------------------------------------------------------

setup_zlib_pc() {
  echo "==> Creating zlib.pc for NDK sysroot"
  local pc_dir="$XCOMPILE_PREFIX/lib/pkgconfig"
  mkdir -p "$pc_dir"

  cat > "$pc_dir/zlib.pc" <<EOF
prefix=$NDK_SYSROOT/usr
exec_prefix=\${prefix}
libdir=\${exec_prefix}/lib/${NDK_LIB_DIR}/${ANDROID_API}
includedir=\${prefix}/include

Name: zlib
Description: zlib compression library
Version: 1.2.11
Libs: -L\${libdir} -lz
Cflags: -I\${includedir}
EOF
  echo "  $pc_dir/zlib.pc"
}

# ======================== Build protobuf-lite for Android ========================
#
# Cross-compiles protobuf 21.12 shared library for Android.
# Host protoc (from PC build) is used for .proto code generation;
# protobuf_BUILD_PROTOC_BINARIES=OFF skips building protoc for target.
# -----------------------------------------------------------------------------

build_protobuf_android() {
  echo "========== Building Protobuf ${PROTOBUF_VER} for Android ($ANDROID_ABI) =========="

  if [[ -f "$XCOMPILE_PREFIX/lib/libprotobuf-lite.so" ]]; then
    echo "  Protobuf already built, skipping"
    return 0
  fi

  local pb_build="$BUILD_ROOT/protobuf-build"
  rm -rf "$pb_build"
  mkdir -p "$pb_build" "$XCOMPILE_PREFIX"

  "$CMAKE_BIN" -G Ninja -S "$PROTOBUF_SRC" -B "$pb_build" \
    -DCMAKE_TOOLCHAIN_FILE="$NDK_TOOLCHAIN" \
    -DANDROID_ABI="$ANDROID_ABI" \
    -DANDROID_PLATFORM="android-$ANDROID_API" \
    -DANDROID_STL="$ANDROID_STL" \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_C_FLAGS="-fPIC -g" \
    -DCMAKE_CXX_FLAGS="-fPIC -g" \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DCMAKE_INSTALL_PREFIX="$XCOMPILE_PREFIX" \
    -DCMAKE_INSTALL_LIBDIR=lib \
    -Dprotobuf_BUILD_TESTS=OFF \
    -Dprotobuf_BUILD_SHARED_LIBS=ON \
    -Dprotobuf_BUILD_PROTOC_BINARIES=OFF \
    -Dprotobuf_MODULE_COMPATIBLE=ON \
    -Dprotobuf_WITH_ZLIB=OFF

  echo "==> Building protobuf (-j${JOBS})"
  "$CMAKE_BIN" --build "$pb_build" --parallel "$JOBS"

  echo "==> Installing protobuf"
  "$CMAKE_BIN" --install "$pb_build"

  # Remove cmake config to force FindProtobuf module mode.
  # The protobuf cmake config doesn't properly handle cross-compilation:
  # protobuf::protoc target has no IMPORTED_LOCATION when
  # protobuf_BUILD_PROTOC_BINARIES=OFF. FindProtobuf module uses
  # Protobuf_PROTOC_EXECUTABLE directly, which works for cross-compilation.
  rm -rf "$XCOMPILE_PREFIX/lib/cmake/protobuf"

  echo "  Protobuf ${PROTOBUF_VER} for Android built: $XCOMPILE_PREFIX"
}

# ======================== Build Valhalla for Android ========================
#
# Cross-compiles Valhalla as a shared library (libvalhalla.so) for Android.
# Only routing modules are built (no data tools, services, or tests).
#
# Key CMake options:
#   ENABLE_DATA_TOOLS=OFF   No tile building tools
#   ENABLE_TOOLS=OFF        No executables
#   ENABLE_SERVICES=OFF     No prime_server workers
#   ENABLE_HTTP=OFF         No curl (tiles loaded locally)
#   ENABLE_LZ4=OFF          Optional compression
#   ENABLE_GEOTIFF=OFF      No GeoTIFF output
#   BUILD_SHARED_LIBS=ON    Produce libvalhalla.so
#
# RPATH=$ORIGIN lets the .so find its dependencies in the same directory.
# -----------------------------------------------------------------------------

build_valhalla_android() {
  echo "========== Building Valhalla for Android ($ANDROID_ABI) =========="

  local vh_build="$BUILD_ROOT/valhalla-build"
  rm -rf "$STAGE_DIR"
  mkdir -p "$vh_build" "$STAGE_DIR"

  if (( SKIP_CONFIGURE )); then
    echo "==> Skipping configure Valhalla for Android (--skip-configure)"
    [[ -f "$vh_build/CMakeCache.txt" ]] || \
      die "build dir not configured: $vh_build (run without --skip-configure first)"
  else
    rm -rf "$vh_build"
    mkdir -p "$vh_build"

    # Environment for cross-compilation
    export PATH="$PC_PROTOBUF/bin:${PATH}"
    # PKG_CONFIG_LIBDIR prevents pkg-config from finding host x86_64 libraries
    export PKG_CONFIG_PATH="$XCOMPILE_PREFIX/lib/pkgconfig"
    export PKG_CONFIG_LIBDIR="$XCOMPILE_PREFIX/lib/pkgconfig"
    export CMAKE_PREFIX_PATH="$XCOMPILE_PREFIX"

    echo "==> Configuring Valhalla for Android"

    "$CMAKE_BIN" -G Ninja -S "$VALHALLA_SRC" -B "$vh_build" \
      -DCMAKE_TOOLCHAIN_FILE="$NDK_TOOLCHAIN" \
      -DANDROID_ABI="$ANDROID_ABI" \
      -DANDROID_PLATFORM="android-$ANDROID_API" \
      -DANDROID_STL="$ANDROID_STL" \
      -DCMAKE_BUILD_TYPE=RelWithDebInfo \
      -DCMAKE_C_FLAGS="-fPIC -g" \
      -DCMAKE_CXX_FLAGS="-fPIC -g" \
      -DCMAKE_SHARED_LINKER_FLAGS="-llog" \
      -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
      -DCMAKE_INSTALL_PREFIX="$(cd "$STAGE_DIR" && pwd)" \
      -DCMAKE_INSTALL_LIBDIR=lib \
      -DCMAKE_INSTALL_RPATH='$ORIGIN' \
      -DCMAKE_PREFIX_PATH="$XCOMPILE_PREFIX" \
      -DBUILD_SHARED_LIBS=ON \
      -DENABLE_DATA_TOOLS=OFF \
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
      -DProtobuf_PROTOC_EXECUTABLE="$PC_PROTOBUF/bin/protoc" \
      -DPROTOBUF_INCLUDE_DIR="$XCOMPILE_PREFIX/include" \
      -DProtobuf_INCLUDE_DIRS="$XCOMPILE_PREFIX/include" \
      -DProtobuf_LIBRARY="$XCOMPILE_PREFIX/lib/libprotobuf.so" \
      -DProtobuf_LITE_LIBRARY="$XCOMPILE_PREFIX/lib/libprotobuf-lite.so" \
      -DBoost_NO_BOOST_CMAKE=ON \
      -DBOOST_INCLUDEDIR="$BOOST_INCLUDEDIR" \
      -DBoost_INCLUDE_DIR="$BOOST_INCLUDEDIR"
  fi

  echo "==> Building Valhalla (-j${JOBS})"
  "$CMAKE_BIN" --build "$vh_build" --parallel "$JOBS"

  echo "==> Installing to staging"
  "$CMAKE_BIN" --install "$vh_build"
}

# ======================== Collect output ========================

collect_output() {
  echo "==> Collecting output to: $ABI_OUT"

  rm -rf "$ABI_OUT"
  mkdir -p "$ABI_OUT"

  # libvalhalla.so + versioned symlinks
  cp -a "$STAGE_DIR"/lib/libvalhalla.so* "$ABI_OUT/"

  # libprotobuf-lite.so + versioned symlinks (runtime dependency)
  cp -a "$XCOMPILE_PREFIX"/lib/libprotobuf-lite.so* "$ABI_OUT/"

  # libc++_shared.so from NDK (C++ STL runtime)
  local libcxx="$NDK_SYSROOT/usr/lib/$NDK_LIB_DIR/libc++_shared.so"
  cp -a "$libcxx" "$ABI_OUT/"

  # Strip & separate debug info for all .so files
  echo "==> Stripping and separating debug info"
  while IFS= read -r -d '' f; do
    strip_elf "$f"
  done < <(find "$ABI_OUT" -type f -name "*.so*" -print0)
}

# ======================== Summary ========================

print_summary() {
  echo ""
  echo "================================================"
  echo "  Valhalla Android build complete!"
  echo "================================================"
  echo ""
  echo "  ABI:     $ANDROID_ABI"
  echo "  API:     android-$ANDROID_API"
  echo "  Output:  $ABI_OUT"
  echo ""
  echo "  Libraries:"

  # List .so files (excluding .debug)
  for f in "$ABI_OUT"/*.so; do
    [[ -f "$f" ]] || continue
    local dbg="${f}.debug"
    if [[ -f "$dbg" ]]; then
      printf "    %-40s %s  (+debug)\n" "$(basename "$f")" "$(du -h "$f" | cut -f1)"
    else
      printf "    %-40s %s\n" "$(basename "$f")" "$(du -h "$f" | cut -f1)"
    fi
  done

  echo ""
  echo "  Debug symbols:"
  ls "$ABI_OUT"/*.debug 2>/dev/null | while read -r f; do
    printf "    %-40s %s\n" "$(basename "$f")" "$(du -h "$f" | cut -f1)"
  done

  echo ""
  echo "  Usage in Android project:"
  echo "    1. Copy all .so files to: app/src/main/jniLibs/$ANDROID_ABI/"
  echo "    2. Copy .debug files separately (not bundled in APK)"
  echo ""
  echo "  JNI entry point:"
  echo "    System.loadLibrary(\"valhalla\");"
  echo "    // Use valhalla::tyr::actor_t via JNI wrapper for routing"
  echo ""
}

# ======================== Main ========================

ONLY_CLEAN=0
while (($# > 0)); do
  case "$1" in
    clean)
      ONLY_CLEAN=1
      shift
      ;;
    --abi)
      (($# >= 2)) || die "$1 requires a value"
      ANDROID_ABI="$2"
      shift 2
      ;;
    --api)
      (($# >= 2)) || die "$1 requires a value"
      ANDROID_API="$2"
      shift 2
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

# Compute all paths based on final ABI/API values
setup_paths

if ((ONLY_CLEAN)); then
  do_clean
  echo "==> Clean done"
  exit 0
fi

check_prerequisites
if (( ! SKIP_CLEAN )); then
  do_clean
fi
setup_zlib_pc
build_protobuf_android
build_valhalla_android
collect_output
print_summary
