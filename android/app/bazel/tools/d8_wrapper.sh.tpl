#!/bin/bash
set -euo pipefail

# Resolve self and find runfiles
SELF="$0"
if [[ -L "$SELF" ]]; then
    SELF="$(readlink -f "$SELF")"
fi

# Find runfiles directory
RUNFILES=""
if [[ -n "${RUNFILES_DIR:-}" && -d "${RUNFILES_DIR}" ]]; then
    RUNFILES="$RUNFILES_DIR"
elif [[ -d "${SELF}.runfiles" ]]; then
    RUNFILES="${SELF}.runfiles"
fi

# The d8_compat_dx path (execroot-relative, baked in at build time)
D8_EXEC_PATH="@D8_COMPAT_DX@"

# Try exec path first (works when d8_compat_dx is a direct tool dep)
if [[ -x "$D8_EXEC_PATH" ]]; then
    D8="$D8_EXEC_PATH"
elif [[ -n "$RUNFILES" ]]; then
    # Fall back to runfiles lookup
    D8=""
    for candidate in \
        "$RUNFILES/androidsdk/d8_compat_dx" \
        "$RUNFILES/bazel_android_app/external/androidsdk/d8_compat_dx" \
        "$RUNFILES/_main/external/androidsdk/d8_compat_dx"; do
        if [[ -x "$candidate" ]]; then
            D8="$candidate"
            break
        fi
    done
    if [[ -z "$D8" ]]; then
        echo "ERROR: d8_wrapper: cannot locate d8_compat_dx in runfiles at $RUNFILES" >&2
        exit 1
    fi
else
    echo "ERROR: d8_wrapper: cannot locate d8_compat_dx (no runfiles, exec path $D8_EXEC_PATH not found)" >&2
    exit 1
fi

# Convert --min_sdk_version to --min-sdk-version
ARGS=()
for arg in "$@"; do
    case "$arg" in
        --min_sdk_version)
            ARGS+=("--min-sdk-version")
            ;;
        *)
            ARGS+=("$arg")
            ;;
    esac
done

exec "$D8" "${ARGS[@]}"
