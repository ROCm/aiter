#!/bin/bash

set -ex

TRITON_COMMIT=${TRITON_COMMIT:-756afc06}
TRITON_WHEEL_DIR=${TRITON_WHEEL_DIR:-}
TRITON_BUILD_WHEEL_DIR=${TRITON_BUILD_WHEEL_DIR:-}
TRITON_RETRY_ATTEMPTS=${TRITON_RETRY_ATTEMPTS:-3}
TRITON_RETRY_SLEEP_SECONDS=${TRITON_RETRY_SLEEP_SECONDS:-30}
PIP_DEFAULT_TIMEOUT=${PIP_DEFAULT_TIMEOUT:-120}
PIP_RETRIES=${PIP_RETRIES:-20}

retry_with_backoff() {
    local max_attempts="$1"
    local base_sleep="$2"
    shift 2

    local attempt=1
    while true; do
        if "$@"; then
            return 0
        fi

        if [[ "$attempt" -ge "$max_attempts" ]]; then
            return 1
        fi

        echo "Triton setup failed on attempt ${attempt}"
        sleep $((attempt * base_sleep))
        attempt=$((attempt + 1))
    done
}

install_triton_from_wheel() {
    pip uninstall -y triton || true
    pip install \
        --default-timeout "$PIP_DEFAULT_TIMEOUT" \
        --retries "$PIP_RETRIES" \
        "$TRITON_WHEEL_DIR"/*.whl
}

build_triton_from_source() {
    (
        rm -rf triton
        git clone https://github.com/triton-lang/triton
        cd triton
        git checkout "$TRITON_COMMIT"
        pip install \
            --default-timeout "$PIP_DEFAULT_TIMEOUT" \
            --retries "$PIP_RETRIES" \
            -r python/requirements.txt

        if [[ -n "$TRITON_BUILD_WHEEL_DIR" ]]; then
            mkdir -p "$TRITON_BUILD_WHEEL_DIR"
            MAX_JOBS=64 pip wheel \
                --default-timeout "$PIP_DEFAULT_TIMEOUT" \
                --retries "$PIP_RETRIES" \
                --no-deps \
                -w "$TRITON_BUILD_WHEEL_DIR" \
                .
        else
            pip uninstall -y triton || true
            MAX_JOBS=64 pip install \
                --default-timeout "$PIP_DEFAULT_TIMEOUT" \
                --retries "$PIP_RETRIES" \
                .
        fi
    )
}

if [[ -n "$TRITON_WHEEL_DIR" ]] && ls "$TRITON_WHEEL_DIR"/*.whl 1>/dev/null 2>&1; then
    echo "Installing triton from pre-built wheel in $TRITON_WHEEL_DIR"
    retry_with_backoff "$TRITON_RETRY_ATTEMPTS" "$TRITON_RETRY_SLEEP_SECONDS" install_triton_from_wheel
else
    echo "Building triton from source..."
    retry_with_backoff "$TRITON_RETRY_ATTEMPTS" "$TRITON_RETRY_SLEEP_SECONDS" build_triton_from_source
fi
