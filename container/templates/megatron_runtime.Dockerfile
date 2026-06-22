{#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#}
# === BEGIN templates/megatron_runtime.Dockerfile ===
##################################
########## Runtime Image #########
##################################

# Phase 0/3 image: NGC PyTorch base + Megatron-LM (cloned at build time) +
# Dynamo wheels + NIXL (Phase-3 disagg KV transport). KVBM + GPU memory
# service still land later. Phase-1/2 metrics + KV events are stashed under
# .stashed_phase1_phase2/ and re-applied independently.

# Transport stage — runtime pulls /workspace_src in one bind-mount cp.
FROM scratch AS workspace_files
COPY --chmod=775 tests /workspace_src/tests
COPY --chmod=775 examples /workspace_src/examples
COPY --chmod=775 deploy /workspace_src/deploy
COPY --chmod=775 dev /workspace_src/dev
COPY --chmod=775 components/src/dynamo/common /workspace_src/components/src/dynamo/common
COPY --chmod=775 components/src/dynamo/frontend /workspace_src/components/src/dynamo/frontend
COPY --chmod=775 components/src/dynamo/megatron /workspace_src/components/src/dynamo/megatron
COPY --chmod=775 lib /workspace_src/lib
COPY --chmod=664 ATTRIBUTION* LICENSE /workspace_src/

# Transport stage for dynamo_base artifacts.
FROM scratch AS dynamo_base_export
COPY --from=dynamo_base /usr/bin/nats-server /usr/bin/nats-server
COPY --from=dynamo_base /usr/local/bin/etcd/ /usr/local/bin/etcd/
COPY --from=dynamo_base /bin/uv /usr/bin/uv
COPY --from=dynamo_base /bin/uvx /usr/bin/uvx

{% if target == "runtime" %}
FROM ${RUNTIME_IMAGE}:${RUNTIME_IMAGE_TAG} AS runtime_full
{% else %}
FROM ${RUNTIME_IMAGE}:${RUNTIME_IMAGE_TAG} AS runtime
{% endif %}

ARG MEGATRON_REPO
ARG MEGATRON_REF
ARG TARGETARCH

# DYNAMO_HOME points at /workspace so bundled scripts under
# $DYNAMO_HOME/examples resolve. Dynamo + Megatron-LM both install into
# /opt/dynamo/venv (created below with --system-site-packages so upstream
# PyTorch/CUDA libs stay importable) — the NGC PyTorch base ships PEP 668
# system Python so we can't install into /usr directly.
ENV DYNAMO_HOME=/workspace \
    HOME=/home/dynamo \
    MEGATRON_HOME=/opt/megatron-lm \
    VIRTUAL_ENV=/opt/dynamo/venv \
    PATH=/opt/dynamo/venv/bin:/usr/local/bin/etcd:${PATH} \
    PYTHONPATH=/opt/megatron-lm:${PYTHONPATH:-}

WORKDIR /workspace

# OS packages: git for the Megatron clone, openssh + rdma for downstream
# multi-node bring-up later, python3-venv so we can build /opt/dynamo/venv.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        git \
        openssh-server \
        librdmacm1 \
        rdma-core \
        python3-venv

# Pull nats-server, etcd, uv, uvx into their final paths.
COPY --from=dynamo_base_export / /

# Create the Dynamo venv with --system-site-packages so the upstream PyTorch
# image's pinned torch/numpy/triton/cuda libs remain importable. Everything
# below installs into this venv via uv pip (no --system flag).
RUN mkdir -p /opt/dynamo \
    && python3 -m venv --system-site-packages /opt/dynamo/venv \
    && ln -sf /usr/bin/uv /opt/dynamo/venv/bin/uv

# Clone Megatron-LM at the configured ref and install it editable so users
# can later mount their own checkout over /opt/megatron-lm at `docker run`
# time without rebuilding. --no-deps preserves the NGC PyTorch image's solve;
# Megatron's own pyproject extras are layered on top below.
RUN git clone --depth 1 --branch "${MEGATRON_REF}" "${MEGATRON_REPO}" /opt/megatron-lm \
    && uv pip install --no-deps -e /opt/megatron-lm

# Megatron-Inference + the Dynamo backend's runtime dep set. Kept narrow
# (msgpack + pyzmq for the wire protocol, msgspec/uvloop for ai-dynamo-runtime,
# transformers for the tokenizer path). Installed --no-deps so we don't
# perturb upstream PyTorch's pinned numpy/triton/etc. NIXL is NOT in this
# requirements file — it's COPYed from wheel_builder below.
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    --mount=type=bind,source=./container/deps/requirements.megatron.txt,target=/tmp/requirements.megatron.txt \
    uv pip install --no-deps --requirement /tmp/requirements.megatron.txt

# Megatron's `ssm` extra — required by hybrid/SSM models (e.g. Nemotron-H /
# nanov3), whose Mamba layers import mamba_ssm + causal_conv1d. No aarch64
# wheels exist on PyPI, so these compile from source.
#
# --no-deps is CRITICAL: mamba-ssm/causal-conv1d's runtime deps (torch, triton,
# einops, transformers) are already pinned in the NGC PyTorch base. Letting uv
# resolve them pulls a PyPI torch + nvidia-cublas-cu13 wheel into the venv,
# which shadows the base's cuBLAS and breaks the NGC-built transformer_engine
# (undefined symbol: cublasLtGroupedMatrixLayoutInit_internal). Same --no-deps
# discipline as every other install in this image.
#
# --no-build-isolation builds against the base image's torch/CUDA (matches
# Megatron's own uv `no-build-isolation-package` config; the NGC PyTorch base
# ships nvcc). The FORCE_BUILD vars stop the packages short-circuiting to a
# (nonexistent) prebuilt wheel. ninja is the only build tool not already
# present. Adds a slow CUDA compile to the build, but it layer-caches.
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    uv pip install --no-deps ninja && \
    MAMBA_FORCE_BUILD=TRUE CAUSAL_CONV1D_FORCE_BUILD=TRUE \
    uv pip install --no-deps --no-build-isolation "causal-conv1d~=1.5" "mamba-ssm~=2.2"

# Dynamo user (group 0 for OpenShift), reset upstream /workspace baggage.
# /opt/dynamo was already created above to host the venv. Must come BEFORE
# the NIXL COPYs below — those use --chown=dynamo:0 and would fail with
# "invalid user" if the user doesn't exist yet.
RUN userdel -r ubuntu > /dev/null 2>&1 || true \
    && useradd -m -s /bin/bash -g 0 dynamo \
    && [ `id -u dynamo` -eq 1000 ] \
    && mkdir -p /home/dynamo/.cache \
    && ln -sf /usr/bin/python3 /usr/local/bin/python \
    && rm -rf /workspace && mkdir /workspace \
    && chown dynamo:0 /home/dynamo /home/dynamo/.cache /opt/dynamo /workspace /opt/megatron-lm \
    && mkdir -p /etc/profile.d \
    && echo 'umask 002' > /etc/profile.d/00-umask.sh

# Phase-3 disagg: NIXL native libs + Python wheels, copied from wheel_builder
# (same pattern as dynamo_runtime.Dockerfile / trtllm_runtime.Dockerfile).
# wheel_builder.Dockerfile builds NIXL from source against UCX when the
# framework block in context.yaml has `nixl_ref` set — megatron does
# (currently 0.10.1).
ENV NIXL_PREFIX=/opt/nvidia/nvda_nixl \
    NIXL_LIB_DIR=/opt/nvidia/nvda_nixl/lib64 \
    NIXL_PLUGIN_DIR=/opt/nvidia/nvda_nixl/lib64/plugins

ENV LD_LIBRARY_PATH=\
${NIXL_LIB_DIR}:\
${NIXL_PLUGIN_DIR}:\
/usr/local/ucx/lib:\
/usr/local/ucx/lib/ucx:\
/usr/local/cuda/lib64:\
/usr/local/cuda/compat:\
${LD_LIBRARY_PATH:-}

# Native NIXL + UCX runtime libs, and the locally-built wheels.
COPY --chown=dynamo:0 --from=wheel_builder /usr/local/ucx/ /usr/local/ucx/
COPY --chown=dynamo:0 --from=wheel_builder ${NIXL_PREFIX}/ ${NIXL_PREFIX}/
COPY --chown=dynamo:0 --from=wheel_builder /opt/dynamo/dist/nixl/ /opt/dynamo/wheelhouse/nixl/
COPY --chown=dynamo:0 --from=wheel_builder /workspace/nixl/build/src/bindings/python/nixl-meta/nixl-*.whl /opt/dynamo/wheelhouse/nixl/

# Place wheels in /opt/dynamo/wheelhouse unconditionally — dev/local-dev
# images install from source and skip the pip install RUN below but still
# need the wheels on disk.
COPY --chmod=775 --chown=dynamo:0 --from=wheel_builder /opt/dynamo/dist/*.whl /opt/dynamo/wheelhouse/

{% if target not in ("dev", "local-dev") %}
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    export UV_CACHE_DIR=/root/.cache/uv && \
    # Dynamo wheels — --no-deps preserves upstream's solve.
    uv pip install --no-deps /opt/dynamo/wheelhouse/ai_dynamo_runtime*.whl && \
    uv pip install --no-deps /opt/dynamo/wheelhouse/ai_dynamo*any.whl && \
    # Phase-3 NIXL wheels (cu12 CUDA wheel + metapackage facade). Both wheels
    # are produced by wheel_builder; the metapackage exposes `nixl._api` and
    # the cu12 wheel ships libnixl.so. --no-deps consistent with above.
    uv pip install --no-deps /opt/dynamo/wheelhouse/nixl/nixl*.whl
{% endif %}

# Pull /workspace_src (incl. ATTRIBUTION/LICENSE) from the transport stage.
RUN --mount=type=bind,from=workspace_files,source=/workspace_src,target=/tmp/workspace_src \
    --mount=type=bind,source=./container/launch_message/runtime.txt,target=/opt/dynamo/launch_message.txt \
    cp -a /tmp/workspace_src/. /workspace/ && \
    chown -R dynamo:0 /workspace && \
    sed '/^#\s/d' /opt/dynamo/launch_message.txt > /opt/dynamo/.launch_screen && \
    chmod 755 /opt/dynamo/.launch_screen && \
    echo 'cat /opt/dynamo/.launch_screen' >> /etc/bash.bashrc

USER dynamo

# Kept at the bottom — SHA changes per build; layers above stay cached.
ARG DYNAMO_COMMIT_SHA
ENV DYNAMO_COMMIT_SHA=${DYNAMO_COMMIT_SHA}

ENTRYPOINT []
CMD ["/bin/bash"]

{% if target == "runtime" %}
# Rebase on upstream so this stage inherits upstream's image config
# (ENV/WORKDIR/USER/CMD) and overlay runtime_full's filesystem as a single
# layer.
FROM ${RUNTIME_IMAGE}:${RUNTIME_IMAGE_TAG} AS runtime
RUN rm -rf /workspace /home/ubuntu /usr/local/bin/etcd
COPY --from=runtime_full / /

ENV DYNAMO_HOME=/workspace \
    HOME=/home/dynamo \
    MEGATRON_HOME=/opt/megatron-lm \
    VIRTUAL_ENV=/opt/dynamo/venv \
    PATH=/opt/dynamo/venv/bin:/usr/local/bin/etcd:${PATH} \
    PYTHONPATH=/opt/megatron-lm:${PYTHONPATH:-} \
    NIXL_PREFIX=/opt/nvidia/nvda_nixl \
    NIXL_LIB_DIR=/opt/nvidia/nvda_nixl/lib64 \
    NIXL_PLUGIN_DIR=/opt/nvidia/nvda_nixl/lib64/plugins \
    LD_LIBRARY_PATH=/opt/nvidia/nvda_nixl/lib64:/opt/nvidia/nvda_nixl/lib64/plugins:/usr/local/ucx/lib:/usr/local/ucx/lib/ucx:/usr/local/cuda/lib64:/usr/local/cuda/compat:${LD_LIBRARY_PATH:-}

WORKDIR /workspace

ARG DYNAMO_COMMIT_SHA
ENV DYNAMO_COMMIT_SHA=${DYNAMO_COMMIT_SHA}

USER dynamo

ENTRYPOINT []
CMD ["/bin/bash"]
{% endif %}
