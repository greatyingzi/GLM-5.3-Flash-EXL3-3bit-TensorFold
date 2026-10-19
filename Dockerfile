# Final 3-bit image (equivalent of tensorfold-glm53:3bit-v18-k3, verified against the live serving image):
#   - allow_3bit.py        -> EXL3_ALLOWED_BITS = (3, 4) in families/glm5_next/__init__.py   [present]
#   - fix_v7_pick.py       -> pick buffer clamp (tf-v7 marker)                               [present]
#   - fix_v12_smem.py      -> group_kernel 96KB dynamic smem (tf-v12 marker)                 [present]
#   - k3 route             -> exl3k3.cu/.cpp installed + exl3_mm.py accepts trellis [...,48]  [present]
#   - fix_v5_nocapture.py  -> NOT applied (withdrawn in v17; verify CUDA graphs restored)     [absent]
#   - CUDA_LAUNCH_BLOCKING=0 (debug env removed in v16; leaving it on costs ~40% decode speed)
#
# Base: the Mia's AI Lab GLM-5.3-Flash-EXL3-2x-DGX-Sparks recipe image (TensorFold v0.6.0, Apache-2.0).
# If you built the intermediate `tensorfold-glm53:lowbit` / `:3bit` images from this repo's earlier
# Dockerfiles, FROM that instead — the patch scripts are idempotent.
FROM tensorfold-glm53:v0.6.0

COPY patches /opt/tf-3bit/patches
COPY k3-kernel /opt/tf-3bit/k3-kernel

RUN set -e; \
    cp /opt/tf-3bit/k3-kernel/exl3k3.cu /opt/tf-3bit/k3-kernel/exl3k3.cpp /tmp/; \
    python3 /opt/tf-3bit/patches/allow_3bit.py && \
    python3 /opt/tf-3bit/patches/fix_v7_pick.py && \
    python3 /opt/tf-3bit/patches/fix_v12_smem.py && \
    python3 /opt/tf-3bit/patches/patch_k3.py && \
    echo "== build-time self-checks ==" && \
    grep -q "EXL3_ALLOWED_BITS = (3, 4)" /usr/local/lib/python3.12/dist-packages/tensorfold/families/glm5_next/__init__.py && \
    grep -q "tf-v7" /usr/local/lib/python3.12/dist-packages/tensorfold/families/glm5_next/cuda/exl3_mm.py && \
    grep -q "tf-v12" /usr/local/lib/python3.12/dist-packages/tensorfold/cuda/exl3/experts.cu && \
    grep -q "48" /usr/local/lib/python3.12/dist-packages/tensorfold/families/glm5_next/cuda/exl3_mm.py && \
    test -f /usr/local/lib/python3.12/dist-packages/tensorfold/families/glm5_next/cuda/exl3k3.cu && \
    echo "all patches verified" && \
    rm -rf /cache/torch_extensions

# NOTE: do NOT set CUDA_LAUNCH_BLOCKING=1 in production images (measured 34 -> 58 tok/s when removed).
ENV CUDA_LAUNCH_BLOCKING=0 \
    TF_GLM_EXL3_K3=1
