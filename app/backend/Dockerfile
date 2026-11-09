# TrialO screening backend (educational prototype)
#
# CPU wheels of torch by default (the served `smoke_cpu` preset is a 270m CPU
# model). For the GPU `final_a100` preset, replace the torch install with the
# CUDA wheel line commented below.
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

# Markdown -> PDF (weasyprint) needs pango/cairo system libraries, and text
# rendering needs at least one installed font family.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libpango-1.0-0 \
        libpangocairo-1.0-0 \
        libcairo2 \
        libgdk-pixbuf-2.0-0 \
        libffi-dev \
        fontconfig \
        fonts-dejavu-core \
        shared-mime-info \
    && rm -rf /var/lib/apt/lists/*

# CPU torch first so the rest of the requirements resolve against it
# (comment out this line and the matching CUDA line when building for GPU).
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.14.1"
# For a GPU build (CUDA 12.x): pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

COPY trailo_app /app/trailo_app

# The repository's own research code is mounted at runtime (read-only) so the
# app always drives the same trailo.eval machinery it was validated with.
ENV TRIALO_REPO_ROOT=/workspace \
    PYTHONPATH=/workspace/src:/app \
    HF_HOME=/workspace/hf

WORKDIR /app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/api/health')"

CMD ["uvicorn", "trailo_app.main:app", "--host", "0.0.0.0", "--port", "8000"]
