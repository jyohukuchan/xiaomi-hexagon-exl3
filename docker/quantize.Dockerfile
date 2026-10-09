FROM pytorch/pytorch:2.10.0-cuda13.0-cudnn9-devel@sha256:48af19ebb88034e0325decc2b6142e5b9bfc276eaa5197ab6d94582f1d78ea4c

ENV MAX_JOBS=2 TORCH_CUDA_ARCH_LIST=8.9
RUN apt-get update && apt-get install -y --no-install-recommends git python3-venv && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv --system-site-packages /opt/quant-venv
ENV PATH="/opt/quant-venv/bin:${PATH}"
RUN python -m pip install --no-cache-dir 'setuptools>=77' wheel ninja
RUN git clone https://github.com/turboderp-org/exllamav3.git /opt/exllamav3 \
    && git -C /opt/exllamav3 checkout 151539c77abc7ab7425d30da7a4e8e3c5c154e7b \
    && python -m pip install --no-cache-dir --no-build-isolation /opt/exllamav3
RUN python -m pip install --no-cache-dir transformers==5.19.0 sentencepiece==0.2.2
WORKDIR /opt/exllamav3
