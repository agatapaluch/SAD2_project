# Use an older Ubuntu that still has Python 2.7 packages easily available
FROM ubuntu:18.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /work

# System deps for building numpy/scipy + BLAS/LAPACK + Fortran compiler
RUN apt-get update && apt-get install -y --no-install-recommends \
    python2.7 python2.7-dev \
    curl ca-certificates \
    build-essential gfortran \
    libopenblas-dev liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

# Install pip for Python 2
RUN curl -fsSL https://bootstrap.pypa.io/pip/2.7/get-pip.py -o /tmp/get-pip.py \
    && python2.7 /tmp/get-pip.py \
    && rm /tmp/get-pip.py

# Pin tooling compatible with Python 2.7
RUN python2.7 -m pip install --no-cache-dir "pip<21" "setuptools<45" wheel

# Install numpy/scipy versions that support Python 2.7, then BNfinder
RUN python2.7 -m pip install --no-cache-dir "numpy==1.16.6" \
    && python2.7 -m pip install --no-cache-dir "scipy==1.2.3" \
    && python2.7 -m pip install --no-cache-dir "BNfinder==2.0.4" \
    && python2.7 -m pip install --no-cache-dir "networkx==2.2" "boolean.py==3.8" \
    && python2.7 -m pip install --no-cache-dir "pandas==0.24.2" "tqdm==4.64.1" \
    && python2.7 -m pip install --no-cache-dir "matplotlib==2.2.5" "seaborn==0.9.1"
# Default to bash so you can run commands interactively
CMD ["bash"]
