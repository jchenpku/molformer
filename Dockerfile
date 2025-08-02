# Use the official NVIDIA CUDA 11.0 development image as the base
FROM nvidia/cuda:11.0.3-devel-ubuntu18.04

# Set environment variables to prevent interactive prompts and set timezone
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Install system dependencies, Python 3.8, and the correct version of Pip
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    ca-certificates \
    python3.8 \
    python3.8-dev \
    python3.8-distutils && \
    # FIX 1: Use the correct get-pip.py script for Python 3.8
    wget https://bootstrap.pypa.io/pip/3.8/get-pip.py && \
    python3.8 get-pip.py && \
    rm get-pip.py && \
    # Create standard symlinks
    ln -sf /usr/bin/python3.8 /usr/bin/python && \
    ln -sf /usr/local/bin/pip /usr/bin/pip && \
    # Clean up apt cache
    rm -rf /var/lib/apt/lists/*

# Install Python packages in stages to resolve dependency conflicts

# Stage 1: Install PyTorch first, as other packages depend on it for their build process
RUN pip install torch==1.7.1+cu110 -f https://download.pytorch.org/whl/torch_stable.html

# Stage 2: Downgrade pip to handle malformed metadata in old packages
RUN python -m pip install "pip<24.1"

# Stage 3: Install all other Python packages
RUN pip install \
    numpy==1.22.3 \
    pandas==1.2.4 \
    scikit-learn==0.24.2 \
    scipy==1.6.2 \
    rdkit-pypi==2022.03.2 \
    transformers==4.6.0 \
    pytorch-lightning==1.1.5 \
    pytorch-fast-transformers==0.4.0 \
    datasets==1.6.2 \
    jupyterlab==3.4.0 \
    ipywidgets==7.7.0 \
    bertviz==1.4.0

# Stage 4: Compile and install Apex with hardware compatibility fixes
# FIX 2: Set TORCH_CUDA_ARCH_LIST to compile for architectures supported by CUDA 11.0
RUN git clone https://github.com/NVIDIA/apex /tmp/apex && \
    cd /tmp/apex && \
    git checkout tags/22.03 -b v22.03 && \
    TORCH_CUDA_ARCH_LIST="7.5 8.0" pip install -v --no-cache-dir --no-build-isolation \
    --config-settings "--build-option=--cpp_ext" \
    --config-settings "--build-option=--cuda_ext" ./ && \
    # Clean up source code and pip cache
    cd / && rm -rf /tmp/apex && \
    pip cache purge

# Set the default working directory
WORKDIR /workspace

# Set the default command to launch a bash shell
CMD ["bash"]
