### 1. Summary of the Interactive Installation Process

This documents the manual, step-by-step process we followed to build the environment inside a container, including all the necessary fixes for modern hardware and software conflicts.

**Objective:** To create a persistent, GPU-enabled Docker container with the specific, legacy dependencies required by Molformer, addressing modern compatibility issues along the way.

**Step 1: Launch a Base Container**  
We started with a specific NVIDIA CUDA development image to ensure the correct CUDA Toolkit (11.0) and C++ compilers were available from the beginning.

```bash
docker run --gpus all -it --name molformer-interactive -v $(pwd):/workspace nvidia/cuda:11.0.3-devel-ubuntu18.04 bash
```

**Step 2: Install System Dependencies & Correct Pip Version**  
We installed basic tools and Python 3.8. A key fix was required here because the standard `get-pip.py` script no longer supports Python 3.8.

```bash
# Install system tools
apt-get update && apt-get install -y build-essential git wget python3.8 python3.8-dev python3.8-distutils

# FIX: Download the pip installation script specifically for Python 3.8
wget https://bootstrap.pypa.io/pip/3.8/get-pip.py
python3.8 get-pip.py
rm get-pip.py

# Set python3.8 as the default
ln -sf /usr/bin/python3.8 /usr/bin/python
ln -sf /usr/local/bin/pip /usr/bin/pip
```

**Step 3: Install Python Packages in a Specific Order**  
A single `pip install` command failed due to two issues:

1. `pytorch-fast-transformers` requires `torch` to be installed _before_ its own installation begins.
2. `datasets==1.6.2` has malformed metadata that is rejected by modern versions of `pip`.

The solution was a three-stage installation:

```bash
# Stage A: Install PyTorch first to satisfy other packages' build dependencies.
pip install torch==1.7.1+cu110 -f https://download.pytorch.org/whl/torch_stable.html

# Stage B (FIX): Downgrade pip to a version that accepts the old 'datasets' package metadata.
python -m pip install "pip<24.1"

# Stage C: Install all remaining packages.
pip install numpy==1.22.3 pandas==1.2.4 scikit-learn==0.24.2 scipy==1.6.2 \
            rdkit-pypi==2022.03.2 transformers==4.6.0 pytorch-lightning==1.1.5 \
            pytorch-fast-transformers==0.4.0 datasets==1.6.2 jupyterlab==3.4.0 \
            ipywidgets==7.7.0 bertviz==1.4.0
```

**Step 4: Compile Apex with Hardware Compatibility Fixes**  
The final step, compiling Apex, failed due to two hardware/software mismatches:

1. The PyTorch 1.7.1 build tools could not recognize the modern GPU architecture (e.g., RTX 40-series `8.9`).
2. The CUDA 11.0 compiler in the container did not support the compile flag for newer architectures (e.g., RTX 30-series `sm_86`).

The solution was to use the `TORCH_CUDA_ARCH_LIST` environment variable to force compilation for older, compatible architectures that the compiler understands and the modern GPU can run.

```bash
# Clone the correct version of Apex
cd /tmp
git clone https://github.com/NVIDIA/apex
cd apex
git checkout tags/22.03 -b v22.03

# FIX: Set the arch list and install
TORCH_CUDA_ARCH_LIST="7.5 8.0" pip install -v --no-cache-dir --no-build-isolation \
--config-settings "--build-option=--cpp_ext" \
--config-settings "--build-option=--cuda_ext" ./
```

**Step 5: Cleanup and Commit the Final Image**  
To create a clean, reusable image, we removed temporary build files and caches before committing the container's state.

```bash
# Clean up caches and temp files inside the container
pip cache purge
apt-get clean
rm -rf /tmp/apex /root/get-pip.py /var/lib/apt/lists/*
exit

# Commit the container to a new image from the host machine
docker commit \
  --change='WORKDIR /workspace' \
  --change='CMD ["bash"]' \
  molformer-interactive \
  molformer-env:latest
```
