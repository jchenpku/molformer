### **Final Guide: Manually Building a Compatible MolFormer Docker Environment**

This document outlines the step-by-step process to manually create a Docker environment for running the MolFormer embedding pipeline on modern NVIDIA GPUs (e.g., RTX 30/40-series). The process addresses several version incompatibilities between the original repository's dependencies, modern hardware, and newer software libraries.

**Objective:** To create a persistent, GPU-enabled Docker container with a working MolFormer environment, capable of large-scale inference.

**Step 1: Launch a Modern Base Container**  
We start with a newer NVIDIA CUDA development image (11.8 on Ubuntu 20.04) to ensure native support for modern GPU architectures and to provide a more up-to-date toolchain.

```bash
# Launch an interactive container, mounting your project directory
# Replace /path/to/your/project with your actual host path
docker run -it --gpus all --name molformer-interactive \
  -v /path/to/your/project:/workspace \
  nvidia/cuda:11.8.0-devel-ubuntu20.04 bash
```

**Step 2: Install System Dependencies**  
Inside the container, update the system and install essential tools. Python 3.8 is the default `python3` on Ubuntu 20.04, simplifying setup.

```bash
# Update package lists
apt-get update

# Install system tools
apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    ca-certificates \
    python3.8 \
    python3-pip \
    sed

# Set python3 and pip3 as the default 'python' and 'pip'
ln -sf /usr/bin/python3.8 /usr/bin/python
ln -sf /usr/bin/pip3 /usr/bin/pip

# Clean up apt cache
rm -rf /var/lib/apt/lists/*
```

**Step 3: Install Python Packages in a Specific Order**  
To resolve complex dependency conflicts between PyTorch, its dependencies, and other libraries like `lancedb`, we install packages in a specific, staged order.

```bash
# Stage A: Upgrade build tools and pin a Python 3.8-compatible networkx version.
# This prevents PyTorch 2.x from installing an incompatible version of networkx.
pip install --upgrade pip setuptools wheel
pip install networkx==2.8.8

# Stage B: Install a modern PyTorch with CUDA 11.8 support.
# This is the key fix for the low-level MAGMA/CUDA errors on RTX 40-series GPUs.
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --index-url https://download.pytorch.org/whl/cu118

# Stage C: Install the remaining application and pipeline dependencies.
# We pin lancedb and pylance to known compatible versions.
pip install \
    numpy pandas scikit-learn scipy \
    rdkit-pypi \
    transformers==4.6.0 \
    pytorch-lightning==1.5.10 \
    pytorch-fast-transformers==0.4.0 \
    datasets==1.6.2 \
    pyarrow lmdb lancedb==0.6.1 pylance==0.10.6 tqdm
```

**Step 4: Clone, Patch, and Compile Apex**  
The official Apex repository's latest version uses Python 3.10+ syntax, which is incompatible with our Python 3.8 environment. The solution is to clone the latest code (which is compatible with PyTorch 2.x APIs) and then apply a small patch to remove the incompatible syntax before compiling.

```bash
# 1. Clone the latest Apex repository
git clone https://github.com/NVIDIA/apex /tmp/apex
cd /tmp/apex

# 2. PATCH: Replace the Python 3.10+ syntax in setup.py with a compatible version.
sed -i "s/parallel: int | None = None/parallel = None/" setup.py

# 3. Set the target GPU architectures (including 8.9 for RTX 4090)
export TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9"

# 4. Install using the direct setup.py method, which is robust
python setup.py install --cpp_ext --cuda_ext

# 5. Clean up the build files
cd /
rm -rf /tmp/apex
```

**Step 5: Set up the MolFormer Code**  
The MolFormer repository is not an installable package. The correct way to make its modules available is by adding its location to the `PYTHONPATH`.

```bash
# 1. Clone the repository
git clone https://github.com/IBM/molformer.git /workspace/molformer

# 2. Add its path to the bash profile to make the setting persistent for new shells
echo 'export PYTHONPATH="${PYTHONPATH}:/workspace/molformer"' >> ~/.bashrc

# 3. Apply the change to the current shell session immediately
source ~/.bashrc

# 4. Verify that it's set correctly
echo $PYTHONPATH
# The output should contain ':/workspace/molformer'
```
**Note:** The forked repository `jchenpku/molformer` is a great alternative that makes the code installable via `pip install -e .`. The method above works directly with the official IBM repository.

**Step 6: Final Cleanup and Committing the Image**  
To create a clean, reusable image from your interactive session, clean up caches before exiting and committing.

```bash
# Clean up pip cache
rm -rf ~/.cache/pip

# Exit the interactive container session
exit

# On your host machine, commit the container's state to a new image
docker commit \
  --change='WORKDIR /workspace/molformer' \
  --change='CMD ["bash"]' \
  molformer-interactive \
  molformer-env:final
```
