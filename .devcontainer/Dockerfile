FROM rocm/pytorch:latest

ARG DEBIAN_FRONTEND=noninteractive

# 1. Install System Dependencies (APT/Ubuntu style)
RUN apt-get update && apt-get install -y \
    git \
    cmake \
    vim \
    python3-pip \
    libdrm-dev \
    libelf-dev \
    libomp-dev \
    pkg-config \
    gcc \
    g++ \
    make \
    zsh \
    curl \
    wget \
    jq \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# 2. Install Oh My Zsh (for root)
RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended

# 3. Install Python Build Tools
RUN pip3 install --upgrade pip setuptools==75.1.0 wheel

# 4. Install PyTorch Nightly (ROCm 7.0)
RUN pip3 uninstall -y torch && \
    pip3 install --no-cache-dir --pre torch torchvision \
    --index-url https://download.pytorch.org/whl/nightly/rocm7.1

# 5. Keep container alive
CMD ["sleep", "infinity"]
