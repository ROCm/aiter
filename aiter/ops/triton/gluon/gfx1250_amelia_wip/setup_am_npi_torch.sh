#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration file location
CONFIG_FILE="$HOME/.am_npi_setup_config"

# Helper functions
print_header() {
    echo -e "\n${BLUE}================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Function to save configuration
save_config() {
    cat > "$CONFIG_FILE" << EOF
# AM NPI Torch Setup Configuration
# Generated on $(date)
NTID="$NTID"
CONTAINER_NAME="$CONTAINER_NAME"
EOF
    chmod 600 "$CONFIG_FILE"
}

# Function to load configuration
load_config() {
    if [[ -f "$CONFIG_FILE" ]]; then
        source "$CONFIG_FILE"
        return 0
    fi
    return 1
}

# Gather user inputs
print_header "Triton AM Environment Setup - NPI PyTorch (MI450)"

echo -e "${YELLOW}This script will set up a Docker container for Triton AM development${NC}"
echo -e "${YELLOW}using NPI PyTorch builds (matched to the ROCm NPI base image).${NC}\n"

# Load any existing config for defaults
load_config

NTID=${NTID:-$(whoami)}
CONTAINER_NAME=${CONTAINER_NAME:-${NTID}_am_npi_docker}

echo -e "${YELLOW}Press Enter to accept the default shown in brackets.${NC}\n"

read -p "Enter your NTID (username) [default: $NTID]: " NEW_NTID
NTID=${NEW_NTID:-$NTID}

read -p "Enter Docker container name [default: $CONTAINER_NAME]: " NEW_CONTAINER_NAME
CONTAINER_NAME=${NEW_CONTAINER_NAME:-$CONTAINER_NAME}

echo ""
echo -e "${YELLOW}How do you want to clone Triton and AITER repositories?${NC}"
echo "  1) SSH  (uses ssh-agent forwarding, requires SSH key on github.amd.com)"
echo "  2) HTTPS (uses a Personal Access Token)"
read -p "Choose clone method [1/2, default: 1]: " CLONE_METHOD_CHOICE
CLONE_METHOD_CHOICE=${CLONE_METHOD_CHOICE:-1}

GIT_PAT=""
if [[ "$CLONE_METHOD_CHOICE" == "2" ]]; then
    CLONE_METHOD="https"
    read -sp "Enter your GitHub Personal Access Token (PAT) for github.amd.com: " GIT_PAT
    echo ""
    if [[ -z "$GIT_PAT" ]]; then
        print_error "PAT cannot be empty for HTTPS cloning."
        exit 1
    fi
else
    CLONE_METHOD="ssh"
fi

# Confirmation
echo -e "\n${YELLOW}Configuration Summary:${NC}"
echo "  NTID: $NTID"
echo "  Container Name: $CONTAINER_NAME"
echo "  Clone Method: $CLONE_METHOD"
echo ""
read -p "Continue with these settings? (y/n): " CONFIRM
if [[ $CONFIRM == [nN] ]]; then
    echo "Setup cancelled."
    exit 0
fi

# Save configuration for next time
save_config
print_success "Configuration saved to $CONFIG_FILE"

# ============================================================================
# Fetch Latest AM+FFM-Lite Package from Artifactory
# ============================================================================
print_header "Fetching Latest AM+FFM-Lite Package"

ARTIFACTORY_BASE="https://atlartifactory.amd.com:8443/artifactory"
AM_RELEASE_PATH="SW-ROCDTIF-MI-DEV-LOCAL/Packages/AM+FFM-LITE/Release"
API_URL="${ARTIFACTORY_BASE}/api/storage/${AM_RELEASE_PATH}"

FETCH_SUCCESS=false
while [[ $FETCH_SUCCESS == false ]]; do
    read -sp "Enter your NTID ($NTID) password for Artifactory: " AM_PASSWORD
    echo ""

    echo -e "${BLUE}Querying Artifactory for latest package...${NC}"

    HTTP_CODE=$(curl -s -o /tmp/am_api_response.json -w "%{http_code}" -u "${NTID}:${AM_PASSWORD}" "$API_URL" 2>/dev/null)

    if [[ "$HTTP_CODE" == "401" || "$HTTP_CODE" == "403" ]]; then
        print_error "Authentication failed (HTTP $HTTP_CODE). Wrong password?"
        read -p "Retry? (y/n): " RETRY_FETCH
        if [[ $RETRY_FETCH != [yY] ]]; then
            print_error "Cannot continue without Artifactory access."
            exit 1
        fi
        continue
    elif [[ "$HTTP_CODE" != "200" ]]; then
        print_error "Failed to connect to Artifactory (HTTP $HTTP_CODE)."
        print_error "URL: $API_URL"
        read -p "Retry? (y/n): " RETRY_FETCH
        if [[ $RETRY_FETCH != [yY] ]]; then
            print_error "Cannot continue without Artifactory access."
            exit 1
        fi
        continue
    fi

    AM_PACKAGE=$(python3 -c "
import json, sys
with open('/tmp/am_api_response.json') as f:
    data = json.load(f)
files = sorted([
    c['uri'].lstrip('/')
    for c in data.get('children', [])
    if not c.get('folder', False) and c['uri'].endswith('.tar.gz')
])
print(files[-1] if files else '')
" 2>/dev/null)

    rm -f /tmp/am_api_response.json

    if [[ -z "$AM_PACKAGE" ]]; then
        print_error "No .tar.gz packages found in the Release directory."
        print_error "Check: ${ARTIFACTORY_BASE}/${AM_RELEASE_PATH}/"
        exit 1
    fi

    FETCH_SUCCESS=true
done

AM_EXTRACT_DIR="${AM_PACKAGE%.tar.gz}"
print_success "Latest package: $AM_PACKAGE"
echo ""

# ============================================================================
# STEP 1: Setup SSH Key Forwarding
# ============================================================================
print_header "Step 1: Setting up SSH Key Forwarding"

if grep -q "SSH_ENV=\"\$HOME/.ssh/agent-environment\"" ~/.bashrc; then
    print_warning "SSH forwarding setup already exists in ~/.bashrc, skipping..."
else
    echo "Adding SSH forwarding setup to ~/.bashrc..."
    cat >> ~/.bashrc << 'EOF'

# SSH Key Forwarding Setup
export SSH_ENV="$HOME/.ssh/agent-environment"
start_agent() {
    echo "Starting new ssh-agent..."
    (umask 066; ssh-agent -a "$HOME/.ssh/ssh-agent.sock" > "$SSH_ENV")
    source "$SSH_ENV"
    for key in ~/.ssh/id_*; do
        [[ -f "$key" && "$key" != *.pub ]] && ssh-add -l | grep -q "$(ssh-keygen -lf "$key" | awk '{print $2}')" || ssh-add "$key" 2>/dev/null
    done
    export SSH_AUTH_SOCK_DIR="$(dirname "$SSH_AUTH_SOCK")"
}

if [[ -f "$SSH_ENV" ]]; then
    source "$SSH_ENV" > /dev/null
    [[ "$SSH_AUTH_SOCK" != "$HOME/.ssh/ssh-agent.sock" || ! -S "$HOME/.ssh/ssh-agent.sock" ]] && start_agent
    kill -0 "$SSH_AGENT_PID" 2>/dev/null || start_agent
else
    start_agent
fi

export SSH_AUTH_SOCK_DIR="$(dirname "$SSH_AUTH_SOCK")"
EOF
    print_success "SSH forwarding setup added to ~/.bashrc"
fi

# Fix SSH permissions
print_header "Fixing SSH Permissions"

if [[ ! -d ~/.ssh ]]; then
    print_warning "~/.ssh directory does not exist, creating it..."
    mkdir -p ~/.ssh
    chmod 700 ~/.ssh
    print_success "Created ~/.ssh directory"
else
    sudo chown -R $NTID:$NTID ~/.ssh
    chmod 700 ~/.ssh

    for key in ~/.ssh/id_*; do
        if [[ -f "$key" && "$key" != *.pub ]]; then
            chmod 600 "$key"
            print_success "Set permissions for private key: $key"
        elif [[ -f "$key" && "$key" == *.pub ]]; then
            chmod 644 "$key"
            print_success "Set permissions for public key: $key"
        fi
    done

    [[ -f ~/.ssh/config ]] && chmod 644 ~/.ssh/config
    [[ -f ~/.ssh/known_hosts ]] && chmod 644 ~/.ssh/known_hosts

    print_success "SSH permissions fixed"
fi

source ~/.bashrc

if [[ -z "$SSH_AUTH_SOCK" ]]; then
    print_warning "SSH_AUTH_SOCK not set after sourcing .bashrc, initializing SSH agent..."
    export SSH_ENV="$HOME/.ssh/agent-environment"
    if [[ -f "$SSH_ENV" ]]; then
        source "$SSH_ENV" > /dev/null
        if ! kill -0 "$SSH_AGENT_PID" 2>/dev/null; then
            (umask 066; ssh-agent -a "$HOME/.ssh/ssh-agent.sock" > "$SSH_ENV")
            source "$SSH_ENV"
            for key in ~/.ssh/id_*; do
                [[ -f "$key" && "$key" != *.pub ]] && ssh-add "$key" 2>/dev/null
            done
        fi
    else
        (umask 066; ssh-agent -a "$HOME/.ssh/ssh-agent.sock" > "$SSH_ENV")
        source "$SSH_ENV"
        for key in ~/.ssh/id_*; do
            [[ -f "$key" && "$key" != *.pub ]] && ssh-add "$key" 2>/dev/null
        done
    fi
    export SSH_AUTH_SOCK_DIR="$(dirname "$SSH_AUTH_SOCK")"
fi

if [[ -z "$SSH_AUTH_SOCK" ]]; then
    print_error "Failed to set up SSH agent. SSH forwarding will not work."
    print_error "Please check SSH setup and try again."
    exit 1
else
    print_success "SSH agent configured successfully"
    print_success "SSH_AUTH_SOCK: $SSH_AUTH_SOCK"
    echo -e "${YELLOW}SSH keys loaded:${NC}"
    ssh-add -l 2>/dev/null || echo "  No keys loaded"
fi

# ============================================================================
# STEP 2: Create Project Structure
# ============================================================================
print_header "Step 2: Creating Project Structure"

AM_BASE_DIR="$HOME/AM"
WORK_DIR="$AM_BASE_DIR/am_npi_docker"
mkdir -p "$WORK_DIR/.devcontainer"
cd "$WORK_DIR"
print_success "Created work directory: $WORK_DIR"

# ============================================================================
# STEP 3: Create Dockerfile
# ============================================================================
print_header "Step 3: Creating Dockerfile"

cat > "$WORK_DIR/.devcontainer/Dockerfile" << 'EOF'
FROM registry-sc-harbor.amd.com/rocm-ci-images/compute-rocm-npi-mi450:latest-ubuntu-24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN pip config set global.break-system-packages true

WORKDIR /root
EOF

print_success "Dockerfile created at $WORK_DIR/.devcontainer/Dockerfile"

# ============================================================================
# STEP 4: Build Docker Image
# ============================================================================
print_header "Step 4: Building Docker Image"

cd "$WORK_DIR/.devcontainer"
if ! DOCKER_BUILDKIT=0 docker build -t ${CONTAINER_NAME}:latest -f Dockerfile .; then
    print_error "Docker image build failed."
    exit 1
fi
print_success "Docker image built: ${CONTAINER_NAME}:latest"

# ============================================================================
# STEP 5: Run Docker Container
# ============================================================================
print_header "Step 5: Running Docker Container"

echo -e "${BLUE}Pre-check: Verifying SSH setup on host...${NC}"
echo -e "SSH_AUTH_SOCK on host: ${GREEN}$SSH_AUTH_SOCK${NC}"

if ssh-add -l &>/dev/null; then
    print_success "SSH agent has keys loaded"
    echo -e "${YELLOW}SSH keys loaded:${NC}"
    ssh-add -l
else
    SSH_ADD_EXIT=$?
    if [ $SSH_ADD_EXIT -eq 1 ]; then
        print_warning "SSH agent is running but has no keys loaded"
        echo -e "${YELLOW}Loading SSH keys automatically...${NC}"

        KEY_ADDED=false
        for key in ~/.ssh/id_*; do
            if [[ -f "$key" && "$key" != *.pub ]]; then
                echo -e "  Adding key: $key"
                if ssh-add "$key" 2>/dev/null; then
                    KEY_ADDED=true
                    print_success "Added: $key"
                fi
            fi
        done

        if [ "$KEY_ADDED" = true ]; then
            print_success "SSH keys loaded successfully"
            echo -e "${YELLOW}Loaded keys:${NC}"
            ssh-add -l
        else
            print_error "No SSH keys could be added"
            echo -e "${RED}Please run: ${GREEN}ssh-add ~/.ssh/id_rsa${NC} ${RED}\(or your key file\)${NC}"
            read -p "Continue anyway? (y/n): " CONTINUE_NO_SSH
            if [[ $CONTINUE_NO_SSH != [yY] ]]; then
                exit 1
            fi
        fi
    else
        print_error "SSH agent is not accessible"
        echo -e "${RED}Please make sure SSH agent is running and configured${NC}"
        exit 1
    fi
fi
echo ""

SKIP_CONTAINER_CREATION=false

if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    print_warning "Container '$CONTAINER_NAME' already exists"
    read -p "Do you want to stop and remove it? (y/n): " REMOVE_CONTAINER
    if [[ $REMOVE_CONTAINER != [nN] ]]; then
        print_warning "Stopping and removing existing container: $CONTAINER_NAME"
        docker stop $CONTAINER_NAME || true
        docker rm $CONTAINER_NAME || true
        print_success "Container removed"
    else
        CONTAINER_RUNNING=$(docker inspect -f '{{.State.Running}}' $CONTAINER_NAME 2>/dev/null || echo "false")

        read -p "Do you want to skip container creation and proceed with the existing container? (y/n): " SKIP_CREATION
        if [[ $SKIP_CREATION == [yY] ]]; then
            SKIP_CONTAINER_CREATION=true

            if [[ "$CONTAINER_RUNNING" == "false" ]]; then
                print_warning "Starting existing container..."
                docker start $CONTAINER_NAME
                print_success "Container started"
            else
                print_success "Container is already running"
            fi

            print_success "Skipping container creation, will use existing container"
        else
            print_error "Cannot proceed with existing container. Please remove it manually or choose a different name."
            exit 1
        fi
    fi
fi

if [[ $SKIP_CONTAINER_CREATION == false ]]; then
    RENDER_GID=$(getent group render | cut -d: -f3)
    if [[ -z "$RENDER_GID" ]]; then
        print_warning "Render group not found, skipping --group-add render"
        RENDER_GROUP_ARG=""
    else
        RENDER_GROUP_ARG="--group-add $RENDER_GID"
        print_success "Render group ID: $RENDER_GID"
    fi

    DOCKER_RUN_CMD="docker run -dt \
        --name $CONTAINER_NAME \
        --device /dev/kfd --device /dev/dri \
        --group-add video $RENDER_GROUP_ARG \
        --cap-add=SYS_PTRACE \
        --security-opt seccomp=unconfined \
        --shm-size=16gb \
        --ulimit memlock=-1:-1 \
        --ulimit stack=67108864:67108864 \
        -v /tmp/.X11-unix:/tmp/.X11-unix \
        -v $(readlink -f ${SSH_AUTH_SOCK}):${SSH_AUTH_SOCK} \
        -e SSH_AUTH_SOCK=${SSH_AUTH_SOCK} \
        -w /root \
        ${CONTAINER_NAME}:latest \
        /bin/bash"

    print_success "SSH forwarding enabled: $SSH_AUTH_SOCK"

    if ! eval $DOCKER_RUN_CMD; then
        print_error "Failed to start Docker container '$CONTAINER_NAME'."
        exit 1
    fi
    print_success "Docker container started: $CONTAINER_NAME"

    echo -e "${BLUE}Verifying SSH forwarding in container...${NC}"
    docker exec $CONTAINER_NAME bash -c "
        echo 'SSH_AUTH_SOCK: '\$SSH_AUTH_SOCK && \
        if [ -S \"\$SSH_AUTH_SOCK\" ]; then \
            echo '✓ SSH socket file exists and is accessible' && \
            ssh-add -l 2>/dev/null && echo '✓ SSH agent is working' || echo '✗ Cannot list SSH keys'; \
        else \
            echo '✗ SSH socket file not found or not accessible'; \
        fi
    "
fi

# ============================================================================
# STEP 6: Install Prerequisites in Container
# ============================================================================
print_header "Step 6: Installing Prerequisites in Container"

if ! docker exec $CONTAINER_NAME bash -c "
    apt-get update -y && \
    apt-get install -y \
      git clang lld ccache \
      python3 python3-dev python3-pip \
      numactl libelf1 libzstd-dev
"; then
    print_error "Failed to install prerequisites in container."
    exit 1
fi
print_success "Prerequisites installed in container"

# ============================================================================
# STEP 7: Download and Setup AM+FFM-Lite Package
# ============================================================================
print_header "Step 7: Downloading AM+FFM-Lite Package"

AM_URL="${ARTIFACTORY_BASE}/${AM_RELEASE_PATH}/${AM_PACKAGE}"

echo -e "${BLUE}URL: $AM_URL${NC}"
echo -e "${BLUE}Downloading with stored credentials...${NC}\n"

DOWNLOAD_SUCCESS=false
DOWNLOAD_ATTEMPTS=0
while [[ $DOWNLOAD_SUCCESS == false ]]; do
    DOWNLOAD_ATTEMPTS=$((DOWNLOAD_ATTEMPTS + 1))
    if docker exec -e "DL_USER=$NTID" -e "DL_PASS=$AM_PASSWORD" $CONTAINER_NAME bash -c \
        "cd /root && wget --user=\"\$DL_USER\" --password=\"\$DL_PASS\" '$AM_URL'"; then
        DOWNLOAD_SUCCESS=true
        print_success "AM+FFM-Lite package downloaded successfully"
    else
        print_error "Download failed (attempt $DOWNLOAD_ATTEMPTS)"
        read -p "Retry download? (y/n): " RETRY
        if [[ $RETRY != [yY] ]]; then
            print_error "Cannot continue without the AM+FFM-Lite package."
            exit 1
        fi
    fi
done

echo -e "${YELLOW}Extracting package...${NC}"
if ! docker exec $CONTAINER_NAME bash -c "
    cd /root && tar xf $AM_PACKAGE
"; then
    print_error "Failed to extract $AM_PACKAGE."
    exit 1
fi
print_success "AM+FFM-Lite package extracted to /root/$AM_EXTRACT_DIR"

echo -e "${YELLOW}Patching env scripts to include ROCm lib path...${NC}"
docker exec $CONTAINER_NAME bash -c "
    sed -i 's|export LD_LIBRARY_PATH=\"\\\$pkgroot/rocm\\\${LD_LIBRARY_PATH:+:\\\$LD_LIBRARY_PATH}\"|export LD_LIBRARY_PATH=\"\\\$pkgroot/rocm:/opt/rocm-7.3.0/lib\\\${LD_LIBRARY_PATH:+:\\\$LD_LIBRARY_PATH}\"|' /root/$AM_EXTRACT_DIR/ffmlite_env.sh && \
    sed -i 's|export LD_LIBRARY_PATH=\\\$pkgroot/rocm:\\\$pkgroot/package:\\\$pkgroot/package/lib64:\\\$pkgroot/package/bin|export LD_LIBRARY_PATH=\\\$pkgroot/rocm:\\\$pkgroot/package:\\\$pkgroot/package/lib64:\\\$pkgroot/package/bin:/opt/rocm-7.3.0/lib|' /root/$AM_EXTRACT_DIR/am_env.sh
"
print_success "Patched ffmlite_env.sh and am_env.sh with /opt/rocm-7.3.0/lib"

echo -e "${YELLOW}Enabling itrace in am_env.sh...${NC}"
docker exec $CONTAINER_NAME bash -c "
    sed -i 's|#\"test.enable_itrace=true\"|\"test.enable_itrace=true\"|' /root/$AM_EXTRACT_DIR/am_env.sh && \
    sed -i 's|#\"test.itrace_perf_detail=true\"|\"test.itrace_perf_detail=true\"|' /root/$AM_EXTRACT_DIR/am_env.sh
"
print_success "Enabled itrace in am_env.sh"

echo -e "${YELLOW}Ensuring model.gpu.use_hw_registers=true is in am_env.sh...${NC}"
docker exec $CONTAINER_NAME bash -c "
    if ! grep -q '\"model.gpu.use_hw_registers=true\"' /root/$AM_EXTRACT_DIR/am_env.sh; then
        sed -i '/\"model.gpu.compute_only_model=true\"/a\\        \"model.gpu.use_hw_registers=true\"' /root/$AM_EXTRACT_DIR/am_env.sh
        echo 'Added model.gpu.use_hw_registers=true'
    else
        echo 'model.gpu.use_hw_registers=true already present'
    fi
"
print_success "Ensured model.gpu.use_hw_registers=true in am_env.sh"

echo -e "${YELLOW}Adding TRITON_GFX1250_MODEL_PATH to container's .bashrc...${NC}"
docker exec $CONTAINER_NAME bash -c "
    echo 'export TRITON_GFX1250_MODEL_PATH=/root/$AM_EXTRACT_DIR' >> /root/.bashrc
"
print_success "TRITON_GFX1250_MODEL_PATH set in container's .bashrc"

# ============================================================================
# STEP 8: Sanity Tests
# ============================================================================
print_header "Step 8: Sanity Tests"

echo -e "\n${YELLOW}Running FFM test...${NC}"
if docker exec $CONTAINER_NAME bash -c "
    cd /root/$AM_EXTRACT_DIR && \
    source ffmlite_env.sh && \
    ./tests/Histogram
"; then
    print_success "FFM Histogram test completed"
else
    print_warning "FFM Histogram test failed."
fi

echo -e "\n${YELLOW}Running AM test...${NC}"
if docker exec $CONTAINER_NAME bash -c "
    cd /root/$AM_EXTRACT_DIR && \
    source am_env.sh && \
    ./tests/Histogram
"; then
    print_success "AM Histogram test completed"
else
    print_warning "AM Histogram test failed."
fi

# ============================================================================
# STEP 9: Setup Triton Environment (NPI PyTorch)
# ============================================================================
print_header "Step 9: Setting up Triton Environment (NPI PyTorch)"

# --- 9a: Install hip-python --- might not need
echo -e "${YELLOW}Installing hip-python...${NC}"
if ! docker exec $CONTAINER_NAME bash -c "
    pip install -i https://test.pypi.org/simple/ hip-python
"; then
    print_error "Failed to install hip-python."
    exit 1
fi
print_success "hip-python installed"

# --- 9b: Auto-detect latest NPI PyTorch build and install ---
NPI_TORCH_BASE="https://compute-artifactory.amd.com/artifactory/compute-pytorch-rocm/compute-rocm-npi-mi450"
echo -e "${YELLOW}Detecting latest NPI PyTorch build...${NC}"

ROCM_BUILD_NUM=$(curl -s "${NPI_TORCH_BASE}/" | grep -oP 'href="\K[0-9]+' | sort -n | tail -1)

if [[ -z "$ROCM_BUILD_NUM" ]]; then
    print_error "Failed to auto-detect latest NPI PyTorch build number."
    print_error "Check connectivity to: $NPI_TORCH_BASE"
    exit 1
fi

NPI_TORCH_URL="${NPI_TORCH_BASE}/${ROCM_BUILD_NUM}/mi450"
print_success "Latest NPI build detected: $ROCM_BUILD_NUM"
echo -e "${YELLOW}Installing NPI PyTorch from build $ROCM_BUILD_NUM...${NC}"
echo -e "${BLUE}URL: $NPI_TORCH_URL${NC}"

if ! docker exec $CONTAINER_NAME bash -c "
    URL_BASE='$NPI_TORCH_URL' && \
    TORCH_WHL=\$(curl -s \${URL_BASE}/ | grep -oP 'torch-[^\"]*\\.whl' | head -n1) && \
    echo \"Found torch wheel: \$TORCH_WHL\" && \
    pip install --no-deps \${URL_BASE}/\${TORCH_WHL} && \
    pip install filelock typing-extensions sympy networkx jinja2 fsspec
"; then
    print_error "Failed to install NPI PyTorch."
    print_error "Check that build $ROCM_BUILD_NUM has PyTorch wheels at:"
    print_error "  $NPI_TORCH_URL"
    exit 1
fi
print_success "NPI PyTorch installed (build $ROCM_BUILD_NUM)"

echo -e "${BLUE}Verifying PyTorch installation:${NC}"
docker exec $CONTAINER_NAME bash -c "pip show torch | head -5"

# --- 9c: Cleanup PyTorch bundled packages ---
echo -e "\n${YELLOW}Cleaning up PyTorch-bundled triton and HIP runtime...${NC}"
docker exec $CONTAINER_NAME bash -c "
    pip uninstall -y triton pytorch-triton pytorch-triton-rocm 2>/dev/null || true && \
    TORCH_LIB=\$(pip show torch | grep '^Location:' | cut -d' ' -f2)/torch/lib && \
    rm -f \${TORCH_LIB}/libamdhip64.so && \
    echo 'Removed libamdhip64.so from PyTorch lib dir'
"
print_success "PyTorch cleanup complete"

# --- 9d: Clone and build Triton ---
CLONE_OK=true

if [[ "$CLONE_METHOD" == "ssh" ]]; then
    TRITON_REPO="git@github.amd.com:GFX-IP-Arch/triton.git"
    AITER_REPO="https://github.com/ROCm/aiter.git"
    print_success "Using SSH for Triton clone"

    echo -e "${YELLOW}Setting up SSH in container for github.amd.com...${NC}"
    docker exec $CONTAINER_NAME bash -c "
        mkdir -p ~/.ssh && \
        chmod 700 ~/.ssh && \
        ssh-keyscan github.amd.com >> ~/.ssh/known_hosts 2>/dev/null
    "

    echo -e "${YELLOW}Testing SSH connection to github.amd.com...${NC}"
    if docker exec $CONTAINER_NAME bash -c "ssh -T git@github.amd.com 2>&1" | grep -q "successfully authenticated"; then
        print_success "SSH connection successful"
    else
        print_error "SSH connection failed"
        echo -e "${RED}Troubleshooting:${NC}"
        echo -e "${YELLOW}1. On host, verify SSH agent: ${GREEN}ssh-add -l${NC}"
        echo -e "${YELLOW}2. On host, check SSH_AUTH_SOCK: ${GREEN}echo \$SSH_AUTH_SOCK${NC}"
        print_warning "Skipping Triton and AITER clone/build -- you can do this manually later."
        CLONE_OK=false
    fi
else
    TRITON_REPO="https://${GIT_PAT}@github.amd.com/GFX-IP-Arch/triton.git"
    AITER_REPO="https://github.com/ROCm/aiter.git"
    print_success "Using HTTPS with PAT for Triton clone"
fi

if [[ $CLONE_OK == true ]]; then

if docker exec $CONTAINER_NAME bash -c "[ -d /root/triton-dev ]"; then
    print_warning "Triton directory already exists at /root/triton-dev - skipping clone"
else
    echo -e "${YELLOW}Cloning Triton repository...${NC}"
    if ! docker exec $CONTAINER_NAME bash -c "
        cd /root && git clone '$TRITON_REPO' triton-dev
    "; then
        print_error "Failed to clone Triton repository"
        exit 1
    fi
    print_success "Triton cloned to /root/triton-dev"
fi

echo -e "${YELLOW}Installing Triton requirements and dependencies...${NC}"
if ! docker exec $CONTAINER_NAME bash -c "
    cd /root/triton-dev && \
    pip install -r python/requirements.txt && \
    pip install -U nanobind \
      numpy scipy pandas matplotlib einops \
      pytest pytest-xdist pytest-repeat expecttest \
      pylama pre-commit clang-format
"; then
    print_error "Failed to install Triton dependencies."
    exit 1
fi
print_success "Triton dependencies installed"

echo -e "${YELLOW}Building and installing Triton (this may take a while)...${NC}"
if ! docker exec $CONTAINER_NAME bash -c "
    cd /root/triton-dev && \
    pip install -e .
"; then
    print_error "Triton build failed."
    print_error "You can retry manually inside the container:"
    print_error "  docker exec -it $CONTAINER_NAME bash"
    print_error "  cd /root/triton-dev && pip install -e ."
    exit 1
fi
print_success "Triton installed"

# --- 9e: Verify ---
echo -e "\n${BLUE}Verifying Triton installation:${NC}"
docker exec $CONTAINER_NAME bash -c "
    python3 -c 'import triton; print(\"Triton imported successfully\")'
"
print_success "Triton setup complete!"

# ============================================================================
# STEP 9f: Clone and Setup AITER
# ============================================================================
print_header "Step 9f: Setting up AITER (AI Tensor Engine for ROCm)"

if docker exec $CONTAINER_NAME bash -c "[ -d /root/aiter ]"; then
    print_warning "AITER directory already exists at /root/aiter - skipping clone"
else
    echo -e "${YELLOW}Cloning AITER repository...${NC}"
    if ! docker exec $CONTAINER_NAME bash -c "
        cd /root && git clone --recursive '$AITER_REPO'
    "; then
        print_error "Failed to clone AITER repository"
        exit 1
    fi
    print_success "AITER cloned to /root/aiter"

    echo -e "${YELLOW}Switching to shared/triton-gfx12 branch...${NC}"
    if ! docker exec $CONTAINER_NAME bash -c "
        cd /root/aiter && git checkout shared/triton-gfx12 && git pull
    "; then
        print_warning "Failed to switch to shared/triton-gfx12 branch."
        print_warning "You can do this manually: cd /root/aiter && git checkout shared/triton-gfx12 && git pull"
    else
        print_success "AITER switched to shared/triton-gfx12 branch"
    fi
fi

echo -e "${YELLOW}Installing AITER in development mode...${NC}"
if ! docker exec $CONTAINER_NAME bash -c "
    cd /root/aiter && pip install -e .
"; then
    print_warning "AITER install failed. You can retry manually:"
    print_warning "  cd /root/aiter && pip install -e ."
else
    print_success "AITER installed"
fi

echo -e "\n${BLUE}Verifying AITER installation:${NC}"
docker exec $CONTAINER_NAME bash -c "
    python3 -c 'import aiter; print(\"AITER imported successfully\")' 2>/dev/null
" && print_success "AITER setup complete!" || print_warning "AITER import check failed -- may need manual setup"

fi # end CLONE_OK

# ============================================================================
# STEP 10: FFM Dev Flow Verification
# ============================================================================
print_header "Step 10: FFM Dev Flow Verification"

echo -e "${YELLOW}Setting up and verifying the FFM dev flow...${NC}\n"
if docker exec $CONTAINER_NAME bash -c "
    export TRITON_GFX1250_MODEL_PATH=/root/$AM_EXTRACT_DIR && \
    source \$TRITON_GFX1250_MODEL_PATH/ffmlite_env.sh && \
    echo 'LD_LIBRARY_PATH:' && echo \$LD_LIBRARY_PATH && \
    echo '' && \
    echo 'Triton target:' && \
    python3 -c \"import triton; print(triton.runtime.driver.active.get_current_target())\"
"; then
    print_success "FFM dev flow verified"
else
    print_warning "FFM dev flow verification failed."
    print_warning "Expected output: GPUTarget(backend='hip', arch='gfx1250', warp_size=32)"
fi

# ============================================================================
# STEP 11: AM Dev Flow Verification
# ============================================================================
print_header "Step 11: AM Dev Flow Verification"

echo -e "${YELLOW}Setting up and verifying the AM dev flow...${NC}\n"
if docker exec $CONTAINER_NAME bash -c "
    export TRITON_GFX1250_MODEL_PATH=/root/$AM_EXTRACT_DIR && \
    source \$TRITON_GFX1250_MODEL_PATH/am_env.sh && \
    echo 'LD_LIBRARY_PATH:' && echo \$LD_LIBRARY_PATH && \
    echo '' && \
    echo 'Triton target:' && \
    python3 -c \"import triton; print(triton.runtime.driver.active.get_current_target())\"
"; then
    print_success "AM dev flow verified"
else
    print_warning "AM dev flow verification failed."
    print_warning "Expected output: GPUTarget(backend='hip', arch='gfx1250', warp_size=32)"
fi

# ============================================================================
# COMPLETION - Write reminder file and print summary
# ============================================================================

REMINDER_FILE="$HOME/AM_NPI_QUICK_REFERENCE.txt"
cat > "$REMINDER_FILE" << "REMINDER_EOF"
ENTER THE CONTAINER
  docker exec -it CONTAINER_NAME bash

PATHS INSIDE CONTAINER
  AM+FFM package:  /root/AM_EXTRACT_DIR
  Triton source:   /root/triton-dev
  AITER source:    /root/aiter

DAILY DEVELOPMENT - FFM FLOW
  source $TRITON_GFX1250_MODEL_PATH/ffmlite_env.sh
  python3 -c "import triton; print(triton.runtime.driver.active.get_current_target())"
  # Expected: GPUTarget(backend='hip', arch='gfx1250', warp_size=32)

DAILY DEVELOPMENT - AM FLOW
  source $TRITON_GFX1250_MODEL_PATH/am_env.sh
  python3 -c "import triton; print(triton.runtime.driver.active.get_current_target())"
  # Expected: GPUTarget(backend='hip', arch='gfx1250', warp_size=32)

NOTE: TRITON_GFX1250_MODEL_PATH is already set in the container's .bashrc

REMINDER_EOF
sed -i "s|CONTAINER_NAME|$CONTAINER_NAME|g; s|AM_EXTRACT_DIR|$AM_EXTRACT_DIR|g" "$REMINDER_FILE"

print_header "Setup Complete!"

echo -e "${GREEN}Your Triton AM environment (NPI PyTorch) is ready!${NC}"

echo -e "\n${BLUE}────────────────────────────────────────────────${NC}"
echo -e "${BLUE}  Paths Inside Container${NC}"
echo -e "${BLUE}────────────────────────────────────────────────${NC}"
echo -e "  AM+FFM package:  ${GREEN}/root/$AM_EXTRACT_DIR${NC}"
echo -e "  Triton source:   ${GREEN}/root/triton-dev${NC}"
echo -e "  AITER source:    ${GREEN}/root/aiter${NC}"

echo -e "\n${BLUE}────────────────────────────────────────────────${NC}"
echo -e "${BLUE}  Daily Development - FFM Flow${NC}"
echo -e "${BLUE}────────────────────────────────────────────────${NC}"
echo -e "  ${GREEN}source \$TRITON_GFX1250_MODEL_PATH/ffmlite_env.sh${NC}"

echo -e "\n${BLUE}────────────────────────────────────────────────${NC}"
echo -e "${BLUE}  Daily Development - AM Flow${NC}"
echo -e "${BLUE}────────────────────────────────────────────────${NC}"
echo -e "  ${GREEN}source \$TRITON_GFX1250_MODEL_PATH/am_env.sh${NC}"

echo -e "\n${YELLOW}TRITON_GFX1250_MODEL_PATH is already set in the container's .bashrc${NC}"

echo -e "\n${BLUE}Quick reference saved to:${NC} ${GREEN}$REMINDER_FILE${NC}"
echo -e "${YELLOW}Use 'cat $REMINDER_FILE' anytime to see paths and commands.${NC}"

echo ""
