FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

LABEL maintainer="supermoon <super_moon@gm.gist.ac.kr>"
ENV DEBIAN_FRONTEND=noninteractive

# 1. Install system dependencies
# Added: libxcb1, libgl1, and libglib2.0-0 to fix the CV2 ImportError
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    vim \
    tmux \
    htop \
    build-essential \
    libaio-dev \
    libxcb1 \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# 2. Upgrade pip and install core scientific stack
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    numpy \
    pandas \
    scipy \
    scikit-learn \
    matplotlib \
    seaborn \
    tqdm \
    pyyaml \
    h5py \
    pillow \
    requests \
    jupyter \
    ipykernel \
    ipython \
    networkx \
    psutil \
    wandb \
    tensorboard \
    dtw-python \
    tslearn \
    opencv-python  # You can change this to opencv-python-headless to save space

# 3. Install LLM/Deep Learning stack
RUN pip install --no-cache-dir -U \
    transformers \
    accelerate \
    datasets \
    peft \
    bitsandbytes \
    trl \
    einops \
    sentencepiece \
    protobuf \
    evaluate \
    rouge_score

# 4. Install Flash Attention (Building can take a while)
RUN pip install --no-cache-dir flash-attn --no-build-isolation

# 5. Set up Jupyter Kernel
RUN python -m ipykernel install --user --name=docker_env --display-name "Python (Docker)"

CMD ["/bin/bash"]