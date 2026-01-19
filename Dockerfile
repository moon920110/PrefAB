FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

LABEL maintainer="supermoon <super_moon@gm.gist.ac.kr>"
ENV DEBIAN_FRONTEND=noninteractive
#ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    vim \
    tmux \
    htop \
    build-essential \
    libaio-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

RUN pip install --upgrade pip

RUN pip install \
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
    tensorboard

RUN pip install -U \
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

RUN pip install flash-attn --no-build-isolation
RUN python -m ipykernel install --user --name=docker_env --display-name "Python (Docker)"

CMD ["/bin/bash"]