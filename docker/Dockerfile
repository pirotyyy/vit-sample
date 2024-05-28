FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV TORCH_HOME /app/.cache

WORKDIR /app

COPY . .

RUN apt update && \
    apt install -y \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    wget \
    curl \
    llvm \
    libncurses5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    libffi-dev \
    liblzma-dev \
    git \
    python3-pip

ARG USER_UID
ARG USER_GID

ARG USER_NAME=pirotyyy
ARG GROUP_NAME=pirotyyy

RUN if ! getent group $USER_GID >/dev/null; then \
    groupadd -g $USER_GID $GROUP_NAME; \
    fi

# ユーザーを作成
RUN useradd -m -u $USER_UID -g $USER_GID -s /bin/bash $USER_NAME

# 初期ユーザーの変更
USER $USER_NAME
ENV PATH /home/$USER_NAME/.local/bin:$PATH

ENV HOME /home/$USER_NAME
ENV PYENV_ROOT $HOME/.pyenv
ENV PATH $PYENV_ROOT/bin:$PATH

ARG PYTHON_VERSION="3.10.14"

RUN git clone https://github.com/pyenv/pyenv.git ~/.pyenv
RUN echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc && \
    echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc && \
    echo 'eval "$(pyenv init --path)"' >> ~/.bashrc
RUN eval "$(pyenv init --path)"

RUN pyenv install $PYTHON_VERSION && \
    pyenv global $PYTHON_VERSION

FROM base AS torch-2.2.2

RUN pip3 install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121
RUN pip3 install ipykernel jupyter tqdm hydra-core omegaconf