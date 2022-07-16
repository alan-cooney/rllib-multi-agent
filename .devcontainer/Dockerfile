# See here for image contents: https://github.com/microsoft/vscode-dev-containers/tree/v0.236.0/containers/python-3/.devcontainer/base.Dockerfile

# [Choice] Python version (use -bullseye variants on local arm64/Apple Silicon): 3, 3.10, 3.9, 3.8, 3.7, 3.6, 3-bullseye, 3.10-bullseye, 3.9-bullseye, 3.8-bullseye, 3.7-bullseye, 3.6-bullseye, 3-buster, 3.10-buster, 3.9-buster, 3.8-buster, 3.7-buster, 3.6-buster
ARG VARIANT="3.10-bullseye"
FROM mcr.microsoft.com/vscode/devcontainers/python:0-${VARIANT}

# [Choice] Node.js version: none, lts/*, 16, 14, 12, 10
ARG NODE_VERSION="none"
RUN if [ "${NODE_VERSION}" != "none" ]; then su vscode -c "umask 0002 && . /usr/local/share/nvm/nvm.sh && nvm install ${NODE_VERSION} 2>&1"; fi

# Install dependencies (mostly for pygame)
RUN apt-get update && apt-get install -y \
    libsdl2-mixer-dev  \
    cmake \
    gputils \
    libfreetype6-dev  \
    libjpeg-dev  \
    libportmidi-dev  \
    libsdl2-dev  \
    libsdl2-image-dev \
    libsdl2-ttf-dev  \
    python3-dev  \
    python3-numpy \
    python3-pygame \
    python3-setuptools  \
    swig

# Install setuptools
RUN pip install build setuptools

# Set Python path
ENV PYTHONPATH="/workspaces/rllib-multi-agent"
