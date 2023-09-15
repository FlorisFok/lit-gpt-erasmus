FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

# Python
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y python3.9 python3.9-dev
RUN apt-get install -y python3-pip

# Torch with flash attention capabilities
RUN pip install --index-url https://download.pytorch.org/whl/nightly/cu118 --pre 'torch>=2.1.0dev'
RUN pip uninstall ninja -y && pip install ninja

# Modern implementation of transformer
RUN apt-get update && apt-get install -y sudo curl git
RUN pip install -v -U git+https://github.com/facebookresearch/xformers.git@main#egg=xformers

RUN mkdir app
WORKDIR /app

# Modern implemention of Flash Attention
RUN git clone https://github.com/Dao-AILab/flash-attention
WORKDIR /app/flash-attention
RUN pip install --upgrade pip
RUN pip install packaging
RUN python3 setup.py install

WORKDIR /app

COPY requirements.txt .

# Default ligthning
RUN pip install -r requirements.txt tokenizers sentencepiece

# Extra for Ligthning Multi Node
RUN pip install zstandard datasets lightning deepdiff fastapi websockets click lightning_cloud backoff bs4 croniter psutil arrow inquirer

# Faster attention + layernorm + entropy
WORKDIR /app/flash-attention
WORKDIR /app/flash-attention/csrc/rotary
RUN pip install .
WORKDIR /app/flash-attention/csrc/layer_norm
RUN pip install .
WORKDIR /app/flash-attention/csrc/xentropy
RUN pip install .

RUN pip uninstall -y lightning
RUN pip install git+https://github.com/Lightning-AI/lightning.git@c5cb53269465633076ad9220e225336ebb815547

RUN pip install -U deepspeed

COPY . /app
WORKDIR /app

CMD ["echo", "ok"]