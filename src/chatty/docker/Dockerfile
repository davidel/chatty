FROM debian:bookworm-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_BREAK_SYSTEM_PACKAGES=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    ca-certificates \
    curl \
    git \
    build-essential \
    g++ \
    python3 \
    python3-pip \
    python3-venv \
    python3-pytest \
    procps \
    sudo \
    && rm -rf /var/lib/apt/lists/*

RUN echo "ALL ALL=(ALL:ALL) NOPASSWD: ALL" >> /etc/sudoers

WORKDIR /workspace
CMD ["sleep", "infinity"]
