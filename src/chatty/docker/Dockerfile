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
    procps \
    sudo \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /etc && printf "[global]\nbreak-system-packages = true\n" > /etc/pip.conf

RUN pip install --no-cache-dir "pytest>=8.0"

ARG USER_ID=1000
ARG GROUP_ID=1000
ARG USER_NAME=chatty

RUN if [ "${USER_ID}" -ne 0 ]; then \
      (groupadd -g ${GROUP_ID} ${USER_NAME} 2>/dev/null || true) && \
      (useradd -u ${USER_ID} -g ${GROUP_ID} -m -s /bin/bash ${USER_NAME} 2>/dev/null || true); \
    fi && \
    echo "ALL ALL=(ALL:ALL) NOPASSWD: ALL" >> /etc/sudoers

ENV PATH="/workspace/.venv/bin:/home/${USER_NAME}/.local/bin:${PATH}"

USER ${USER_NAME}
WORKDIR /workspace
CMD ["sleep", "infinity"]
