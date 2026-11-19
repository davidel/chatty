ARG BASE_IMAGE=debian:stable-slim
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_BREAK_SYSTEM_PACKAGES=1

RUN if command -v apt-get >/dev/null 2>&1; then \
      apt-get update && apt-get install -y --no-install-recommends \
        bash ca-certificates curl git build-essential g++ python3 python3-pip python3-venv procps sudo \
        && rm -rf /var/lib/apt/lists/*; \
    elif command -v apk >/dev/null 2>&1; then \
      apk update && apk add --no-cache \
        bash ca-certificates curl git build-base g++ python3 py3-pip procps sudo shadow; \
    elif command -v dnf >/dev/null 2>&1; then \
      dnf -y install \
        bash ca-certificates curl git gcc gcc-c++ make python3 python3-pip procps-ng sudo \
        && dnf clean all; \
    elif command -v yum >/dev/null 2>&1; then \
      yum -y install \
        bash ca-certificates curl git gcc gcc-c++ make python3 python3-pip procps-ng sudo \
        && yum clean all; \
    elif command -v pacman >/dev/null 2>&1; then \
      pacman -Syu --noconfirm \
        bash ca-certificates curl git base-devel python python-pip procps-ng sudo \
        && pacman -Scc --noconfirm; \
    else \
      echo "Error: Unsupported package manager in base image." >&2 && exit 1; \
    fi

RUN mkdir -p /etc && printf "[global]\nbreak-system-packages = true\n" > /etc/pip.conf

RUN python3 -m pip install --no-cache-dir "pytest>=8.0"

ARG USER_ID=1000
ARG GROUP_ID=1000
ARG USER_NAME=chatty

RUN if [ "${USER_ID}" -ne 0 ]; then \
      (groupadd -g ${GROUP_ID} ${USER_NAME} 2>/dev/null || true) && \
      (useradd -u ${USER_ID} -g ${GROUP_ID} -m -s /bin/bash ${USER_NAME} 2>/dev/null || true); \
    fi && \
    echo "ALL ALL=(ALL:ALL) NOPASSWD: ALL" >> /etc/sudoers

ENV PATH="/workspace/.venv/bin:/home/${USER_NAME}/.local/bin:${PATH}"
ENV PYTHONPYCACHEPREFIX=/tmp/pycache

USER ${USER_NAME}

RUN git config --global --add safe.directory "*"

WORKDIR /workspace
CMD ["sleep", "infinity"]
