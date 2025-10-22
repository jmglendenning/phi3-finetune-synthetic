FROM nvidia/cuda:12.1.1-runtime-ubuntu20.04

# Install base dependencies
RUN apt update && apt install -y git wget python3 python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python

# Set working directory
WORKDIR /app

# Copy requirements and install with CUDA support
COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --extra-index-url https://download.pytorch.org/whl/cu121 -r requirements.txt

# Copy the full app into the container
COPY . .

# Expose the FastAPI server port
EXPOSE 8081

# Default command to launch FastAPI server (update if you use a different script/command)

CMD ["bash"]