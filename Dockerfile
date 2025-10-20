FROM nvidia/cuda:12.1.1-runtime-ubuntu20.04

RUN apt update && apt install -y git wget python3 python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

CMD ["bash"]