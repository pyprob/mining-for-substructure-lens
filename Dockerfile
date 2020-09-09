FROM nvidia/cuda:10.2-cudnn7-runtime
    MAINTAINER Andreas Munk <andreas@ammunk.com>

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3 \
    python3-dev \
    git \
    ca-certificates \
    python3-distutils \
    curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
ENV DEBIAN_FRONTEND=noninteractive

ENV PATH="/opt/conda/bin:$PATH"

ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

RUN git clone -b pyprob_implementation https://github.com/pyprob/mining-for-substructure-lens.git /code

RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3 get-pip.py && rm get-pip.py

RUN cd code && pip install -r requirements.txt
RUN cd code && sed -i 's/pipenv run python/python3/g' simple_test_script.bash
