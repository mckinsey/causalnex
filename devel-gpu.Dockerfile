FROM nvidia/cuda:11.7.0-base-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y && apt install -y python3.8 python3-pip libgraphviz-dev graphviz
RUN ln -s $(which python3) /usr/local/bin/python
# Copy all files to container as intalling .[pytorch] requires setup.py, which requires other files
COPY . /tmp
WORKDIR /tmp

# install test requirements which also installs requirements.txt
RUN python3 -m pip --no-cache-dir install -r test_requirements.txt

RUN python3 -m pip --no-cache-dir install ".[pytorch]"
