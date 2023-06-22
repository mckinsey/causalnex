FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y && apt install -y python3.8 python3-pip
RUN ln -s $(which python3) /usr/local/bin/python
#COPY requirements.txt test_requirements.txt tmp/
COPY . /tmp
WORKDIR /tmp
# install requirements
#RUN python3 -m pip --no-cache-dir install -r tmp/requirements.txt

# install test requirements which also installs requirements.txt
RUN python3 -m pip --no-cache-dir install -r test_requirements.txt

RUN python3 -m pip --no-cache-dir install ".[pytorch]"
