FROM asnpdsacr.azurecr.io/public/tritonserver:23.05-tf2-python-py3

#RUN DEBIAN_FRONTEND="noninteractive" apt-get update && apt-get -y install tzdata

RUN apt-get update \
  && apt-get install -y build-essential \
      gcc \
      g++ \
      gdb \
      clang \
      make \
      ninja-build \
      cmake \
      autoconf \
      automake \
      libtool \
      valgrind \
      locales-all \
      dos2unix \
      rsync \
      tar
RUN apt-get install -y python3-pip python3.10-dev
RUN apt-get install -y rapidjson-dev libarchive-dev zlib1g-dev
RUN apt-get install -y git
RUN pip3 install numpy
RUN rm -r /opt/tritonserver/backends/python
RUN git config --global --add safe.directory '*'
RUN apt-get install -y ssh

RUN useradd -m user && yes password | passwd user
RUN apt-get install gdbserver