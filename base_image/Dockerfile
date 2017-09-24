FROM ubuntu

ADD https://bitbucket.org/eigen/eigen/get/3.3.4.tar.bz2 /contrib/eigen.tar.bz2

WORKDIR /contrib

RUN apt-get update
RUN apt-get install -y bzip2
RUN apt-get install -y build-essential cmake
RUN apt-get install -y libblas-dev liblapack-dev
RUN mkdir eigen && tar -xf eigen.tar.bz2 -C eigen
RUN cd eigen; cmake ./eigen*; make install

RUN apt-get install -y git
RUN git clone https://github.com/jbeder/yaml-cpp
RUN cd yaml-cpp; cmake -DBUILD_SHARED_LIBS=ON .; make -j install

ENV LD_LIBRARY_PATH "/usr/local/lib"

RUN apt-get install -y libspdlog-dev
RUN apt-get install -y gdb