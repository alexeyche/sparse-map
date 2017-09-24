#!/usr/bin/env bash
set -ex

cmd=${@:-/bin/bash}
IMAGE=${IMAGE:-sm}

docker run -it --net=host -v $HOME/prog/sparse-map-workdir:/artefacts $IMAGE $cmd
