#!/usr/bin/env bash
set -ex

cmd=${@:-/bin/bash}
IMAGE=${IMAGE:-sm}

docker run -it --net=host \
  -v $HOME/distr/sparse-map-workdir:/artefacts \
  -v $HOME/distr/sparse-map/src:/sparse-map-src \
  $IMAGE \
  $cmd
