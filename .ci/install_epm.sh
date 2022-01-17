#!/usr/bin/env bash

CURRENT_PATH=`pwd`

# delete any previous code
rm -rf dist/EspressoPerformanceMonitor
cd dist

# clone the EPM repository
git clone ssh://git@gitlab.cern.ch:7999/lhcb-ft/EspressoPerformanceMonitor.git
cd EspressoPerformanceMonitor
git checkout v0.81
git submodule update --init --recursive

# make the project
mkdir build && cd build && cmake ..
make -j10

cd $CURRENT_PATH

# vim:foldmethod=marker
