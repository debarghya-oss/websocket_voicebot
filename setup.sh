#!/bin/bash

# to be run on ubuntu 24.04+ if you are running this codebase on bare metal
# if you are running the code through a docker container, you can skip this setup and use the provided Dockerfile instead
# We highly recommend using the docker container for ease of use and to avoid any potential issues with dependencies and compatibility, but if you prefer to run it on bare metal, this setup script will help you get everything installed and configured properly.

#get system up to date
sudo apt update
sudo apt install -y software-properties-common

# add python repo for getting python 3.12
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update

#install python 3.12 and venv and dev packages
sudo apt install -y python3.12 python3.12-venv python3.12-dev

# install portaudio10-dev and ffmpeg for audio processing
sudo apt install -y portaudio19-dev ffmpeg