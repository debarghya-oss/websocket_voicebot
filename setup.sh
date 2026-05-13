#!/bin/bash

# to be run on ubuntu 24.04+ 

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