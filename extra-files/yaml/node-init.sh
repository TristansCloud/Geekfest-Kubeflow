#!/bin/bash

# Change the ssh key used.
# echo "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQD3EcwZDKUVK7p9JwL7URvLzoibkcsE4rdV58WneIs4gUtlkb1G4CTmGyULN4l+1a/eyhd38ouaCyUyplO9H5lojcKlpjp3tzGFW3Ebt56QEF5n2/GKBf42mKbznQzgaT8xHLCJvlZruitML2hwnDkaL6JFuu+v1ibF4xPDKuLCLzEYIatB7lNPRKWrkdOezr/VRzxbpuufxwTYCzWUVJZFhBGPCJI6tfZj13DjjwLnLDqEocI/3cjBr1T81xSws/E9Iy4D1XF9D8T2gHTo15T55b2cTTU3Gc2pTIV1GBtDTfWgFHVMutg6vvInx19rypkNBORNSUGUN6Gn6y3HRuTB cca-user@visualization-nfs" >> ~/.ssh/authorized_keys

#Install Docker

curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo apt-key fingerprint 0EBFCD88
sudo add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"

sudo apt-get update -y
#You can add a docker specific version. At the time, this is the most up to date docker version that RKE supports

VERSION_STRING=5:18.09.0~3-0~ubuntu-bionic

sudo apt-get install -y --allow-downgrades docker-ce=$VERSION_STRING docker-ce-cli=$VERSION_STRING containerd.io

#Docker permissions
sudo usermod -aG docker cca-user
sudo usermod -aG sudo cca-user

#nfs
sudo apt install -y nfs-common

