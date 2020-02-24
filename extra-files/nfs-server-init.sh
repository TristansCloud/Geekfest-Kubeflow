#!/bin/bash

sudo apt-get update
sudo apt install nfs-common
sudo apt install nfs-kernel-server
sudo mkdir /nfsroot

echo "/nfsroot *(rw,no_root_squash,no_subtree_check)" | sudo tee -a /etc/exports

