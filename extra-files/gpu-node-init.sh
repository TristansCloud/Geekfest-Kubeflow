#!/bin/bash
# This "script" works only once, on a new VM. Do not try to run this in prod.
# Do not run this again if it failed the first time.
# This does not contain any checks, error handlings or anything else.i
# requirement : ubuntu 18.04 with an unpartitioned data drive already installed (but not mounted, or formatted)
if [[ $EUID -ne 0 ]]; then
   echo "This script must be run as root" 
   exit 1
fi
apt-get update -y && apt-get upgrade -y && apt-get install -y unzip build-essential apt-transport-https ca-certificates curl gnupg-agent software-properties-common
wget "https://objects-qc.cloud.ca/v1/46ad958dc279492999ea46ea84548477/GPU/NVIDIA-GRID-XenServer-7.x-430.67-430.63-432.08.zip"
unzip NVIDIA-GRID-XenServer-7.x-430.67-430.63-432.08.zip && chmod +x NVIDIA-Linux-x86_64-430.63-grid.run && ./NVIDIA-Linux-x86_64-430.63-grid.run -q -a -n -s
(echo n; echo p; echo 1; echo ""; echo ""; echo w; echo q;) | fdisk /dev/xvdb
mkfs.ext4 /dev/xvdb1 && mkdir /data && mount /dev/xvdb1 /data && mkdir /data/docker-data
disk=$(blkid |grep xvdb1 | awk '{print $2}') && echo "$disk /data ext4 defaults 0 0">>/etc/fstab
ln -s /data/docker-data /var/lib/docker
apt-get remove docker docker-engine docker.io containerd runc
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
apt-get update -y
apt-get install -y docker-ce docker-ce-cli containerd.io
systemctl enable docker
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
apt-get update && sudo apt-get install -y nvidia-container-toolkit nvidia-container-runtime nvidia-docker2
systemctl restart docker
echo "{
    \"default-runtime\": \"nvidia\",
    \"runtimes\": {
        \"nvidia\": {
            \"path\": \"/usr/bin/nvidia-container-runtime\",
            \"runtimeArgs\": []
        }
    }
}" >/etc/docker/daemon.json
echo "
ServerAddress=45.72.188.178
ServerPort=7070
FeatureType=1
" >/etc/nvidia/gridd.conf
systemctl restart docker
systemctl restart nvidia-gridd
