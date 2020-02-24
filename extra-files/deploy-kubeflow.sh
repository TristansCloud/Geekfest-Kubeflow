#!/bin/bash


# The order of operations for deploying kubeflow:
#   1. Setup master and worker nodes with node-init.sh.
#   2. Setup NFS server for dynamic volume provisioning with nfs-server-init.sh.
#   3. Deploy cluster, IMPORTANT: Kubeflow is not compatible with k8s v1.16 or greater. This install was tested on k8s v1.15.9.
#   4. Wherever kubectl is configured, run helm-install-nfs.sh. This will install heml and configure dynamic volume provisioning through NFS.
#   5. Run this script (deploy-kubeflow.sh)
#       * This script requires kfctl to be setup https://www.kubeflow.org/docs/started/k8s/. For vanilla install follow kfctl_k8s_istio.



sudo mkdir /opt/kubeflow-deployment

#export CONFIG_URI="https://raw.githubusercontent.com/kubeflow/manifests/v0.7-branch/kfdef/kfctl_k8s_istio.0.7.1.yaml" # update this to whichever version you are using
# this version had some issues where I had to manually download the manifest file (manifests-0.7-branch.tar.gz) and sudo tar -xvf 
# This didn't work for v0.7.1.


