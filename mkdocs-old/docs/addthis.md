# Add to ml_3.0

change to this:
# (Do not complete) Push model to docker repo

This cannot be done from the Jupyter notebook, and has already been done for you. However if you are interested in how to build a docker image these steps are included and should be completed outside of a docker container. Your Jupyter notebook is running in a docker container. Look up docker in docker to see why running docker commands from within a container is tricky. There is a Kubeflow python library, kubeflow.faring, for building docker images in Jupyter Notebooks, but we won't use this today. Kubeflow.fairing integrates well with GCP kubeflow deployments. Here's the actual dockerfile for the image you will be running, this one using tensorflow v1.11 with the model and model dependencies in the `tf_mnist` directory:


# to ml_4.0

**Step 5** Shutdown `titanic_dataset_ml.ipynb` and exit the JupyterLab browser tab. From the Kubeflow UI, navigate to the pipelines home page. You should be able to see your pipeline listed. The pipeline portion of kubeflow is not yet fully integrated with namespace control, so you may have to switch pages at the bottom to find your pipeline. Once you've found it, click on it.