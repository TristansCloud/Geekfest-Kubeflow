# Pipelines

**Objective:**

  * Become a data scientist! Explore the likelhood of an individual surving the Titanic and develop models. Understand the responsibilities data scientists need to focus on.
  * Understand the necessary parts of a `pipeline.py` script to save and run your pipeline.
  * Submit your Jupyter workflow as a pipeline using Kale.

**Step 1** In your Jupyter Notebook server, navigate to Geekfest-Kubeflow/Titanic/ and open `titanic_dataset_ml.ipynb`

**Step 3** Unleash your inner data scientist! This is a good example of the types of issues data scientists focus on. Try to figure out if you would make it if you were onboard the titanic as you progress through the notebook. This was originally a Kaggle competition and some interesting models have been developed around this dataset.

**Step 4** Try some different parameters in your models. Google the model if you need formatting help. If you want to a real challenge and have docker installed on your local machine, try building your own docker image to run in Katib and autotune these parameters for you. I would suggest building the training and test datasets into the docker image.

 * For Random Forest try:

```python
RandomForestClassifier(
    n_estimators=100, # The number of trees in the forest. try 1 - 200.
    criterion='gini', # try 'gini' or 'entropy'
    max_depth=None # try 1 - 20
)
```

 * For Logistic Regression:

```python
LogisticRegression(
    solver='lbfgs', # try 'liblinear'
    max_iter=110 # try 10 - 200
)
```

 * For SVM:
 ```python
SVC(
    max_iter=-1 # try -1 and 2 - 100
)
 ```

 * For Decision Tree:

```python
DecisionTreeClassifier(
    max_depth=None, # try between 2 - 20.
    min_samples_split=2 # try larger numbers.
)
```

## Kale

Kale is a Python package that aims to automatically deploy a running Kubeflow Pipelines instance, without requiring Kubeflow DSL.

**Step 1** Switch to the Kubeflow UI tab in your browser, keeping `titanic_dataset_ml.ipynb` running. From the menu navigate to the Pipelines menu. Click on experiments from the lefthand menu.

![experiments](./images/experimentspage.png)

**Step 2** From the top of the page, click + Create experiment. This will be your personal experiment where you will run your pipelines. Name your experiment and click next. From the run page, click `Skip this step` from the bottom of the page.

**Step 3** Go back to `titanic_dataset_ml.ipynb`, click the Kubeflow icon on the menubar on the left side of the screen. This should bring up the **Kale Deployment Panel**. Enable Kale.

![kalepage](./images/kalepage.png)

You should see a few things happen when you enable Kale. On the left hand side you see the Kale Deployment Panel. This is where you can compile pipelines and set up runs directly from your jupyter notebook. You should also see annotations around each cell in your jupyter notebook. The first cell will be labeled `imports`. Click the pencil icon in the top right corner of the imports notebook cell.

**Step 4** Close the cell type dialogue box. Open the loaddata cell type dialogue box by clicking the pencil icon in the top right corner of the loaddata notebook cell. This is the first specified step in the pipeline, the data imports from the previous notebook cell are appended to each step of the pipeline. Close the cell type dialogue box.

![kaledataprocessingcell](./images/kaledataprocessingcell.png)

**Step 5** Scroll through the notebook to the first data processing cell. Click the pencil icon. Notice that this step depends on loaddata completing first. Close the dialogue box and scroll two cells down. Click the pencil icon in that cell. 

![unnamedstep](./images/unnamedstep.png)

You'll see that the pipeline step is unnamed and nothing is indicated in "depends on". This is because the step is unnamed, which causes it to be added to the upstream named pipeline step; in this case, dataprocessing. Click the cell type dropdown menu to see the different annotations available with Kale. Just make sure you return it to an unnamed pipeline step when you're done.

**Step 6** Scroll down the notebook to the ML section. Note that each model is a seperate pipeline step and that each step depends on the feature engineering step. When feature engineering completes, all models will be ran in parallel.

### Deploy a pipeline

**Step 1** Chose your personal experiment under pipeline metadata and name your pipeline.

**Step 2** Click the option to add a volume and switch the volume type from create empty volume to use existing volume. Now your pipeline will have an existing persistant volume mounted into each pod. The persistant volume exists in the kubeflow directory and can be thought of as a central data repo.

Put the mount point as `/mnt/` and the existing volume name as `data`. in the loaddata notebook cell, change the path to 

```python
path = "/mnt/"
```

**Step 3** Click the three dots next to Compile and run and select Compile and save. Then click Compiled and save.

**Step 4** Numerous things happened when you clicked Compiled and save. A `.tar.gz` pipeline was created in the home directory of your Jupyter Notebook server, and a `.kale.py` script of what will be run during the pipeline was created in your working directory. The `.tar.gz` file is a portable version of your kubeflow pipeline that can be transported between across kubeflow deployments. Open the `.kale.py` script and see what was automatically created from your Notebook's script. 

The code for each step of the pipeline is availiable in the `.kale.py` file. The file is broken down into a couple of sections for each step of the pipeline. 

The first step, loaddata, is apparent at the start

```python
def loaddata():

    import os
    import shutil
    from kale.utils import pod_utils
    from kale.marshal import resource_save as _kale_resource_save
    from kale.marshal import resource_load as _kale_resource_load

    _kale_data_directory = "/marshal"

    if not os.path.isdir(_kale_data_directory):
        os.makedirs(_kale_data_directory, exist_ok=True)

    import numpy as np
    import pandas as pd
    import seaborn as sns
    from matplotlib import pyplot as plt
    from matplotlib import style

    from sklearn import linear_model
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import Perceptron
    from sklearn.linear_model import SGDClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.naive_bayes import GaussianNB

    path = "data/"

    PREDICTION_LABEL = 'Survived'

    test_df = pd.read_csv(path + "test.csv")
    train_df = pd.read_csv(path + "train.csv")
```

You can see that the imports that you set as the first cell in `titanic_dataset_ml.ipynb` are appended to the loaddata definition. The next portion saves any arguments, variables, data, etc.. loaded during the pipeline into a persistant volume shared across pipeline steps.

```python
    # -----------------------DATA SAVING START---------------------------------
    if "train_df" in locals():
        _kale_resource_save(train_df, os.path.join(
            _kale_data_directory, "train_df"))
    else:
        print("_kale_resource_save: `train_df` not found.")
    if "test_df" in locals():
        _kale_resource_save(test_df, os.path.join(
            _kale_data_directory, "test_df"))
    else:
        print("_kale_resource_save: `test_df` not found.")
    if "PREDICTION_LABEL" in locals():
        _kale_resource_save(PREDICTION_LABEL, os.path.join(
            _kale_data_directory, "PREDICTION_LABEL"))
    else:
        print("_kale_resource_save: `PREDICTION_LABEL` not found.")
    # -----------------------DATA SAVING END-----------------------------------
```

The next step in the pipeline begins a bit differently, as it depends on a previous step and therefore needs to load any variables or data saved by the previous step. This is accomplished by:

```python
def datapreprocessing():

    import os
    import shutil
    from kale.utils import pod_utils
    from kale.marshal import resource_save as _kale_resource_save
    from kale.marshal import resource_load as _kale_resource_load

    _kale_data_directory = "/marshal"

    if not os.path.isdir(_kale_data_directory):
        os.makedirs(_kale_data_directory, exist_ok=True)

    # -----------------------DATA LOADING START--------------------------------
    _kale_directory_file_names = [
        os.path.splitext(f)[0]
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f))
    ]

    if "test_df" not in _kale_directory_file_names:
        raise ValueError("test_df" + " does not exists in directory")

    _kale_load_file_name = [
        f
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f)) and
        os.path.splitext(f)[0] == "test_df"
    ]
    if len(_kale_load_file_name) > 1:
        raise ValueError("Found multiple files with name " +
                         "test_df" + ": " + str(_kale_load_file_name))
    _kale_load_file_name = _kale_load_file_name[0]
    test_df = _kale_resource_load(os.path.join(
        _kale_data_directory, _kale_load_file_name))

    if "train_df" not in _kale_directory_file_names:
        raise ValueError("train_df" + " does not exists in directory")

    _kale_load_file_name = [
        f
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f)) and
        os.path.splitext(f)[0] == "train_df"
    ]
    if len(_kale_load_file_name) > 1:
        raise ValueError("Found multiple files with name " +
                         "train_df" + ": " + str(_kale_load_file_name))
    _kale_load_file_name = _kale_load_file_name[0]
    train_df = _kale_resource_load(os.path.join(
        _kale_data_directory, _kale_load_file_name))
    # -----------------------DATA LOADING END----------------------------------
```

This portion containerizes each step:

```python
loaddata_op = comp.func_to_container_op(
    loaddata, base_image='gcr.io/secretlab-199414/scotttestimage:1.0.1')


datapreprocessing_op = comp.func_to_container_op(
    datapreprocessing, base_image='gcr.io/secretlab-199414/scotttestimage:1.0.1')


featureengineering_op = comp.func_to_container_op(
    featureengineering, base_image='gcr.io/secretlab-199414/scotttestimage:1.0.1')


decisiontree_op = comp.func_to_container_op(
    decisiontree, base_image='gcr.io/secretlab-199414/scotttestimage:1.0.1')


svm_op = comp.func_to_container_op(
    svm, base_image='gcr.io/secretlab-199414/scotttestimage:1.0.1')


naivebayes_op = comp.func_to_container_op(
    naivebayes, base_image='gcr.io/secretlab-199414/scotttestimage:1.0.1')


logisticregression_op = comp.func_to_container_op(
    logisticregression, base_image='gcr.io/secretlab-199414/scotttestimage:1.0.1')


randomforest_op = comp.func_to_container_op(
    randomforest, base_image='gcr.io/secretlab-199414/scotttestimage:1.0.1')


results_op = comp.func_to_container_op(
    results, base_image='gcr.io/secretlab-199414/scotttestimage:1.0.1')
```

And this defines the actual pipeline. Note the use of `after()` to signify which step comes before:

```python
@dsl.pipeline(
    name='titanic-ml-does-it-show-l0ff3',
    description='Predict which passengers survived the Titanic shipwreck'
)
def auto_generated_pipeline():
    pvolumes_dict = OrderedDict()

    marshal_vop = dsl.VolumeOp(
        name="kale_marshal_volume",
        resource_name="kale-marshal-pvc",
        modes=dsl.VOLUME_MODE_RWM,
        size="1Gi"
    )
    pvolumes_dict['/marshal'] = marshal_vop.volume

    loaddata_task = loaddata_op()\
        .add_pvolumes(pvolumes_dict)\
        .after()
    loaddata_task.container.working_dir = "/home/jovyan/examples/titanic-ml-dataset"
    loaddata_task.container.set_security_context(
        k8s_client.V1SecurityContext(run_as_user=0))

    datapreprocessing_task = datapreprocessing_op()\
        .add_pvolumes(pvolumes_dict)\
        .after(loaddata_task)
    datapreprocessing_task.container.working_dir = "/home/jovyan/examples/titanic-ml-dataset"
    datapreprocessing_task.container.set_security_context(
        k8s_client.V1SecurityContext(run_as_user=0))

    featureengineering_task = featureengineering_op()\
        .add_pvolumes(pvolumes_dict)\
        .after(datapreprocessing_task)
    featureengineering_task.container.working_dir = "/home/jovyan/examples/titanic-ml-dataset"
    featureengineering_task.container.set_security_context(
        k8s_client.V1SecurityContext(run_as_user=0))

    decisiontree_task = decisiontree_op()\
        .add_pvolumes(pvolumes_dict)\
        .after(featureengineering_task)
    decisiontree_task.container.working_dir = "/home/jovyan/examples/titanic-ml-dataset"
    decisiontree_task.container.set_security_context(
        k8s_client.V1SecurityContext(run_as_user=0))

    svm_task = svm_op()\
        .add_pvolumes(pvolumes_dict)\
        .after(featureengineering_task)
    svm_task.container.working_dir = "/home/jovyan/examples/titanic-ml-dataset"
    svm_task.container.set_security_context(
        k8s_client.V1SecurityContext(run_as_user=0))

    naivebayes_task = naivebayes_op()\
        .add_pvolumes(pvolumes_dict)\
        .after(featureengineering_task)
    naivebayes_task.container.working_dir = "/home/jovyan/examples/titanic-ml-dataset"
    naivebayes_task.container.set_security_context(
        k8s_client.V1SecurityContext(run_as_user=0))

    logisticregression_task = logisticregression_op()\
        .add_pvolumes(pvolumes_dict)\
        .after(featureengineering_task)
    logisticregression_task.container.working_dir = "/home/jovyan/examples/titanic-ml-dataset"
    logisticregression_task.container.set_security_context(
        k8s_client.V1SecurityContext(run_as_user=0))

    randomforest_task = randomforest_op()\
        .add_pvolumes(pvolumes_dict)\
        .after(featureengineering_task)
    randomforest_task.container.working_dir = "/home/jovyan/examples/titanic-ml-dataset"
    randomforest_task.container.set_security_context(
        k8s_client.V1SecurityContext(run_as_user=0))

    results_task = results_op()\
        .add_pvolumes(pvolumes_dict)\
        .after(randomforest_task, logisticregression_task, naivebayes_task, svm_task, decisiontree_task)
    results_task.container.working_dir = "/home/jovyan/examples/titanic-ml-dataset"
    results_task.container.set_security_context(
        k8s_client.V1SecurityContext(run_as_user=0))
```

Finally, under ```if __name__ == "__main__":``` the last section of code has two important sections. The first saves a .tar.gz version of the pipeline.

```python
    pipeline_func = auto_generated_pipeline
    pipeline_filename = pipeline_func.__name__ + '.pipeline.tar.gz'
    import kfp.compiler as compiler
    compiler.Compiler().compile(pipeline_func, pipeline_filename)
```

The next two sections find your experiment and submit the pipeline.

```python
    # Get or create an experiment and submit a pipeline run
    import kfp
    client = kfp.Client()
    experiment = client.create_experiment('g')

    # Submit a pipeline run
    run_name = 'titanic-ml-does-it-show-l0ff3_run'
    run_result = client.run_pipeline(
        experiment.id, run_name, pipeline_filename, {})
```


If you wanted to specify a pipline that uses different images for is steps or needs some other functionality, you can edit your `.kale.py` file and execute it.


**Step 5** Shutdown `titanic_dataset_ml.ipynb` and exit the JupyterLab browser tab. From the Kubeflow UI, navigate to the pipelines home page. You should be able to see your pipeline listed. Click on it.

**Step 6** Clicking your pipeline will bring you to a graph showing each step. `kale-marshal-volume` is a volume that gets created and shared among pipeline steps. 
Click on each step for more information.

**Step 7** Click Create experiment in the top right corner. Give your experiment a unique name, then click next.

**Step 8** Name your run. If you've configured your model to accept kubeflow arguments then you can specify their values under run parameters. Specifying run parameters is a good challenge for anyone looking to apply what they learned in the previous section of the workshop. Here is also where you could schedule your pipeline as a CronJob. Click start when you are ready.

**Step 9** Under the experiments page find your experiment and click on your run. You will see a dynamic graph that is filled in as pipeline steps complete. Click on each step and go to logs to watch the pipeline step complete.

**Step 10** Go to the logs of the results pipeline step. You will see the performance of the various models.


### Further examples

The taxicab example uses Tensorflow Model Analysis. See if you can debug the missing modules from the import step (you'll have to restart the notebook kernel). The taxicab pipeline uses the functions option of Kale to define some arguments that are passed to each step of the pipeline. You'll also see the pipeline parameter cell, which uses argparse as a backend to pass configurable parameters at the time of pipeline execution. The values for each parameter that are set in the notebook are used as the default parameters for the pipeline.