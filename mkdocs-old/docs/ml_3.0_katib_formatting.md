# Understanding Katib formatting

**Objectives:**

  * Understand how to annotate a typical tensorflow model to be compatable with Katib hyperparameter tuning
  * Experiment with the hyperparameter search space and optimization algorithms. 

**Step 1** Go to Geekfest-Kubeflow/Katib-formatting and open `model.ipynb`. Katib will run a `.py` version of this model, but for now you'll interact with the model
through a notebook.

The argparse portion at the bottom of the script is what allows Katib to pass hyperparameter suggestions during a trial. Take a peek at the argparse documentation https://docs.python.org/3/library/argparse.html.

```python
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--fake_data', nargs='?', const=True, type=bool,
                      default=False,
                      help='If true, uses fake data for unit testing.')
  parser.add_argument('--max_steps', type=int, default=100,
                      help='Number of steps to run trainer.')
  parser.add_argument('--learning_rate', type=float, default=0.01,
                      help='Initial learning rate')
  parser.add_argument('--dropout', type=float, default=0.9,
                      help='Keep probability for training dropout.')
  parser.add_argument('--batch_size', type=int, default=100,
                      help='Training batch size')
  parser.add_argument(
      '--data_dir',
      type=str,
      default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
                           'tensorflow/mnist/input_data'),
      help='Directory for storing input data')
  parser.add_argument(
      '--log_dir',
      type=str,
      default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
                           'tensorflow/mnist/logs/mnist_with_summaries'),
      help='Summaries log directory')
  FLAGS, unparsed = parser.parse_known_args()
  tf.compat.v1.app.run(main=main, argv=[sys.argv[0]] + unparsed)
```
This line, `FLAGS, unparsed = parser.parse_known_args()` sets FLAGS as the argument to call when referencing your external arguments

At numerous places in the script these arguements are called through `FLAGS`.

```python

def train():
  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir,
                                    fake_data=FLAGS.fake_data)
.
.
.
  with tf.compat.v1.name_scope('train'):
    train_step = tf.compat.v1.train.AdamOptimizer(FLAGS.learning_rate).minimize(
        cross_entropy)
.
.
.
  train_writer = tf.compat.v1.summary.FileWriter(FLAGS.log_dir + '/train',
                                                 sess.graph)
  test_writer = tf.compat.v1.summary.FileWriter(FLAGS.log_dir + '/test')
.
.
.
    if train or FLAGS.fake_data:
      xs, ys = mnist.train.next_batch(FLAGS.batch_size, fake_data=FLAGS.fake_data)
      k = FLAGS.dropout
    else:
      xs, ys = mnist.test.images, mnist.test.labels
      k = 1.0
    return {x: xs, y_: ys, keep_prob: k}

  for i in range(FLAGS.max_steps):
```

**Step 2** Try running the model and change a few of the hyperparmeters by modifying any of these `default=values`. Look at how they affect model performance:

```python
  parser.add_argument('--max_steps', type=int, default=100,  
                      help='Number of steps to run trainer.')
  parser.add_argument('--learning_rate', type=float, default=0.01,
                      help='Initial learning rate')
  parser.add_argument('--dropout', type=float, default=0.9,
                      help='Keep probability for training dropout.')
  parser.add_argument('--batch_size', type=int, default=100,
                      help='Training batch size')
```

# (Do not complete) Push model to docker repo

This cannot be done from the Jupyter notebook, and has already been done for you. However if you are interested in how to build a docker image these steps are included and should be completed outside of a docker container. Your Jupyter notebook is running in a docker container. Look up docker in docker to see why running docker commands from within a container is tricky. Here's the actual dockerfile, this one using tensorflow v1.11 with the model in the `tf_mnist` directory:

```Dockerfile
FROM tensorflow/tensorflow:1.11.0

ADD . /var/tf_mnist
ENTRYPOINT ["python", "/var/tf_mnist/mnist.py"]
```

To build and push this file to a remote repository (here Docker Hub):

```bash
DOCKER_URL=docker.io/username/mytfmodel:tag # Put your docker registry here
docker build . --no-cache  -f Dockerfile -t ${DOCKER_URL}

docker push ${DOCKER_URL}
```


# Katib hyperparameter tuning

Experiments are the backbone of hyperparameter tuning in Katib. Each experiment runs a number of trials where 
different hyperparameters are tried. Katib currently supports the following tuning algorithms:

* random search
* grid search
* hyperband
* bayesian optimization
* NAS based on reinforcement learning

**Step 1** From the Kubeflow Notebook server page, navigate to Katib. Click on Hyperparameter tuning.

**Step 2** Copy and paste the following yaml into the YAML File page that opens. You could also provide the same information through 
a diffferent UI by clicking Parameters. Rename `{NAME}` to something unique, i.e. firstname-favoriteanimal. Katib is does not yet natively integrate with profiles, so for now your Katib experiment will show up alongside everyone elses. This should be fixed with the upcoming version of Kubeflow.



```yaml
apiVersion: "kubeflow.org/v1alpha3"
kind: Experiment
metadata:
  namespace: kubeflow
  name: {NAME}
spec:
  parallelTrialCount: 1
  maxTrialCount: 10
  maxFailedTrialCount: 3
  objective:
    type: maximize
    goal: 0.999
    objectiveMetricName: accuracy_1
  algorithm:
    algorithmName: random
  metricsCollectorSpec:
    source:
      fileSystemPath:
        path: /train
        kind: Directory
    collector:
      kind: TensorFlowEvent
  parameters:
    - name: --learning_rate
      parameterType: double
      feasibleSpace:
        min: "0.01"
        max: "0.05"
    - name: --batch_size
      parameterType: int
      feasibleSpace:
        min: "50"
        max: "200"
  trialTemplate:
    goTemplate:
        rawTemplate: |-
          apiVersion: "kubeflow.org/v1"
          kind: TFJob
          metadata:
            name: {{.Trial}}
            namespace: {{.NameSpace}}
          spec:
           tfReplicaSpecs:
            Worker:
              replicas: 1 
              restartPolicy: OnFailure
              template:
                spec:
                  containers:
                    - name: tensorflow 
                      resouces:
                        limits:
                          cpu: "1"
                      image: gcr.io/kubeflow-ci/tf-mnist-with-summaries:1.0
                      imagePullPolicy: Always
                      command:
                        - "python"
                        - "/var/tf_mnist/mnist_with_summaries.py"
                        - "--log_dir=/train/metrics"
                        {{- with .HyperParameters}}
                        {{- range .}}
                        - "{{.Name}}={{.Value}}"
                        {{- end}}
                        {{- end}}
```

Edit the yaml to include dropout as a hyperparameter that katib will tune. You can set the feasible space for dropout to between 0.1 and 0.9. Click deploy at the bottom of the page. The parameters specified in this yaml are what will be read through argparse function. Refer back to Understanding Katib formatting to make sure you understand whats going on.

**Step 4** Click the Katib menu browser near the top of the page, then HP (HyperParameter), then monitor. Here you should see your experiment.

![katibmenu](./images/katibmenu.png)

![katiboptions](./images/katiboptions.png)

Click the experiment to see the search results which are added as pods complete. You can highlight certain hyperparameter values by dragging your cursor on the hyperparameter axis to highlight your desired section. You can select combos of hyperparameters by highlighting sections of different hyperparameters. For a graph of model performance across steps click on each trial listed in the experiment. You will need to refresh the page to see updates.

![katibresults](./images/katibresults.png)

![trialdata](./images/trialdata.png)


## Neural Architecture Search

![funnypaper](./images/funnypaper.png)

**Step 1** read the README at https://github.com/kubeflow/katib/tree/master/pkg/suggestion/v1alpha3/NAS_Reinforcement_Learning

***This is to introduce NAS in kubeflow, but you will not run the example***

This is an example yaml for a NAS experiment in katib. ```algorithmSettings``` define the LSTM creating the CNN and ```operations``` defines the search space for model construction. You configure the LSTM under `algorithmSettings:`

```yaml

# This CPU example aims to show all the possible operations
# is not very likely to get good result due to the extensive search space

# In practice, setting up a limited search space with more common operations is more likely to get better performance. 
# For example, Efficient Neural Architecture Search via Parameter Sharing (https://arxiv.org/abs/1802.03268)
# uses only 6 operations, 3x3/5x5 convolution, 3x3/5x5 separable_convolution and 3x3 max_pooling/avg_pooling.

# It uses only 1 layer of CNN and 1 train epoch to show CPU support and it has very bad results.
# In practice, if you increase number of layers, training process on CPU will take more time.

apiVersion: "kubeflow.org/v1alpha3"
kind: Experiment
metadata:
  namespace: kubeflow
  name: nas-rl-example-cpu
spec:
  parallelTrialCount: 2
  maxTrialCount: 3
  maxFailedTrialCount: 2
  objective:
    type: maximize
    goal: 0.99
    objectiveMetricName: Validation-Accuracy
  algorithm:
    algorithmName: nasrl
    algorithmSettings:
      - name: "lstm_num_cells"
        value: "64"
      - name: "lstm_num_layers"
        value: "1"
      - name: "lstm_keep_prob"
        value: "1.0"
      - name: "optimizer"
        value: "adam"
      - name: "init_learning_rate"
        value: "1e-3"
      - name: "lr_decay_start"
        value: "0"
      - name: "lr_decay_every"
        value: "1000"
      - name: "lr_decay_rate"
        value: "0.9"
      - name: "skip-target"
        value: "0.4"
      - name: "skip-weight"
        value: "0.8"
      - name: "l2_reg"
        value: "0"
      - name: "entropy_weight"
        value: "1e-4"
      - name: "baseline_decay"
        value: "0.9999"
  trialTemplate:
    goTemplate:
        rawTemplate: |-
          apiVersion: batch/v1
          kind: Job
          metadata:
            name: {{.Trial}}
            namespace: {{.NameSpace}}
          spec:
            template:
              spec:
                containers:
                - name: {{.Trial}}
                  image: docker.io/kubeflowkatib/nasrl-cifar10-cpu
                  command:
                  - "python3.5"
                  - "-u"
                  - "RunTrial.py"
                  {{- with .HyperParameters}}
                  {{- range .}}
                  - "--{{.Name}}=\"{{.Value}}\""
                  {{- end}}
                  {{- end}}
                  - "--num_epochs=1"
                restartPolicy: Never
  nasConfig:
    graphConfig:
      numLayers: 1
      inputSizes:
        - 32
        - 32
        - 3
      outputSizes:
        - 10
    operations:
      - operationType: convolution
        parameters:
          - name: filter_size
            parameterType: categorical
            feasibleSpace:
              list:
              - "3"
              - "5"
              - "7"
          - name: num_filter
            parameterType: categorical
            feasibleSpace:
              list:
              - "32"
              - "48"
              - "64"
              - "96"
              - "128"
          - name: stride
            parameterType: categorical
            feasibleSpace:
              list:
              - "1"
              - "2"
      - operationType: separable_convolution
        parameters:
          - name: filter_size
            parameterType: categorical
            feasibleSpace:
              list:
              - "3"
              - "5"
              - "7"
          - name: num_filter
            parameterType: categorical
            feasibleSpace:
              list:
              - "32"
              - "48"
              - "64"
              - "96"
              - "128"
          - name: stride
            parameterType: categorical
            feasibleSpace:
              list:
              - "1"
              - "2"
          - name: depth_multiplier
            parameterType: categorical
            feasibleSpace:
              list:
              - "1"
              - "2"
      - operationType: depthwise_convolution
        parameters:
          - name: filter_size
            parameterType: categorical
            feasibleSpace:
              list:
              - "3"
              - "5"
              - "7"
          - name: stride
            parameterType: categorical
            feasibleSpace:
              list:
              - "1"
              - "2"
          - name: depth_multiplier
            parameterType: categorical
            feasibleSpace:
              list:
              - "1"
              - "2"   
      - operationType: reduction
        parameters:
          - name: reduction_type
            parameterType: categorical
            feasibleSpace:
              list:
              - max_pooling
              - avg_pooling
          - name: pool_size
            parameterType: int
            feasibleSpace:
              min: "2"
              max: "3"
              step: "1"
```

This yaml launches pods that execute a trial, each running `RunTrial.py`. Note the use of argparse to pass arguments. The arguments, such as `nn_config` and `architecture` are passed as json strings and are interpreted by the function `ModelConstructor()`.


```python
import keras
import numpy as np
from keras.datasets import cifar10
from ModelConstructor import ModelConstructor
from keras.utils import to_categorical
from keras.utils import multi_gpu_model
from keras.preprocessing.image import ImageDataGenerator
import argparse
import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TrainingContainer')
    parser.add_argument('--architecture', type=str, default="", metavar='N',
                        help='architecture of the neural network')
    parser.add_argument('--nn_config', type=str, default="", metavar='N',
                        help='configurations and search space embeddings')
    parser.add_argument('--num_epochs', type=int, default=10, metavar='N',
                        help='number of epoches that each child will be trained')
    parser.add_argument('--num_gpus', type=int, default=1, metavar='N',
                        help='number of GPU that used for training')
    args = parser.parse_args()

    arch = args.architecture.replace("\'", "\"")
    print(">>> arch received by trial")
    print(arch)

    nn_config = args.nn_config.replace("\'", "\"")
    print(">>> nn_config received by trial")
    print(nn_config)

    num_epochs = args.num_epochs
    print(">>> num_epochs received by trial")
    print(num_epochs)

    num_gpus = args.num_gpus
    print(">>> num_gpus received by trial:")
    print(num_gpus)

# Here the model is built from the architecture and selected hyperparameters passed as arguments

    print("\n>>> Constructing Model...")
    constructor = ModelConstructor(arch, nn_config)
    test_model = constructor.build_model()
    print(">>> Model Constructed Successfully\n")

    if num_gpus > 1:
        test_model = multi_gpu_model(test_model, gpus=num_gpus)
    
    test_model.summary()
    test_model.compile(loss=keras.losses.categorical_crossentropy,
                       optimizer=keras.optimizers.Adam(lr=1e-3, decay=1e-4),
                       metrics=['accuracy'])

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    augmentation = ImageDataGenerator(
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True)
    
    aug_data_flow = augmentation.flow(x_train, y_train, batch_size=128)

    print(">>> Data Loaded. Training starts.")
    for e in range(num_epochs):
        print("\nTotal Epoch {}/{}".format(e+1, num_epochs))
        history = test_model.fit_generator(generator=aug_data_flow,
                                           steps_per_epoch=int(len(x_train)/128)+1,
                                           epochs=1, verbose=1,
                                           validation_data=(x_test, y_test))
        print("Training-Accuracy={}".format(history.history['acc'][-1]))
        print("Training-Loss={}".format(history.history['loss'][-1]))
        print("Validation-Accuracy={}".format(history.history['val_acc'][-1]))
        print("Validation-Loss={}".format(history.history['val_loss'][-1]))
```

`ModelConstructor()` references the following code. See how the same arguments from the yaml are first loaded as json, then used to define the architecture. `ModelConstructor()`in turn calls on `op_library` to construct each layer in Keras.

```python

# ModelConstructor.py

import numpy as np
from keras.models import Model
from keras import backend as K
import json
from keras.layers import Input, Conv2D, ZeroPadding2D, concatenate, MaxPooling2D, \
    AveragePooling2D, Dense, Activation, BatchNormalization, GlobalAveragePooling2D, Dropout
from op_library import concat, conv, sp_conv, dw_conv, reduction


class ModelConstructor(object):
    def __init__(self, arc_json, nn_json):
        self.arch = json.loads(arc_json)
        nn_config = json.loads(nn_json)
        self.num_layers = nn_config['num_layers']
        self.input_sizes = nn_config['input_sizes']
        self.output_size = nn_config['output_sizes'][-1]
        self.embedding = nn_config['embedding']

    def build_model(self):
        # a list of the data all layers
        all_layers = [0 for _ in range(self.num_layers + 1)]
        # a list of all the dimensions of all layers
        all_dims = [0 for _ in range(self.num_layers + 1)]

        # ================= Stacking layers =================
        # Input Layer. Layer 0
        input_layer = Input(shape=self.input_sizes)
        all_layers[0] = input_layer

        # Intermediate Layers. Starting from layer 1.
        for l in range(1, self.num_layers + 1):
            input_layers = list()
            opt = self.arch[l - 1][0]
            opt_config = self.embedding[str(opt)]
            skip = self.arch[l - 1][1:l+1]

            # set up the connection to the previous layer first
            input_layers.append(all_layers[l - 1])

            # then add skip connections
            for i in range(l - 1):
                if l > 1 and skip[i] == 1:
                    input_layers.append(all_layers[i])

            layer_input = concat(input_layers)
            if opt_config['opt_type'] == 'convolution':
                layer_output = conv(layer_input, opt_config)
            if opt_config['opt_type'] == 'separable_convolution':
                layer_output = sp_conv(layer_input, opt_config)
            if opt_config['opt_type'] == 'depthwise_convolution':
                layer_output = dw_conv(layer_input, opt_config)
            elif opt_config['opt_type'] == 'reduction':
                layer_output = reduction(layer_input, opt_config)

            all_layers[l] = layer_output

        # Final Layer
        # Global Average Pooling, then Fully connected with softmax.
        avgpooled = GlobalAveragePooling2D()(all_layers[self.num_layers])
        dropped = Dropout(0.4)(avgpooled)
        logits = Dense(units=self.output_size,
                       activation='softmax')(dropped)

        # Encapsulate the model
        self.model = Model(inputs=input_layer, outputs=logits)

        return self.model
```

Here is `op_library`:

```python
import numpy as np
from keras import backend as K
from keras.layers import Input, Conv2D, ZeroPadding2D, concatenate, MaxPooling2D, \
    AveragePooling2D, Dense, Activation, BatchNormalization, GlobalAveragePooling2D, \
    SeparableConv2D, DepthwiseConv2D


def concat(inputs):
    n = len(inputs)
    if n == 1:
        return inputs[0]

    total_dim = list()
    for x in inputs:
        total_dim.append(K.int_shape(x))
    total_dim = np.asarray(total_dim)
    max_dim = max(total_dim[:, 1])

    padded_input = [0 for _ in range(n)]

    for i in range(n):
        if total_dim[i][1] < max_dim:
            diff = max_dim - total_dim[i][1]
            half_diff = int(diff / 2)
            if diff % 2 == 0:
                padded_input[i] = ZeroPadding2D(padding=(half_diff, half_diff))(inputs[i])
            else:
                padded_input[i] = ZeroPadding2D(padding=((half_diff, half_diff + 1),
                                                         (half_diff, half_diff + 1)))(inputs[i])
        else:
            padded_input[i] = inputs[i]

    result = concatenate(inputs=padded_input, axis=-1)
    return result


def conv(x, config):
    parameters = {
        "num_filter":  64,
        "filter_size":  3,
        "stride":       1,
    }
    for k in parameters.keys():
        if k in config:
            parameters[k] = int(config[k])

    activated = Activation('relu')(x)

    conved = Conv2D(
        filters=parameters['num_filter'],
        kernel_size=parameters['filter_size'],
        strides=parameters['stride'],
        padding='same')(activated)

    result = BatchNormalization()(conved)

    return result


def sp_conv(x, config):
    parameters = {
        "num_filter":       64,
        "filter_size":      3,
        "stride":           1,
        "depth_multiplier": 1,
    }

    for k in parameters.keys():
        if k in config:
            parameters[k] = int(config[k])

    activated = Activation('relu')(x)

    conved = SeparableConv2D(
        filters=parameters['num_filter'],
        kernel_size=parameters['filter_size'],
        strides=parameters['stride'],
        depth_multiplier=parameters['depth_multiplier'],
        padding='same')(activated)

    result = BatchNormalization()(conved)

    return result

def dw_conv(x, config):
    parameters = {
        "filter_size":      3,
        "stride":           1,
        "depth_multiplier": 1,
    }
    for k in parameters.keys():
        if k in config:
            parameters[k] = int(config[k])

    activated = Activation('relu')(x)

    conved = DepthwiseConv2D(
        kernel_size=parameters['filter_size'],
        strides=parameters['stride'],
        depth_multiplier=parameters['depth_multiplier'],
        padding='same')(activated)

    result = BatchNormalization()(conved)

    return result


def reduction(x, config):
    # handle the exteme case where the input has the dimension 1 by 1 and is not reductible
    # we will just change the reduction layer to identity layer
    # such situation is very likely to appear though
    dim = K.int_shape(x)
    if dim[1] == 1 or dim[2] == 1:
        print("WARNING: One or more dimensions of the input of the reduction layer is 1. It cannot be further reduced. A identity layer will be used instead.")
        return x

    parameters = {
        'reduction_type':   "max_pooling",
        'pool_size':        2,
        'stride':           None,
    }

    if 'reduction_type' in config:
        parameters['reduction_type'] = config['reduction_type']
    if 'pool_size' in config:
        parameters['pool_size'] = int(config['pool_size'])
    if 'stride' in config:
        parameters['stride'] = int(config['stride'])

    if parameters['reduction_type'] == 'max_pooling':
        result = MaxPooling2D(
            pool_size=parameters['pool_size'],
            strides=parameters['stride']
        )(x)
    elif parameters['reduction_type'] == 'avg_pooling':
        result = AveragePooling2D(
            pool_size=parameters['pool_size'],
            strides=parameters['stride']
        )(x)

    return result
```

Follow along at https://github.com/kubeflow/katib/tree/master/examples/v1alpha3/NAS-training-containers/RL-cifar10, I'm sure NAS will become much more user friendly as it moves out of alpha.