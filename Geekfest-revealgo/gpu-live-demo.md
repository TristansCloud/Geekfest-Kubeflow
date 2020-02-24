Here is the original configuration of the svc/katib-ui:
```yaml
Name:              katib-ui
Namespace:         kubeflow
Labels:            app=katib
                   component=ui
Annotations:       kubectl.kubernetes.io/last-applied-configuration:
                     {"apiVersion":"v1","kind":"Service","metadata":{"annotations":{},"labels":{"app":"katib","component":"ui"},"name":"katib-ui","namespace":"...
Selector:          app=katib,component=ui
Type:              ClusterIP
IP:                10.43.248.123
Port:              ui  80/TCP
TargetPort:        8080/TCP
Endpoints:         10.42.1.10:8080
Session Affinity:  None
Events:            <none>
```

Convert type to NodePort and open port 31000:
```yaml
Name:                     katib-ui
Namespace:                kubeflow
Labels:                   app=katib
                          component=ui
Annotations:              kubectl.kubernetes.io/last-applied-configuration:
                            {"apiVersion":"v1","kind":"Service","metadata":{"annotations":{},"labels":{"app":"katib","component":"ui"},"name":"katib-ui","namespace":"...
Selector:                 app=katib,component=ui
Type:                     NodePort
IP:                       10.43.248.123
Port:                     ui  80/TCP
TargetPort:               8080/TCP
NodePort:                 ui  31000/TCP
Endpoints:                10.42.1.10:8080
Session Affinity:         None
External Traffic Policy:  Cluster
Events:                   <none>
```
<br /><br />

## Start of demo

Login to visualization server `ssh cca-user@10.10.8.44`

```bash
kubectl get nodes
kubectl describe nodes  |  tr -d '\000' | sed -n -e '/^Name/,/Roles/p' -e '/^Capacity/,/Allocatable/p' -e '/^Allocated resources/,/Events/p'  | grep -e Name  -e  nvidia.com  | perl -pe 's/\n//'  |  perl -pe 's/Name:/\n/g' | sed 's/nvidia.com\/gpu:\?//g'  | sed '1s/^/Node Available(GPUs)  Used(GPUs)/' | sed 's/$/ 0 0 0/'  | awk '{print $1, $2, $3}'  | column -t
kubectl get svc -n kubeflow
kubectl get pods -n kubeflow
```

```bash 
cd NAS
vi nasjob.yaml
```
```bash
kubectl apply -f nasjob.yaml
```

View katib UI

10.10.8.124:31000/katib/








Backup yaml:
```yaml
apiVersion: "kubeflow.org/v1alpha3"
kind: Experiment
metadata:
  namespace: kubeflow
  name: demo-example
spec:
  parallelTrialCount: 1
  maxTrialCount: 12
  maxFailedTrialCount: 3
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
                  image: docker.io/kubeflowkatib/nasrl-cifar10-gpu
                  command:
                  - "python3.5"
                  - "-u"
                  - "RunTrial.py"
                  {{- with .HyperParameters}}
                  {{- range .}}
                  - "--{{.Name}}=\"{{.Value}}\""
                  {{- end}}
                  {{- end}}
                  resources:
                    limits:
                      nvidia.com/gpu: 1
                restartPolicy: Never
  nasConfig:
    graphConfig:
      numLayers: 8
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