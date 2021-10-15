# Using Triton with Inferentia

Triton will soon be usable with [AWS Inferentia](https://aws.amazon.com/machine-learning/inferentia/) and the [Neuron Runtime](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-intro/get-started.html).

## Inferentia backend setup

First step of running Triton with Inferentia is to initiate an AWS Inferentia instance with Deep Learning AMI (tested with Ubuntu 18.04).
`ssh -i <private-key-name>.pem ubuntu@<instance address>`
Note: It is recommended to set your storage space to greater than default value of 110 GiB.

After logging into the inf1* instance, you will need to get access to Github. Follow [steps on Github to set up ssh access](https://docs.github.com/en/authentication/connecting-to-github-with-ssh).
Clone this repo with Github to home repo `/home/ubuntu`.

Then, start the Trtion instance with:
``` 
docker run -v /home/ubuntu/python_backend:/home/ubuntu/python_backend -v /lib/udev:/mylib/udev -v /run:/myrun --shm-size=1g --ulimit memlock=-1 -p 8000:8000 -p 8001:8001 -p 8002:8002 --ulimit stack=67108864 -ti nvcr.io/nvidia/tritonserver:<Triton container version>-py3
```

where `/mylib/udev` and `myrun` are used for Neuron parameter passing. For Triton container version, please refer to [Triton Inference Server Container Release Notes](https://docs.nvidia.com/deeplearning/triton-inference-server/release-notes/index.html). The current build script has been tested with container version `21.09`. 

After logging into the Triton container, go into the `python_backend` folder and run the setup script.
```
source /home/ubuntu/python_backend/inferentia_examples/scripts/setup-pytorch.sh
```
This script will:
1. Setup miniconda enviroment
2. Install necessary dependencies
3. Build `python_backend_stub` and ensure it is correctly build
4. Install `neuron-cc`, the Neuron compiler and `neuron-rtd` the Neuron Runtime

To control the location of the installation, the user can set up the installation path by exporting `INFRENTIA_PATH` to specific path. e.g `export INFRENTIA_PATH=/root`. Default location is `/home/ubuntu`.

If you want to run the script quietly, pipe output to some file
```
source /home/ubuntu/python_backend/inferentia_examples/scripts/setup-pytorch.sh > /dev/null
```
