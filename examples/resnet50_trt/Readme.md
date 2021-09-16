Description:
This is an example to show how to use Python Backend to preprocess your image before using TensorRT to do inference. There will be an ensemble model include preprocess and resnet50_trt.

You need to install numpy, pillow and torchvision by "pip install numpy pillow torchvision".

Your model_repository should have a structure like this:
     
     ensemble_python_resnet50/1
     ensemble_python_resnet50/config.pbtxt
     preprocess/1
     preprocess/config.pbtxt
     preprocess/model.py
     resnet50_trt/1
     resnet50_trt/config.pbtxt
     resnet50_trt/labels.txt
     resnet50_trt/model.plan
     
The above model.plan file is TensorRT engine file, which can be generated like this:
1. Converting PyTorch Model to ONNX-model
Run onnx_exporter.py to convert ResNet50 PyTorch model to ONNX format. width and height dims are fixed at 224 but dynamic axes arguments for dynamic batch are used. Commands from the 2. and 3. subsections shall be executed within this docker container.

     docker run -it --gpus=all -v $(pwd):/workspace nvcr.io/nvidia/pytorch:21.07-py3 bash
     python onnx_exporter.py --save model.onnx
     
2. Building ONNX-model to TensorRT engine
Set the arguments for enabling fp16 precision --fp16. To enable dynamic shapes use --minShapes, --optShapes, and maxShapes with --explicitBatch:

     trtexec --onnx=model.onnx --saveEngine=./model_repository/resnet50_trt/1/model.plan --explicitBatch --minShapes=input:1x3x224x224 --optShapes=input:1x3x224x224 --maxShapes=input:256x3x224x224 --fp16
     
Start the server:
Under model_repository, run this command to start the server docker container:

     docker run --gpus=all --rm -p8000:8000 -p8001:8001 -p8002:8002 -v$(pwd):/workspace/ -v/$(pwd)/model_repository:/models nvcr.io/nvidia/tritonserver:21.07-py3 tritonserver --model-repository=/models
     
Start the client to test:
Under python_backend/examples/resnet50_trt, run these command to start the client docker container and test it:

     wget https://raw.githubusercontent.com/triton-inference-server/server/master/qa/images/mug.jpg -O "mug.jpg"
     docker run --rm --net=host -v $(pwd):/workspace/ nvcr.io/nvidia/tritonserver:21.07-py3-sdk python client.py --image mug.jpg 
     0.22642782455176646ms class:COFFEE MUG
     
     
     
