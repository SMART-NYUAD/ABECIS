import os
import torch
TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
try:
    print("Setting up the ABECIS Environment")
    command = "pip3 install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/" + \
        CUDA_VERSION+"/torch"+TORCH_VERSION+"/index.html"
    os.system(command)
    os.system("pip3 install -r ./requirements.txt")
except Exception as e:
    print("Something went wrong.\n"+str(e))
