import onnx  # https://developer.nvidia.com/nvidia-tensorrt-download
import onnxruntime
import torch
import torch.nn as nn
import numpy as np
import logging
import sys as _sys
import yaml
import yolov5.model_utils as utils
from PIL import Image
from keyword import iskeyword as _iskeyword
from collections import OrderedDict, namedtuple
LOGGER = logging.getLogger('yolo_footkick')

def yaml_load(file='data.yaml'):
    # Single-line safe yaml loading
    with open(file, errors='ignore') as f:
        return yaml.safe_load(f)

#engine runtime
class model_onnx(nn.Module):
    
    def __init__(self, weights='yolov5s.pt', device=torch.device('cuda:0'), data=None, fp16=False):
        #init
        super().__init__()
        w = str(weights[0] if isinstance(weights, list) else weights)
        stride = 32  # default stride
        cuda = torch.cuda.is_available() and device.type != 'cpu'  # use CUDA
        LOGGER.info(f'Loading {w} for ONNX Runtime inference...')
        # check_requirements(('onnx', 'onnxruntime-gpu' if cuda else 'onnxruntime'))
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
        session = onnxruntime.InferenceSession(w, providers=providers)
        output_names = [x.name for x in session.get_outputs()]
        meta = session.get_modelmeta().custom_metadata_map  # metadata
        if 'stride' in meta:
            stride, names = int(meta['stride']), eval(meta['names'])
        if 'names' not in locals():
            names = yaml_load(data)['names'] if data else {i: f'class{i}' for i in range(999)}

        self.__dict__.update(locals())  # assign all variables to self

    def forward(self, im, augment=False, visualize=False):
        # YOLOv5 MultiBackend inference
        b, ch, h, w = im.shape  # batch, channel, height, width
        if self.fp16 and im.dtype != torch.float16:
            im = im.half()  # to FP16
        # ONNX Runtime
        im = im.cpu().numpy()  # torch to numpy
        y = self.session.run(self.output_names, {self.session.get_inputs()[0].name: im})

        if isinstance(y, (list, tuple)):
            return self.from_numpy(y[0]) if len(y) == 1 else [self.from_numpy(x) for x in y]
        else:
            return self.from_numpy(y)

    def from_numpy(self, x):
        return torch.from_numpy(x).to(self.device) if isinstance(x, np.ndarray) else x

    def warmup(self, imgsz=(1, 3, 640, 640)):
        # Warmup model by running inference once

        if  (self.device.type != 'cpu' ):
            im = torch.empty(*imgsz, dtype=torch.half if self.fp16 else torch.float, device=self.device)  # input
            for _ in range(2 if self.jit else 1):  #
                self.forward(im)  # warmup

