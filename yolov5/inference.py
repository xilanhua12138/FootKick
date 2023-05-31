from yolov5.buildmodel_engine import model_engine
from yolov5.buildmodel_onnx import model_onnx
import yolov5.model_utils as utils
import numpy as np
import torch
from yolov5.plots import Annotator ,colors

def model_init(weights_path, model_type, data_path ,device, half=True, imgsz=320):
    if model_type == "engine":
        model = model_engine(weights_path, device=device, fp16=half, data=data_path)
    if model_type == "onnx":
        model = model_onnx(weights_path, device=device, fp16=half, data=data_path)
    model.warmup(imgsz=(1, 3, imgsz, imgsz))  # warmup
    return model

def inference_ultralytics(
    img0, 
    model, 
    imgsz=320,
    conf_thres=0.25,
    iou_thres=0.45,
    classes=None,
    agnostic=False,
    multi_label=False,
    labels=(),
    max_det=1,
    ):
    
    stride, names = model.stride, model.names
    imgsz = utils.check_img_size(imgsz, s=stride)  # check image size
    # Preprocess
    img = utils.letterbox(img0, imgsz, stride=stride, auto=False)[0]  # padded resize
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)  # contiguous
    
    # warmup
    # model.warmup(imgsz=(1, 3, imgsz, imgsz))  # warmup
    img = torch.from_numpy(img).to(model.device)
    img = img.half() if model.fp16 else img.float()  # uint8 to fp16/32
    img /= 255  # 0 - 255 to 0.0 - 1.0
    if len(img.shape) == 3:
        img= img[None]  # expand for batch dim

    # Inference
    pred = model(img)

    # NMS
    pred = utils.non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic=False, max_det=max_det)
    xyxy = []
    conf = []
    cls = []
    # Post process
    annotator = Annotator(img0, line_width=2, example=str(names))
    for i, det in enumerate(pred):  

        for *xyxy_, conf_, cls_ in reversed(det):
            xyxy.append(xyxy_)
            conf.append(conf_)
            cls.append(cls_)
            c = int(cls_)  # integer class
            label =  f'{names[c]} {conf_:.2f}'
            annotator.box_label(xyxy_, label, color=colors(c, True))

    im0 = annotator.result()
    return xyxy, conf, cls, im0


def inference_openmmlab(
    img0, 
    model, 
    imgsz=320,
    conf_thres=0.25,
    iou_thres=0.45,
    classes=None,
    agnostic=False,
    multi_label=False,
    labels=(),
    max_det=1,
    ):
    
    stride, names = model.stride, model.names
    imgsz = utils.check_img_size(imgsz, s=stride)  # check image size
    # Preprocess
    img = utils.letterbox(img0, imgsz, stride=stride, auto=False)[0]  # padded resize
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)  # contiguous
    
    # warmup
    # model.warmup(imgsz=(1, 3, imgsz, imgsz))  # warmup
    img = torch.from_numpy(img).to(model.device)
    img = img.half() if model.fp16 else img.float()  # uint8 to fp16/32
    img /= 255  # 0 - 255 to 0.0 - 1.0
    if len(img.shape) == 3:
        img= img[None]  # expand for batch dim

    # Inference
    pred = model(img)
    xyxy = []
    cls = []

    num_det = int(pred[0])
    xyxy_ = pred[1][0,0,:].numpy()
    conf_ = float(pred[2][0,0])
    cls_ = int(pred[3][0,0])

    annotator = Annotator(img0, line_width=2, example=str(names))

    if cls_ != -1:
        cls.append(cls_)
        c = int(cls_)  # integer class
        label =  f'{names[c]} {conf_:.2f}'
        annotator.box_label(xyxy_, label, color=colors(c, True))
    
    im0 = annotator.result()
    return xyxy_, conf_, cls, im0