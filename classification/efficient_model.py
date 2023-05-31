# 导入所需的库
import cv2 # 用于图像处理
import numpy as np # 用于数组操作
import onnxruntime # 用于加载和运行onnx模型

def build_model(onnx_path,  device):
    # 定义onnx文件的路径
    onnx_file = onnx_path

    # 加载onnx模型并创建推理会话
    if device == "cpu":
        session = onnxruntime.InferenceSession(onnx_file, providers=['CPUExecutionProvider'])
    elif device == "gpu":
        session = onnxruntime.InferenceSession(onnx_file, providers=['CUDAExecutionProvider'])

    return session

def inference(session, img, imgsz):
    # 加载图片并转换为numpy数组s
    image = np.array(img, dtype=np.float32)

    # 对图片进行resize成[32,32,3]
    image = cv2.resize(image, (imgsz, imgsz))

    # 对图片进行归一化，将像素值映射到[0,1]区间
    image = image / 255.0

    # 对图片变成四通道[1,3,32,32]
    image = image.transpose(2, 0, 1) # 将通道维度放在第一个位置
    image = image[np.newaxis, ...] # 增加一个批量维度

    # 获取模型的输入和输出名称
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # 使用onnxruntime进行推理并输出结果
    result = session.run([output_name], {input_name: image})
    return result