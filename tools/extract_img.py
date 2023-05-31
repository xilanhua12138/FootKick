import os
import shutil

# 输入图片和JSON所在的目录以及要输出的目录
shoe_dirs = [
    '/home/yunhaoshui/FootKick/dataset/leather_shoe1_black',
    '/home/yunhaoshui/FootKick/dataset/leather_shoe2_black',
    '/home/yunhaoshui/FootKick/dataset/slipper1_black',
    '/home/yunhaoshui/FootKick/dataset/slipper2_gray',
    '/home/yunhaoshui/FootKick/dataset/slipper3_green',
    '/home/yunhaoshui/FootKick/dataset/sportshoe1_white',
    '/home/yunhaoshui/FootKick/dataset/sportshoe2_white',
    '/home/yunhaoshui/FootKick/dataset/sportshoe3_orange',
    '/home/yunhaoshui/FootKick/dataset/sportshoe4_gray',
    '/home/yunhaoshui/FootKick/dataset/sportshoe5_gray'
    ]
unshoe_dirs = [
    '/home/yunhaoshui/FootKick/dataset/hand',
    '/home/yunhaoshui/FootKick/dataset/phone_shaking',
    '/home/yunhaoshui/FootKick/dataset/pillbottle_rolling',
    '/home/yunhaoshui/FootKick/dataset/socks_rolling',
    '/home/yunhaoshui/FootKick/dataset/tissue_rolling'
    ]
# json_dirs = ['path/to/json/directory']
shoe_output_dir = '/home/yunhaoshui/FootKick/clean_dataset/shoe'
unshoe_output_dir = '/home/yunhaoshui/FootKick/clean_dataset/unshoe'

for shoe_dir in shoe_dirs:
    # 遍历图片目录中的所有文件
    for image_file in os.listdir(shoe_dir):
        # 检查文件扩展名是否为图片格式（这里假设图片格式为 .jpg）
        if image_file.endswith('.bmp'):
            # 根据图片文件名构建相应的 JSON 文件名
            json_file = os.path.splitext(image_file)[0] + '.json'
            # 检查 JSON 文件是否存在
            if os.path.isfile(os.path.join(shoe_dir, json_file)):
                # 如果存在，则将图片文件复制到输出目录中
                shutil.copy2(os.path.join(shoe_dir, image_file), shoe_output_dir)

for unshoe_dir in unshoe_dirs:
    # 遍历图片目录中的所有文件
    for image_file in os.listdir(unshoe_dir):
        # 检查文件扩展名是否为图片格式（这里假设图片格式为 .jpg）
        if image_file.endswith('.bmp'):
            # 根据图片文件名构建相应的 JSON 文件名
            json_file = os.path.splitext(image_file)[0] + '.json'
            # 检查 JSON 文件是否存在
            if os.path.isfile(os.path.join(unshoe_dir, json_file)):
                # 如果存在，则将图片文件复制到输出目录中
                shutil.copy2(os.path.join(unshoe_dir, image_file), unshoe_output_dir)