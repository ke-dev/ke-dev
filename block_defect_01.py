from subprocess import check_output
from mmseg.apis import inference_segmentor, init_segmentor
import mmcv, torch
import time
import cv2
import os
import re
from glob import glob

# config_file = '../mmsegmentation/configs/swin/upernet_swin_tiny_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K.py'

config_file = '/workspace/docker/work_dirs/upernet_swin_tiny_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K/upernet_swin_tiny_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K.py'
checkpoint_file = '/workspace/docker/work_dirs/upernet_swin_tiny_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K/latest.pth'

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = init_segmentor(config_file, checkpoint_file, device)

#dir = "./source_data/raw_data/"
video_path = "../test_picture/jpg_dir/"
gen_path = "../results/block_test_01/"
frames = glob(os.path.join(video_path, '*.png'))

num = 0
sum = 0
start = time.time()
for i, frame in enumerate(frames):
    pattern = re.compile(r'([^<>/\\\|:""\*\?]+)\.\w+$')
    data = pattern.findall(frame)
    data_now = data[0]
    Newdir = os.path.join(video_path, str(data_now) + '.png')

    Gendir = os.path.join(gen_path, str(data_now) + '.png')

    name = str(data_now) + ".png"


    img = Newdir  
    num += 1
    start = time.time()
    result = inference_segmentor(model, img)
    end = time.time()
    t = end-start
    sum += t


    model.show_result(img, result, out_file=Gendir, opacity=0.5)
print("数据个数：", num)
print("总时间：", sum)
print("平均时间：", (sum/num))
