import os

img_w = 640
img_h = 480
classes = ['0', '2', '5']
work = 'train'
number = 5
with open(f'{work}_{number}.txt') as f:
    lines = f.read().splitlines()

img_temp = None

for line in lines:
    label = line.split()
    if label[0] == img_temp :
        mode = 'a+'
    else:
        mode = 'w'
    img_temp = label[0]

    label_txt = open(f'/home/ros/yolov5/gesture_data/labels/{work}/' + label[0].replace('.png', '.txt'), mode)

    label_txt.write(f'{classes.index(label[1])} ')
    label_txt.write(f'{int(label[2]) / img_w} ')
    label_txt.write(f'{int(label[3]) / img_h} ')
    label_txt.write(f'{int(label[4]) / img_w} ')
    label_txt.write(f'{int(label[5]) / img_h}')

    label_txt.close()

