import cv2
from yolo import YOLO, detect_video
from PIL import Image
import os

def detect_img(yolo):
    label = 5
    work = 'val'
    file = open(f'{work}_{label}.txt','a+')

    img_dir = f'/home/ros/gesture_ws/new_{label}/'

    for img_name in os.listdir(img_dir):
        if not img_name.endswith('.png'):
            continue
        img = img_dir + img_name
        image = Image.open(img)
        mat = cv2.imread(img)
        print(img)
        boxes = yolo.detect_image(image)
        skip_count = 0
        if len(boxes) == 0:
            os.remove(img)
            continue
        for box in boxes:
            cv2.rectangle(mat, (box[1], box[0]), (box[3], box[2]), (255, 0, 255), 1)
            cv2.imshow('image', mat)
            if cv2.waitKey(0) & 0xFF == ord('e'):
                cx = int((box[1] + box[3]) // 2)
                cy = int((box[0] + box[2]) // 2)
                w = int(box[3] - box[1])
                h = int(box[2] - box[0])
                center_box = [label, cx, cy, w, h]
                file.write(f'{img_name} ')
                for n in center_box:
                    file.write(f'{n} ')
                file.write('\n')
                cv2.destroyAllWindows()
            if cv2.waitKey(0) & 0xFF == ord('q'):
                skip_count +=1
                if len(boxes) == skip_count:
                    os.remove(img)
                cv2.destroyAllWindows()

        else:
            continue
    file.close()

detect_img(YOLO(**vars()))