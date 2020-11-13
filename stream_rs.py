import pyrealsense2 as rs
import cv2
import numpy as np

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)
image_count = 30
label = 5
label_dir = f'/home/ros/gesture_ws/new_{label}/'
work = 'val'
try:
    while True:

        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())

        cv2.imshow('RealSense', color_image)

        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

        if key & 0xFF == ord('t'):
            cv2.imwrite(label_dir + f'{work}{label}_{image_count}.png', color_image)
            print(f'save image {label}_{image_count}.png')
            image_count += 1


finally:

    # Stop streaming
    pipeline.stop()