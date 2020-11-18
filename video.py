#-------------------------------------#
#       调用摄像头检测
#-------------------------------------#
from yolo import YOLO
from PIL import Image
import numpy as np
import cv2
import time
import pynvml
try:
    import pyrealsense2 as rs
except:
    print("pyrealsense error")
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)
out = cv2.VideoWriter('output/6.mp4',fourcc,30, (640,480))
yolo = YOLO()
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)
# 调用摄像头


fps = 0.0
while(True):
    t1 = time.time()
    # 读取某一帧
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    frame=np.asanyarray(color_frame.get_data())
    # 格式转变，BGRtoRGB
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    # 转变成Image
    frame = Image.fromarray(np.uint8(frame))
    
    # 进行检测
    frame = np.array(yolo.detect_image(frame))

    # RGBtoBGR满足opencv显示格式
    frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)

    fps  = ( fps + (1./(time.time()-t1)) ) / 2
    print("fps = %.2f"%(fps))
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    #print("已用显存 = %.2f M"% (meminfo.used/1024/1024))
    #print("剩余显存 = %.2f M"% (meminfo.free/1024/1024))
    frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    out.write(frame)
    print(frame.shape)
    cv2.imshow("video",frame)


    c= cv2.waitKey(30) & 0xff 
    if c==27:
        out.release()
        pipeline.stop()
        break
