from yolo import YOLO
from PIL import Image
import numpy as np
import cv2
import time
from track import Sort,SortBox,sort_and_draw_csv
import copy

yolo = YOLO()
mot_tracker = Sort()
capture = cv2.VideoCapture("5.mp4")
results = []
display = {}
video_out = 'output/5.mp4'
nb_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
frame_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
# video_writer = cv2.VideoWriter(video_out,
                               # cv2.VideoWriter_fourcc(*'XVID'),
                               # 50.0,
                               # (frame_w, frame_h))

def count(list):
    for i in list:
        if i[1] in display.keys():
            display[i[1]] = int(display[i[1]]) + 1
        else:
            display[i[1]] = 1
    print(display)



fps = 0.0

for i in range(nb_frames):
    t1 = time.time()
    _,frame = capture.read()
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    frame = Image.fromarray(np.uint8(frame))

    bboxes = yolo.get_bbox(frame)
    if bboxes == [None]:
        image = cv2.cvtColor(np.asarray(frame),cv2.COLOR_RGB2BGR)

    else :
        sort_boxes = []  

        for box in bboxes:
            sort_box = SortBox(box,yolo.class_names)
            sort_boxes.append(sort_box)

        frame = cv2.cvtColor(np.asarray(frame),cv2.COLOR_RGB2BGR)
        image,obs = sort_and_draw_csv(image = frame, 
                                  boxes = sort_boxes, 
                                  labels = yolo.class_names,
                                  obj_thresh = 0.5,
                                  mot_tracker = mot_tracker
                                  )
        # video_writer.write(image)
    if results == []:
        results = copy.deepcopy(obs)
    else :
        for ob in obs[::-1]:
            if ob[0] > results[0][0]:
                results.insert(0,ob)

    print(results) 
    fps  = ( fps + (1./(time.time()-t1)) ) / 2
    print("fps= %.2f"%(fps))

    cv2.imshow("video",image)

    c= cv2.waitKey(30) & 0xff 
    if c==27:
        capture.release()
        # video_writer.release()
        break
#capture=cv2.VideoCapture(0)

#fps = 0.0

#while(True):
#    t1 = time.time()
#    # 读取某一帧
#    ref,frame=capture.read()
#    # 格式转变，BGRtoRGB
#    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
#    # 转变成Image
#    frame = Image.fromarray(np.uint8(frame))
    
#    # 进行检测
#    bboxes = yolo.get_bbox(frame)

#    if bboxes == [None]:
#        image = cv2.cvtColor(np.asarray(frame),cv2.COLOR_RGB2BGR)

#    else:
#        frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
#        sort_boxes = []  
#        for box in bboxes:
#            sort_box = SortBox(box,yolo.class_names)
#            sort_boxes.append(sort_box)

#        frame = cv2.cvtColor(np.asarray(frame),cv2.COLOR_RGB2BGR)

#        image = sort_and_draw_csv(image = frame, 
#                                  boxes = sort_boxes, 
#                                  labels = yolo.class_names,
#                                  obj_thresh = 0.5,
#                                  mot_tracker = mot_tracker
#                                  )
#    fps= ( fps + (1./(time.time()-t1)) ) / 2

#    print("fps = %.2f"%(fps))
#    cv2.imshow("video",image)


#    c= cv2.waitKey(30) & 0xff 
    
#    if c==27:
#        capture.release()
#        break

#capture = cv2.VideoCapture("1.mp4")
#nb_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))-20
#frame_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
#frame_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))

#mot_tracker = Sort()
