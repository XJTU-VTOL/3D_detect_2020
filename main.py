import cv2
import sys
import numpy as np
from PIL import Image
from PyQt5.QtWidgets import QApplication, QMainWindow ,QFileDialog
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import MainWindow

from yolo import YOLO
from PIL import Image
import numpy as np
import time
from track import Sort,SortBox,sort_and_draw_csv
import copy
import imutils
import copy

isEnd=False

results = []
display = {}
mot_tracker = Sort()
frame_h = 480
frame_w = 640
isCameraOpen=False
isDetect=False
Turn=False

# 摄像头线程
class Mythread(QThread):
    

    textSignal = pyqtSignal(int)
    def __init__(self, parent=None):
        super().__init__(parent)
        self.min_per = 4
        self.max_per = 30
        self.warmup = 10
        self.kinetic = False

        # initialize the background subtractor
        self.fgbg = cv2.bgsegm.createBackgroundSubtractorGMG()
        self.frames_n = 0

        # judge if kinetic or silent keeps 5 frame
        self.kinetic_count = 0
        self.max_count = 5
        # open a pointer to the video file initialize the width and height of the frame
        (self.W, self.H) = (None, None)
        print("Silent now")



    def run(self):
        global cap,isCameraOpen,isEnd,yolo,results,display,mot_tracker,isDetect
        #try:
        #    import pyrealsense2 as rs
        #except:
        #    print("pyrealsense error")
        yolo = YOLO()

        #pipeline = rs.pipeline()
        #config = rs.config()
        #config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        #config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        # Start streaming
        #pipeline.start(config)
        capture = cv2.VideoCapture(1)
        capture.set(3,1280)
        capture.set(4,720)
        time1 = time.perf_counter()
        while 1:
            if isEnd:
                #pipeline.stop()
                break
            if isCameraOpen:
                    
                #frames = pipeline.wait_for_frames()
                #depth_frame = frames.get_depth_frame()
                #color_frame = frames.get_color_frame()
                #if not depth_frame or not color_frame:
                #    continue
                ## Convert images to numpy arrays
                #depth_image = np.asanyarray(depth_frame.get_data())
                #color_image = np.asanyarray(color_frame.get_data())
                ## Apply colormap on depth image (image must be converted to 8-bit per pixel first)
                #depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
                ref,color_image = capture.read()
                imageRGB = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
                im = Image.fromarray(imageRGB)
                img_pix = im.toqpixmap().scaled(1280*0.7,720*0.7)
                ui.labelimg1.setPixmap(img_pix)

                # color_image1 center is drawed in black 
                pic = cv2.imread('ratio62.jpg')
                color_image1 = np.array(color_image*pic,dtype=np.uint8)

                # color_image2 = copy.deepcopy(color_image)
                color_image2 = imutils.resize(color_image1, width=600)
                mask = self.fgbg.apply(color_image2)

                    # apply a series of erosions and dilations to eliminate noise
                mask = cv2.erode(mask, None, iterations=2)
                mask = cv2.dilate(mask, None, iterations=2)
                    # if the width and height are empty, grab the spatial dimensions
                if self.W is None or self.H is None:
                    (self.H, self.W) = mask.shape[:2]
                        # compute the percentage of the mask that is "foreground"
                p = (cv2.countNonZero(mask) / float(self.W * self.H)) * 100

                    # if there is less than No of the frame as "foreground"then we
                    # know that the motion has stopped and thus we should grab the frame
                    # Shake proof
                ######################################
                if isDetect:
                    #results = []
                   
                    # stop detect if time2-time1>19s
                    time2 = time.perf_counter()
                    if time2-time1>19:
                        openAndCloseDetect()

                    frame = Image.fromarray(np.uint8(imageRGB))
                    bboxes = yolo.get_bbox(frame)
                    #print("bbox:")
                    #print(bboxes)
                    #print()
                    if bboxes == [None]:
                        image = np.asarray(frame)

                    else :
                        sort_boxes = []  

                        for box in bboxes:
                            sort_box = SortBox(box,yolo.class_names)
                            sort_boxes.append(sort_box)

                        frame = np.asarray(frame)
                        image,obs = sort_and_draw_csv(image = frame, 
                                                        boxes = sort_boxes, 
                                                        labels = yolo.class_names,
                                                        obj_thresh = 0.5,
                                                        mot_tracker = mot_tracker)
                        if results == []:
                            results = copy.deepcopy(obs)
                            #print("results1",results)
                            self.textSignal.emit(1)
                        else :
                            for ob in obs[::-1]:
                                if ob[0] > results[0][0]:
                                    results.insert(0,ob)
                                    #print(results)
                                    self.textSignal.emit(2)
                    im = Image.fromarray(image)
                    img_pix = im.toqpixmap().scaled(1280*0.7,720*0.7)
                    ui.labelimg2.setPixmap(img_pix)
                else:
                    self.textSignal.emit(3)

                if Turn:
                    if p < self.min_per and self.kinetic and self.frames_n > self.warmup:
                         # silent
                        self.kinetic_count += 1
                        if self.kinetic_count > self.max_count:
                            self.kinetic = False
                            self.kinetic_count = 0
                            print("Silent now")
                            ui.pushButton3.setText("当前状态:待转动")
                        # Kinetic
                    elif p > self.max_per and not self.kinetic:
                        self.kinetic_count += 1
                        if self.kinetic_count > self.max_count:
                            self.kinetic = True
                            self.kinetic_count = 0
                            print("Kinetic now")
                            ui.pushButton3.setText("当前状态:转动中")
                            ui.textEdit.append("转动中")
                            results.clear()
                        
                        # display the frame and detect if there is a key press
                    cv2.imshow("Frame", color_image1)
                    cv2.imshow("Mask", mask)
                    cv2.waitKey(1)
                    self.frames_n += 1
                    #############################################
            else:

                ui.labelimg1.setText("摄像机已关闭")





# 摄像头开关
def openAndCloseCamera():
    global ui,isCameraOpen
    isCameraOpen = not isCameraOpen
    if isCameraOpen:
        ui.pushButton1.setText("关闭相机")
    else:
        ui.pushButton1.setText("开启相机")

# 开启识别
def openAndCloseDetect():
    global ui,isDetect
    isDetect = not isDetect
    if isDetect:
        ui.pushButton2.setText("识别中")
        ui.pushButton2.setText("结束识别")
    else:
        ui.pushButton2.setText("开始识别")
        text_clear()


def textDisplay(flag):


    global ui,display,results
 
    #print(flag)
    print("results:",results)
    if flag == 1:
        for res in results[::-1]:
            ui.textEdit.append("目标ID: "+str(res[1])+"   "+"Num: "+str(res[0]))

            #print("num: "+str(res[0])+"   "+"id: "+str(res[1])+"\n")
    if flag == 2:
        ui.textEdit.append("目标ID: "+str(results[0][1])+"   Num: "+str(results[0][0]))

        #print("num: "+str(results[0][0])+"   id: "+str(results[0][1])+"\n")
    if flag == 3:
        if results != []:
            display = count(results)
            print(display)
            ui.textEdit.append('\nresults:\n')
            for key in display.keys():
                ui.textEdit.append('目标ID:'+str(key)+'  数量:'+str(display[key]))
                text_add('Goal_ID='+str(key)+';Num='+str(display[key])+'\n')
            results = []
# 统计最终结果
def count(list):
    for i in list:
        if i[1] in display.keys():
            display[i[1]] = int(display[i[1]]) + 1
        else:
            display[i[1]] = 1
    return display

# 创建txt文件
def text_add(msg):
    path = "D:\\3d_detector\\XJTU-XJTU_Unicorns-R1.txt"
    file = open(path,'a')
    file.write(msg)
    file.close()

# 清空txt文件
def text_clear():
    path = "D:\\3d_detector\\XJTU-XJTU_Unicorns-R1.txt"
    file = open(path,'w')
    file.truncate()
    file.close()

#退出程序
def toEnd():
    global isEnd,app
    app.exec_()
    isEnd = True

if __name__ == '__main__':

    ###################################
    #global ui,isCameraOpen,cap,isEnd,yolo,results,display,mot_tracker,isDetect
    #isEnd=False
    #yolo = YOLO()
    #results = []
    #display = {}
    #mot_tracker = Sort()
    #frame_h = 480
    #frame_w = 640
    #isCameraOpen=False
    #isDetect=False
    #####################################
    app = QApplication(sys.argv)
    myMainWindow = QMainWindow()
    # ui
    ui = MainWindow.Ui_MainWindow()
    ui.setupUi(myMainWindow)
    # 线程
    
    thread = Mythread()
    thread.textSignal.connect(textDisplay)
    thread.start()


    # 按钮
    ui.pushButton1.clicked.connect(lambda: openAndCloseCamera())
    ui.pushButton2.clicked.connect(lambda: openAndCloseDetect())
    #显示
    myMainWindow.show()
    #退出
    sys.exit(toEnd())



