import cv2
def video_demo():
    capture = cv2.VideoCapture(1)
    print(capture.get(3),capture.get(4))
    capture.set(3,1280)
    capture.set(4,720)
    print(capture.get(3),capture.get(4))
    while(True):
        ref,frame = capture.read()
        cv2.imshow('frame',frame)
        c = cv2.waitKey(10) & 0xff
        #print(c)
        if c == 27:
            cv2.destroyAllWindows()
            break

if __name__ == '__main__':
    print('开始')
    video_demo()




