
# output box.xmin,box.ymin,box.xmax,box.ymax,labels[label],box.get_score(),ID
import numpy as np
from sklearn.utils.linear_assignment_ import linear_assignment
from filterpy.kalman import KalmanFilter
import cv2


def iou(bb_test,bb_gt):
    """
    Computes IUO between two bboxes in the form [x1,y1,x2,y2]
    """
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
              + (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1]) - wh)
    return (o)

def associate_detections_to_trackers(detections,trackers,iou_threshold = 0.3):
    """
    Assigns detections to tracked object (both represented as bounding boxes)

    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    if (len(trackers) == 0):
        return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)
    iou_matrix = np.zeros((len(detections),len(trackers)),dtype=np.float32)

    for d,det in enumerate(detections):
        for t,trk in enumerate(trackers):
            iou_matrix[d,t] = iou(det,trk)
    matched_indices = linear_assignment(-iou_matrix)

    unmatched_detections = []
    for d,det in enumerate(detections):
        if(d not in matched_indices[:,0]):
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t,trk in enumerate(trackers):
        if(t not in matched_indices[:,1]):
            unmatched_trackers.append(t)

    #filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if(iou_matrix[m[0],m[1]]<iou_threshold):
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1,2))
    if(len(matches)==0):
        matches = np.empty((0,2),dtype=int)
    else:
        matches = np.concatenate(matches,axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
    [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
    the aspect ratio
    """
    w = bbox[2]-bbox[0]
    h = bbox[3]-bbox[1]
    x = bbox[0]+w/2.
    y = bbox[1]+h/2.
    s = w*h    #scale is just area
    r = w/float(h)
    return np.array([x,y,s,r]).reshape((4,1))

def convert_x_to_bbox(x,score=None):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
    [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    w = np.sqrt(x[2]*x[3])
    h = x[2]/w
    if(score==None):
        return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.]).reshape((1,4))
    else:
        return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.,score]).reshape((1,5))


class KalmanBoxTracker(object):
    """
    This class represents the internel state of individual tracked objects observed as bbox.
    """
    count = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
    ID_definition=0
    Fragmentation=0
    def __init__(self,bbox):
        """
        Initialises a tracker using initial bounding box.
        """
        #define constant velocity model
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],  [0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]])
        self.kf.H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]])

        self.kf.R[2:,2:] *= 10.
        self.kf.P[4:,4:] *= 1000. #give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[4:,4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0

        #print('before:',KalmanBoxTracker.count)
        self.id = KalmanBoxTracker.count[0]

        KalmanBoxTracker.count.pop(0)
        #print('after:',KalmanBoxTracker.count)
        KalmanBoxTracker.ID_definition +=1
        print('--------------------------ID_definition = ',KalmanBoxTracker.ID_definition)

        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.classes=bbox[4:]

    def update(self,bbox):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.classes = bbox[4:]
        self.kf.update(convert_bbox_to_z(bbox))

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        #we only need to predict once, if more than one time unmatch, we don't want to predict to prevent the prediction error.
        #if self.time_since_update==0:
        #    if((self.kf.x[6]+self.kf.x[2])<=0):
        #      self.kf.x[6] *= 0.0
        #    self.kf.predict()
        #    self.age += 1
        #    if(self.time_since_update>0):
        #      self.hit_streak = 0
        #    self.time_since_update += 1
        #else:
        #    self.time_since_update += 1
        #    #print('miss ID:', self.id+1)
        #    KalmanBoxTracker.Fragmentation +=1
        #    print('Fragmentation = ',KalmanBoxTracker.Fragmentation)
        #self.history.append(convert_x_to_bbox(self.kf.x))
        #return self.history[-1]
        if((self.kf.x[6]+self.kf.x[2])<=0):
          self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if(self.time_since_update>0):
          self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]   

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        r1=convert_x_to_bbox(self.kf.x)
        r2=[self.classes[i] for i in range(len(self.classes))]
        r=np.append(r1,r2)
        r.reshape(1,-1)
        return r

    def delete_ID(self):
        KalmanBoxTracker.count.append(self.id)

class Sort(object):
    def __init__(self,max_age=50,min_hits=8,trackers_number=20):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.trackers = []
        self.ready = np.zeros((1,4)) #self.ready is used to save the dets from the last frame. and the first frame's ready=[0,0,0,0]
        self.last_location = np.zeros((trackers_number,54))####################################################################
        self.last_matched = []
        self.last_unmatched_trks = []
        self.last_unmatched_dets = []

        self.frame_count = 0
        self.trackers_number = trackers_number
    def update(self,dets):
        """
        Params:
        dets - a numpy array of detections in the format [[x1,y1,x2,y2,classes],[x1,y1,x2,y2,classes],...]
        Requires: this method must be called once for each frame even with empty detections.
        Returns the a similar array, where the last column is the object ID.

        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        self.frame_count += 1
        #get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers),5))
        to_del = []
        ret = []
        for t,trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if(np.any(np.isnan(pos))):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets,trks)
        #print('matched: ',matched, 'unmatched_dets:',unmatched_dets, 'unmatched_trks:', unmatched_trks)
        #input()


        ###计算last_location

        #print('dets:',dets)
        if len(self.last_matched) != 0:
            #print(type(self.last_matched))
            for i,trk in enumerate(self.trackers):
                if(i in self.last_matched[:,1]):
                    self.last_location[i] = self.ready[self.last_matched[np.where(self.last_matched[:,1]==i)[0],0],:]
        
        for ind,umd in enumerate(unmatched_dets):
            for ink,umk in enumerate(unmatched_trks):
                I = iou(self.last_location[umk],dets[umd,:])
                if I>0.65:
                    print('matched:')
                    print(matched)
                    print('umd,umk:')
                    print([umd,umk])
                    matched = np.row_stack((matched,np.asarray([umd,umk])))
                    if len(unmatched_dets)>0:
                        if len(unmatched_trks)>0:
                            print('unmatched_dets')
                            print(unmatched_dets)
                            print(ind)
                            unmatched_dets = np.delete(unmatched_dets,ind,axis=0)
                            unmatched_trks = np.delete(unmatched_trks,ink,axis=0)

        #update matched trackers with assigned detections
        for t,trk in enumerate(self.trackers):
            if(t not in unmatched_trks):
                d = matched[np.where(matched[:,1]==t)[0],0]
                trk.update(dets[d,:][0])

        #create and initialise new trackers for unmatched detections
        # for i in unmatched_dets:
        #     trk = KalmanBoxTracker(dets[i,:])
        #     self.trackers.append(trk)
        # i = len(self.trackers)
        #2018.11.30 compare near frame ,if can match, creat  new trackers
        if len(self.trackers) <self.trackers_number:
            for i in unmatched_dets:
                for j in range(len(self.ready)):
                    I=iou(self.ready[j],dets[i,:])
                    if I>0.65:
                        np.delete(self.ready,j,axis=0)
                        trk = KalmanBoxTracker(dets[i, :])
                        self.trackers.append(trk)
            self.ready=dets
        i = len(self.trackers)

        for trk in reversed(self.trackers):
            d = trk.get_state()
            if((trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits)):
                ret.append(np.concatenate((d,[trk.id+1])).reshape(1,-1)) # +1 as MOT benchmark requires positive
            i -= 1
            #print('self.frame_count: ',self.frame_count,'ID:',trk.id,'trk.time_since_update: ',trk.time_since_update,'trk.hit_streak: ',trk.hit_streak)
            #remove dead tracklet 大于self.max_age中断跟踪,实际中可以设置大一些,防止偶然的漏检测中断跟踪
            if(trk.time_since_update > self.max_age):
                trk.delete_ID()
                self.trackers.pop(i)

        self.last_matched, self.last_unmatched_dets, self.last_unmatched_trks = matched, unmatched_dets, unmatched_trks
        if(len(ret)>0):
            return np.concatenate(ret)
        return np.empty((0,5))



#本程序用于目标跟踪并画出包围框
def sort_and_draw_csv(image, boxes, labels, obj_thresh, mot_tracker, quiet=True):
    det=[]
    obs=[]
    for box in boxes:
        flag=0
        d=[box.xmin,box.ymin,box.xmax,box.ymax]
        for i in range(len(labels)):
            d.append(box.classes[i])
            if box.classes[i]>obj_thresh:
                flag=1
        if flag:
            det.append(d)
    dets=np.array(det)
    trackers = mot_tracker.update(dets) #trackers=[xmin,ymin,xmax,ymax,classes,ID]
    print("trackers:")
    print(trackers)
    print()
    trackers = trackers.tolist()


    for box in trackers:
        label_str='num:'+str(box[-1])+' '
        label=-1
        for i in range(len(labels)):
            if box[i+4]>obj_thresh:
                label_str += (labels[i] + ' ' + str(round(box[i+4] * 100, 2)) + '%')
                label=i
            if not quiet: print(label_str)
        obs.append([int(box[-1]),labels[label]])
        if label >=0:
            text_size = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, 1.1e-3 * image.shape[0], 5)
            width, height = text_size[0][0], text_size[0][1]
            region = np.array([[box[0] - 3, box[1]],
                               [box[0] - 3, box[1] - height - 26],
                               [box[0] + width + 13, box[1] - height - 26],
                               [box[0] + width + 13, box[1]]], dtype='int32')
            cv2.rectangle(img=image, pt1=(round(box[0]), round(box[1])), pt2=(round(box[2]), round(box[3])), color=get_color(label),thickness=5)
            cv2.fillPoly(img=image, pts=[region], color=get_color(label))
            cv2.putText(img=image,
                        text=label_str,
                        org=(round(box[0]) + 13, round(box[1]) - 13),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1e-3 * image.shape[0],
                        color=(0, 0, 0),
                        thickness=2)
    return image,obs

class SortBox:
    def __init__(self, bbox, labels):
        self.bbox = bbox
        self.xmin = bbox[0]
        self.ymin = bbox[1]
        self.xmax = bbox[2]
        self.ymax = bbox[3]

        self.label = int(bbox[-1])
        self.score = bbox[-2]
    
        self.classes = [0]*len(labels)
        self.classes[self.label] = self.score



def get_color(label):
    if label < len(colors):
        return colors[label]
    else:
        print('Label {} has no color, returning default.'.format(label))
        return (0, 255, 0)

colors = [
    [255 , 172 , 0]   , 
    [31  , 0   , 255] , 
    [0   , 159 , 255] , 
    [255 , 95  , 0]   , 
    [0, 255, 255],  
    [255 , 248 , 0]   ,
    [0   , 55  , 160] , 
    [31  , 0   , 255] , 
    [255 , 95  , 0]   , 
    [255 , 0   , 133] , 
    [0   , 255 , 25]  , 
    [255 , 19  , 0]   ,
    [255 , 0   , 0]   ,
    [255 , 38  , 0]   ,
    [255 , 172 , 0]   ,
    [108 , 0   , 255] ,
    [0   , 255 , 6]   ,
    [255 , 0   , 152] ,
    [223 , 0   , 255] ,
    [12  , 0   , 255] ,
    [0   , 255 , 178] ,
    [108 , 255 , 0]   ,
    [184 , 0   , 255] ,
    [255 , 0   , 76]  ,
    [146 , 255 , 0]   ,
    [51  , 0   , 255] ,
    [0   , 197 , 255] ,
    [255 , 0   , 19]  ,
    [255 , 0   , 38]  ,
    [89  , 255 , 0]   ,
    [127 , 255 , 0]   ,
    [255 , 153 , 0]   ,
    [0   , 255 , 216] ,
    [0   , 255 , 121] ,
    [255 , 0   , 248] ,
    [70  , 0   , 255] ,
    [0   , 255 , 159] ,
    [0   , 216 , 255] ,
    [0   , 6   , 255] ,
    [0   , 63  , 255] ,
    [31  , 255 , 0]   ,
    [255 , 57  , 0]   ,
    [255 , 0   , 210] ,
    [0   , 255 , 102] ,
    [242 , 255 , 0]   ,
    [255 , 191 , 0]   ,
    [0   , 255 , 63]  ,
    [255 , 0   , 95]  ,
    [146 , 0   , 255] ,
    [184 , 255 , 0]   ,
    [255 , 114 , 0]   ,
    [0   , 255 , 235] ,
    [255 , 229 , 0]   ,
    [0   , 178 , 255] ,
    [255 , 0   , 114] ,
    [255 , 0   , 57]  ,
    [0   , 140 , 255] ,
    [0   , 121 , 255] ,
    [12  , 255 , 0]   ,
    [255 , 210 , 0]   ,
    [0   , 255 , 44]  ,
    [165 , 255 , 0]   ,
    [0   , 25  , 255] ,
    [0   , 255 , 140] ,
    [0   , 101 , 255] ,
    [0   , 255 , 82]  ,
    [223 , 255 , 0]   ,
    [242 , 0   , 255] ,
    [89  , 0   , 255] ,
    [165 , 0   , 255] ,
    [70  , 255 , 0]   ,
    [255 , 0   , 172] ,
    [255 , 76  , 0]   ,
    [203 , 255 , 0]   ,
    [204 , 0   , 255] ,
    [255 , 0   , 229] ,
    [255 , 133 , 0]   ,
    [127 , 0   , 255] ,
    [0   , 235 , 255] ,
    [0   , 255 , 197] ,
    [255 , 0   , 191] ,
    [0   , 44  , 255] ,
    [50  , 255 , 0]
]
