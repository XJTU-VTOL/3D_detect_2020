import xml.etree.ElementTree as ET
from os import getcwd

sets=[('2007', 'train'), ('2007', 'val'), ('2007', 'test')]

wd = getcwd()
#classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
classes = ["CA001","CA002","CA003","CA004","CD001","CD002","CD003","CD004","CD005","CD006","ZA001","ZA002","ZA003","ZA004","ZA005","ZA006","ZB001","ZB002","ZB003","ZB004","ZB005","ZB006","ZB007","ZB008","ZB009","ZB010","ZC001","ZC002","ZC003","ZC004","ZC005","ZC006","ZC007","ZC008","ZC009","ZC010","ZC011","ZC012","ZC013","ZC014","ZC015","ZC016","ZC017","ZC018","ZC019","ZC020","ZC021","ZC022","ZC023","tag"]

def convert_annotation(year, image_id, list_file):
    in_file = open('VOCdevkit/VOC%s/Annotations/%s.xml'%(year, image_id))
    tree=ET.parse(in_file)
    root = tree.getroot()
    list_file.write('%s/VOCdevkit/VOC%s/JPEGImages/%s.jpg'%(wd, year, image_id))
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))

    list_file.write('\n')

for year, image_set in sets:
    image_ids = open('VOCdevkit/VOC%s/ImageSets/Main/%s.txt'%(year, image_set)).read().strip().split()
    list_file = open('%s_%s.txt'%(year, image_set), 'w')
    for image_id in image_ids:
        convert_annotation(year, image_id, list_file)
    list_file.close()
