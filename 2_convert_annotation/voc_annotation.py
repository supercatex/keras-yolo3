import xml.etree.ElementTree as ET
import os


sets=[('2012', 'train'), ('2012', 'val'), ('2012', 'test')]

classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

basedir = "../../datasets"


def convert_annotation(year, image_id, list_file):
    in_file = open('%s/VOCdevkit/VOC%s/Annotations/%s.xml' % (basedir, year, image_id))
    tree=ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))


for year, image_set in sets:
    image_ids = open('%s/VOCdevkit/VOC%s/ImageSets/Main/%s.txt'%(basedir, year, image_set)).read().strip().split()
    list_file = open('%s_%s.txt'%(year, image_set), 'w')
    for image_id in image_ids:
        if not os.path.exists('%s/VOCdevkit/VOC%s/JPEGImages/%s.jpg'%(basedir, year, image_id)):
            print("jpg not found: %s" % image_id)
            continue
        if not os.path.exists('%s/VOCdevkit/VOC%s/Annotations/%s.xml' % (basedir, year, image_id)):
            print("xml not found: %s" % image_id)
            continue

        list_file.write('%s/VOCdevkit/VOC%s/JPEGImages/%s.jpg'%(basedir, year, image_id))
        convert_annotation(year, image_id, list_file)
        list_file.write('\n')
        print("%s done." % image_id)
    list_file.close()
print("END")
