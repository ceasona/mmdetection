import argparse
import os
import xml.etree.ElementTree as ET

sets = ['train', 'val', 'test']
classes = ['porosity', 'crack', 'tungsten', 'lop', 'slag', 'lof', 'false']
abs_path = os.getcwd()


def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return x, y, w, h


def convert_annotation(yolo_lable, xml_path, image_id):
    in_file = open('{}/{}.xml'.format(xml_path, image_id), encoding='UTF-8')
    out_file = open('{}/{}.txt'.format(yolo_lable, image_id), 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        # difficult = obj.find('Difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text),
             float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        b1, b2, b3, b4 = b
        # 标注越界修正
        if b2 > w:
            b2 = w
        if b4 > h:
            b4 = h
        b = (b1, b2, b3, b4)
        bb = convert((w, h), b)
        out_file.write(
            str(cls_id) + ' ' + ' '.join([str(a) for a in bb]) + '\n')


parser = argparse.ArgumentParser()
parser.add_argument(
    '--xml_path',
    default='data/VOCdevkit/VOC2007/Annotations',
    type=str,
    help='存放voc格式标签的文件夹')
parser.add_argument(
    '--txt_path',
    default='data/VOCdevkit/VOC2007/ImageSets/Main',
    type=str,
    help='存放split_train_val.py 中生成的4个txt 路径')
parser.add_argument(
    '--yolo_lable_path',
    default='data/VOCdevkit/VOC2007/Labels',
    type=str,
    help='存放yolo格式标签数据的路径')
parser.add_argument(
    '--yolo_split_lable_path',
    default='data/',
    type=str,
    help='存放yolo格式切分数据集的txt路径')
parser.add_argument(
    '--images_path',
    default='data/VOCdevkit/VOC2007//JPEGImages',
    type=str,
    help='存放jpg路径')
opt = parser.parse_args()

xml_path = opt.xml_path
txt_path = opt.txt_path
yolo_lable_path = opt.yolo_lable_path
yolo_split_lable_path = opt.yolo_split_lable_path
images_path = opt.images_path

for i in (xml_path, txt_path, yolo_lable_path, yolo_split_lable_path,
          images_path):
    if os.path.exists(i):
        pass
    else:
        os.makedirs(i)

for image_set in sets:
    image_ids = open('{}/{}.txt'.format(txt_path,
                                        image_set)).read().strip().split()
    list_file = open('{}/{}.txt'.format(yolo_split_lable_path, image_set), 'w')
    for image_id in image_ids:
        list_file.write('{}/{}.jpg\n'.format(images_path, image_id))
        convert_annotation(yolo_lable_path, xml_path, image_id)
    list_file.close()
