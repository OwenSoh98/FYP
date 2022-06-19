import os
import cv2
import numpy as np
from lxml import etree


def resize(image, ratio):
    wid = int(image.shape[1] / ratio)
    hei = int(image.shape[0] / ratio)
    image = cv2.resize(image, (wid, hei))
    return image


def draw_bbox(img_name):
    # Test the image
    img = cv2.imread(img_name)
    name_wo_ext = os.path.splitext(img_name)[0]
    xml_path = name_wo_ext + '.xml'
    w = img.shape[1]
    h = img.shape[0]

    xml_file = open(xml_path)
    my_tree = etree.parse(xml_file)
    my_root = my_tree.getroot()

    for x in my_root.findall('object'):
        classes = x.find('name').text
        # Find ground truth bounding box
        bbox = x.find('bndbox')
        x1 = int(float((bbox.find('xmin').text)))
        y1 = int(float((bbox.find('ymin').text)))
        x2 = int(float((bbox.find('xmax').text)))
        y2 = int(float((bbox.find('ymax').text)))

        # Draw box
        img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
        img = cv2.putText(img, 'class:{}'.format(classes), 
            (x1, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255))

    if w > 3000:
        img = resize(img, 8)
    elif w > 2000:
        img = resize(img, 4)
    elif w > 1000:
        img = resize(img, 2)

    cv2.imshow('demo', img)
    cv2.imwrite('test.jpg', img)
    cv2.waitKey()


if __name__ == "__main__":
    # Path to folder
    dataset_path = './raw_data/28_03_2022-12_11_52-filtered/'

    for folder_name in os.listdir(dataset_path):
        if folder_name.endswith('.jpg'):
            if folder_name.endswith('.jpg') or folder_name.endswith('.png'):
                img_path = os.path.join(dataset_path, folder_name)
                print(img_path)
                draw_bbox(img_path)
        # elif folder_name.endswith('.xml'):
        #     continue
        # else:
        #     # Get the path of the folder
        #     folder_path = os.path.join(dataset_path, folder_name)
        #     for filename in os.listdir(folder_path):
        #         # Check if folder is in yolo format
        #         if filename.endswith('.jpg') or filename.endswith('.png'):
        #             for filename in os.listdir(folder_path):
        #                 if filename.endswith('.jpg') or filename.endswith('.png'):
        #                     img_path = os.path.join(folder_path, filename)
        #                     print(img_path)
        #                     draw_bbox(img_path)
