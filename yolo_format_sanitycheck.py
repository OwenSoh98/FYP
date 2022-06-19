import os
import cv2
import numpy as np
from lxml import etree


def resize(image, ratio):
    wid = int(image.shape[1] / ratio)
    hei = int(image.shape[0] / ratio)
    image = cv2.resize(image, (wid, hei))
    return image


def draw_bbox(img_path, xml_path):
    # Test the image
    img = cv2.imread(img_path)
    f = open(xml_path, 'r')
    w = img.shape[1]
    h = img.shape[0]

    for line in f:
        # Separate by spaces
        data = line.split()

        # Getting data
        classes = int(data[0])
        x_center = float(data[1])
        y_center = float(data[2])
        width = float(data[3])
        height = float(data[4])

        # Finding x1, x2, y1, y2 coordinates
        x1 = int((x_center - width / 2) * w)
        y1 = int((y_center - height / 2) * h)
        x2 = int((x_center + width / 2) * w)
        y2 = int((y_center + height / 2) * h)

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
    cv2.waitKey()


if __name__ == "__main__":
    # Path to folder
    img = './data/5000-greater/YOLOv5-Training/5000-greater/images/train/'
    xml = './data/5000-greater/YOLOv5-Training/5000-greater/labels/train/'

    for filename in os.listdir(img):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img_path = os.path.join(img, filename)
            basename = os.path.splitext(os.path.basename(img_path))[0]
            xml_path = os.path.join(xml, basename + '.txt')
            print(img_path)
            draw_bbox(img_path, xml_path)
