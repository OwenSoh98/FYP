import cv2
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from openvino.inference_engine import IECore
from realsense.Camera import *
import math

def nms(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep

def filter(img, distance):
        """ Set pixels to 0 when logic 0 """
        for i in range(3):
            img[:,:,i] = np.multiply(img[:,:,i], distance)
        return img

def filter_distance(img, distance, d):
        """ Separate images into 2 parts, greater than distance and less than distance"""
        distance_smaller = np.copy(distance)
        distance_greater = np.copy(distance)

        distance_smaller[distance_smaller < d] = 1
        distance_smaller[distance_smaller >= d] = 0
        
        distance_greater[distance_greater < d] = 0
        distance_greater[distance_greater >= d] = 1

        img_smaller = np.copy(img)
        img_greater = np.copy(img)

        img_greater = filter(img_greater, distance_greater)
        img_smaller = filter(img_smaller, distance_smaller)

        return img_smaller, img_greater

def propose_region(img, depth):
    height, width, _ = img.shape

    Moments = cv2.moments(depth)
    xc = int(Moments["m10"] / Moments["m00"])
    yc = int(Moments["m01"] / Moments["m00"])

    n = np.count_nonzero(img[:,:,1])

    arr = img[:,:,1]
    arr = arr[::-1,:]

    foo = (arr!=0).argmax(axis=0)
    h = round(height - round(np.average(foo)))
    w = round(n / h)

    x1 = int(xc - w/2)
    x2 = int(xc + w/2)
    y1 = int(yc - h/2)
    y2 = int(yc + h/2)

    if x1 < 0:
        x1 = 0
    
    if x2 > width:
        x2 = width

    if y1 < 0:
        y1 = 0

    if y2 > height:
        y2 = height

    return x1, y1, x2, y2, xc, yc

def create_folder(directory):
    """ Given a path, create the directory """
    directory = os.path.dirname(directory)
    if not os.path.exists(directory):
        print(directory + ' does not exist...')
        os.makedirs(directory)
        print(directory + ' directory created...')

def main():
    ie = IECore()
    net = ie.read_network(
        model="./models/pedestrian-and-vehicle-detector-adas-0001/FP16-test/pedestrian-and-vehicle-detector-adas-0001.xml",
        weights="./models/pedestrian-and-vehicle-detector-adas-0001/FP16-test/pedestrian-and-vehicle-detector-adas-0001.bin",
    )
    exec_net = ie.load_network(net, "CPU")

    output_layer_ir = next(iter(exec_net.outputs))
    input_layer_ir = next(iter(exec_net.input_info))
    N, C, H, W = net.input_info[input_layer_ir].tensor_desc.dims

    objects = ['Background', 'Vehicle', 'Pedestrian']
    path = './raw_data/28_03_2022-12_11_52-filtered/'
    #path = './raw_data/28_03_2022-11_20_14-filtered/'
    d = 25000

    for filename in os.listdir(path):
        if not filename.endswith('.jpg'):
            continue
        print(filename)
        depth_filename = '-'.join(os.path.splitext(filename)[0].split('-')[0:-1]) + '-DEPTH'
        depth_frame = np.load(os.path.join(path, depth_filename + '.npy'))
        ori_rgb_frame = cv2.imread(os.path.join(path, filename))
        rgb_frame1 = np.copy(ori_rgb_frame)
        rgb_frame2 = np.copy(ori_rgb_frame)
        height, width, _ = ori_rgb_frame.shape

        t1 = time.time()
        ######################## IMAGE 1 ########################
        resized_rgb_frame = cv2.resize(ori_rgb_frame, (W, H))
        input_image = np.expand_dims(resized_rgb_frame.transpose(2, 0, 1), 0)
        
        result = exec_net.infer(inputs={input_layer_ir: input_image})['detection_out']  
        result = np.squeeze(np.squeeze(result, axis=0), axis=0)

        arr1 = np.empty((0, 6))
        for i in range(20):
            conf = round(float(result[i,2] * 100), 2)

            if int(conf) == 0:
                continue

            print(result[i,:])
            obj = objects[int(result[i,1])]
            x1 = int(result[i,3] * width)
            y1 = int(result[i,4] * height)
            x2 = int(result[i,5] * width)
            y2 = int(result[i,6] * height)
            rgb_frame1 = cv2.rectangle(rgb_frame1, (x1, y1), (x2, y2), (0, 0, 255), 2)
            # rgb_frame2 = cv2.rectangle(rgb_frame2, (x1, y1), (x2, y2), (0, 0, 255), 2)
            # msg = '{}({}%)'.format(obj, conf)
            # rgb_frame1 = cv2.putText(rgb_frame1, msg, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            # rgb_frame2 = cv2.putText(rgb_frame2, msg, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

            arr1 = np.append(arr1, np.array([[int(result[i,1]), x1, y1, x2, y2, conf]]), axis=0)

        create_folder('./mAP/original/')

        with open('./mAP/original/' + os.path.splitext(filename)[0] + '.txt', mode='w') as f:
            for i in range(len(arr1)):
                idx, x1, y1, x2, y2, conf = arr1[i,:].astype(int)
                content = '{} {} {} {} {} {}\n'.format(objects[idx], conf/100, x1, y1, x2, y2)
                f.write(content)
            f.close()

        ######################## IMAGE 2 ########################
        _, img_greater = filter_distance(ori_rgb_frame, depth_frame, d)
        x1c, y1c, x2c, y2c, xcc, ycc = propose_region(img_greater, depth_frame)

        cropped_img = ori_rgb_frame[y1c:y2c, x1c:x2c, :]
        height, width, _ = cropped_img.shape

        resized_rgb_frame = cv2.resize(cropped_img, (W, H))
        input_image = np.expand_dims(resized_rgb_frame.transpose(2, 0, 1), 0)

        result = exec_net.infer(inputs={input_layer_ir: input_image})['detection_out']  
        result = np.squeeze(np.squeeze(result, axis=0), axis=0)

        for i in range(20):
            conf = round(float(result[i,2] * 100), 2)

            if int(conf) == 0:
                continue

            print(result[i,:])
            obj = objects[int(result[i,1])]
            x1 = int(result[i,3] * width) + x1c
            y1 = int(result[i,4] * height) + y1c
            x2 = int(result[i,5] * width) + x1c
            y2 = int(result[i,6] * height) + y1c
            # rgb_frame2 = cv2.rectangle(rgb_frame2, (x1, y1), (x2, y2), (0, 0, 255), 2)
            # msg = '{}({}%)'.format(obj, conf)
            # rgb_frame2 = cv2.putText(rgb_frame2, msg, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

            arr1 = np.append(arr1, np.array([[int(result[i,1]), x1, y1, x2, y2, conf]]), axis=0)

        preds = nms(arr1[:,1::], 0.5)
        print(preds)
        for pred in preds:
            idx, x1, y1, x2, y2, conf = arr1[pred].astype(int)
            rgb_frame2 = cv2.rectangle(rgb_frame2, (x1, y1), (x2, y2), (255, 0, 0), 2)
            # msg = '{}({}%)'.format(objects[idx], conf)
            # rgb_frame2 = cv2.putText(rgb_frame2, msg, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)

        create_folder('./mAP/improved-small/')

        with open('./mAP/improved-small/' + os.path.splitext(filename)[0] + '.txt', mode='w') as f:
            for pred in preds:
                idx, x1, y1, x2, y2, conf = arr1[pred,:].astype(int)
                content = '{} {} {} {} {} {}\n'.format(objects[idx], conf/100, x1, y1, x2, y2)
                f.write(content)
            f.close()

        ########################################################
        t2 = time.time()
        print('Inference time: {}. FPS: {}.'.format(t2-t1, 1/(t2-t1)))

        # rgb_frame2 = cv2.rectangle(rgb_frame2, (x1c, y1c), (x2c, y2c), (255, 0, 0), 2)
        # img_greater = cv2.rectangle(img_greater, (x1c, y1c), (x2c, y2c), (255, 0, 0), 2)

        cv2.imshow('rgb1', rgb_frame1)
        cv2.imshow('rgb2', rgb_frame2)
        # cv2.imshow('rgb-greater', img_greater)
        cv2.waitKey(0)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()