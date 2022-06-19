import os
import shutil
import numpy as np
from lxml import etree
import cv2

from TrainingFormatter import *
from SeparateData import *

class FilterData:
    def __init__(self, distance):
        # Distance
        self.distance = distance

        # Filepath
        self.train_folder = "./processed_data/train/"
        self.val_folder = "./processed_data/val/"
        self.test_folder = "./processed_data/test/"

        self.dst_train_folder = './temp_data/' + str(distance) + '/train/'
        self.dst_val_folder = './temp_data/' + str(distance) + '/val/'
        self.dst_test_folder = './temp_data/' + str(distance) + '/test/'

        self.create_folder(self.dst_train_folder)
        self.create_folder(self.dst_val_folder)
        self.create_folder(self.dst_test_folder)

        self.loop(self.train_folder, self.dst_train_folder)
        self.loop(self.val_folder, self.dst_val_folder)
        self.loop(self.test_folder, self.dst_test_folder)

        self.create_folder('./data/')
        SeparateData('./temp_data/' + str(distance), distance)
        #TrainingFormatter(self.dst_train_folder, self.dst_val_folder, self.dst_test_folder, self.distance)

    def create_folder(self, directory):
        """ Given a path, create the directory """
        directory = os.path.dirname(directory)
        if not os.path.exists(directory):
            print(directory + ' does not exist...')
            os.makedirs(directory)
            print(directory + ' directory created...')

    def filter(self, img, distance):
        """ Set pixels to 0 when logic 0 """
        for i in range(3):
            img[:,:,i] = np.multiply(img[:,:,i], distance)
        return img

    def filter_distance(self, img, distance):
        """ Separate images into 2 parts, greater than distance and less than distance"""
        distance_smaller = np.copy(distance)
        distance_greater = np.copy(distance)

        distance_smaller[distance_smaller < self.distance] = 1
        distance_smaller[distance_smaller >= self.distance] = 0
        
        distance_greater[distance_greater < self.distance] = 0
        distance_greater[distance_greater >= self.distance] = 1

        img_smaller = np.copy(img)
        img_greater = np.copy(img)

        img_greater = self.filter(img_greater, distance_greater)
        img_smaller = self.filter(img_smaller, distance_smaller)

        return img_smaller, img_greater

    def filter_xml(self, xml_path, distance, args):
        """ Filter by xml """
        my_tree = etree.parse(xml_path)
        my_root = my_tree.getroot()

        for x in my_root.findall('object'):
            name = x.find('name').text
            bbox = x.find('bndbox')
            x1 = float(bbox.find('xmin').text)
            y1 = float(bbox.find('ymin').text)
            x2 = float(bbox.find('xmax').text)
            y2 = float(bbox.find('ymax').text)

            x_center = int(x1 + (x2 - x1) / 2)
            y_center = int(y1 + (y2 - y1) / 2)
            d = distance[y_center, x_center]

            if args == "greater":
                if d < self.distance:
                    x.getparent().remove(x)
            elif args == "smaller":
                if d > self.distance:
                    x.getparent().remove(x)
            
        my_tree.write(xml_path, pretty_print=True)

    def remove_zeros(self, path):
        """ Filter images with no objects inside """
        for file in os.listdir(path):   
            if file.endswith('.xml'):
                filepath = os.path.join(path, file)

                my_tree = etree.parse(filepath)
                my_root = my_tree.getroot()

                count = 0
                for x in my_root.findall('object'):
                    count += 1
                
                if count == 0:
                    os.remove(filepath)
                    img_filepath = os.path.splitext(filepath)[0] + '.jpg'
                    os.remove(img_filepath)

    def filter_images(self, src, tgt):
        """ Apply a mask to images by distances """
        for file in os.listdir(src):   
            if file.endswith('.jpg'):
                """ Image and moving file Operations """
                basename = os.path.splitext(os.path.basename(file))[0].split("RGB")[0]

                src_xml_path = src + basename + 'RGB.xml'
                shutil.copy(src_xml_path, tgt + basename + 'smaller.xml')
                shutil.copy(src_xml_path, tgt + basename + 'greater.xml')

                src_distance = src + basename + 'DEPTH.npy'
                distance = np.load(src_distance)

                img = cv2.imread(src + file)
                img_smaller, img_greater = self.filter_distance(img, distance)

                cv2.imwrite(tgt + basename + 'smaller.jpg', img_smaller)
                cv2.imwrite(tgt + basename + 'greater.jpg', img_greater)

                """ XML Operations """
                self.filter_xml(tgt + basename + 'smaller.xml', distance, "smaller")
                self.filter_xml(tgt + basename + 'greater.xml', distance, "greater")

        self.remove_zeros(tgt)

    def loop(self, src_path, tgt_path):
        for folder in os.listdir(src_path):
            folder_path = os.path.join(src_path, folder) + '/'
            print(folder_path)
            self.filter_images(folder_path, tgt_path)

for i in range(5000, 60000, 2500):
    print(i)
    FilterData(i)
