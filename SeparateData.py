import os
import shutil

from TrainingFormatter import *

class SeparateData:
    def __init__(self, src, distance):
        # Filepath
        self.src_folder = src
        self.greater_folder = self.src_folder + '-greater/'
        self.smaller_folder = self.src_folder + '-smaller/'

        self.create_folder(self.greater_folder)
        self.create_folder(self.smaller_folder)
        self.src_folder = self.src_folder + '/'

        self.greater_train_folder = self.greater_folder + 'train/'
        self.greater_val_folder = self.greater_folder + 'val/'
        self.greater_test_folder = self.greater_folder + 'test/'

        self.smaller_train_folder = self.smaller_folder + 'train/'
        self.smaller_val_folder = self.smaller_folder + 'val/'
        self.smaller_test_folder = self.smaller_folder + 'test/'

        self.loop()
        TrainingFormatter(self.greater_train_folder, self.greater_val_folder, self.greater_test_folder, str(distance) + '-greater')
        TrainingFormatter(self.smaller_train_folder, self.smaller_val_folder, self.smaller_test_folder, str(distance) + '-smaller')

    def create_folder(self, directory):
        """ Given a path, create the directory """
        directory = os.path.dirname(directory)
        if not os.path.exists(directory):
            print(directory + ' does not exist...')
            os.makedirs(directory)
            print(directory + ' directory created...')

    def separate_images(self, src, greater, smaller):
        self.create_folder(greater)
        self.create_folder(smaller)

        for file in os.listdir(src):   
            if file.endswith('.jpg'):
                """ Moving file Operations """
                src_img_path = os.path.join(src, file)
                src_xml_path = os.path.splitext(src_img_path)[0] + '.xml'

                affix = os.path.splitext(os.path.basename(file))[0].split("-")[-1]
                #print(src_img_path)
                if affix == 'greater':
                    shutil.copy(src_img_path, os.path.join(greater, os.path.basename(src_img_path)))
                    shutil.copy(src_xml_path, os.path.join(greater, os.path.basename(src_xml_path)))
                elif affix == 'smaller':
                    shutil.copy(src_img_path, os.path.join(smaller, os.path.basename(src_img_path)))
                    shutil.copy(src_xml_path, os.path.join(smaller, os.path.basename(src_xml_path)))
    
    def loop(self):
        """ Loop """
        for folder in os.listdir(self.src_folder):
            path = os.path.join(self.src_folder, folder)
            if folder == "train":
                self.separate_images(path, self.greater_train_folder, self.smaller_train_folder)
            elif folder == "val":
                self.separate_images(path, self.greater_val_folder, self.smaller_val_folder)
            elif folder == "test":
                self.separate_images(path, self.greater_test_folder, self.smaller_test_folder)
