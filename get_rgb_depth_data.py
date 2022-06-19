import os
import time
import cv2
import numpy as np
from datetime import datetime
from realsense.Camera import *


class GetData:
    def __init__(self):
        self.timeframe = 1

        # Filepath
        self.folder_path = './raw_data/'
        self.data_path= self.create_data_folder()

        # Serial numbers of cameras
        self.depth_serial = '841512070568'
        self.tracking_serial = '845412110856'

        self.collect_camera_data()

    def create_folder(self, directory):
        """ Given a path, create the directory """
        directory = os.path.dirname(directory)
        if not os.path.exists(directory):
            print(directory + ' does not exist...')
            os.makedirs(directory)
            print(directory + ' directory created...')

    def create_data_folder(self):
        """ Create date and time as unique identifier as folder name """
        now = datetime.now().strftime("%d_%m_%Y-%H_%M_%S")
        now = os.path.join(self.folder_path, now) + '/'
        self.create_folder(now)
        return now

    def collect_camera_data(self):
        print('Collecting data at ' + str(1/self.timeframe) + ' fps')
        dc = DepthCamera(self.depth_serial, 640, 480, 30)
        t0 = datetime.now().timestamp()

        while True:
            # To compare time
            t1 = datetime.now().timestamp()

            if t1 - t0 > self.timeframe:
                ret, rgb_frame, depth_frame, heatmap_frame = dc.get_frame()

                # Get current time
                d = datetime.now()
                dt = d.strftime("%d-%m-%Y-%H-%M-%S-")
                print('Captured Data at: ' + dt)

                # Save RGBD image data
                # cv2.imwrite(self.data_path + str(dt) + 'DEPTH.jpg', heatmap_frame)
                np.save(self.data_path + str(dt) + 'DEPTH.npy', depth_frame)
                cv2.imwrite(self.data_path + str(dt) + 'RGB.jpg', rgb_frame)

                # Display RGBD image data
                cv2.imshow('depth color frame', heatmap_frame)
                cv2.imshow('color frame', rgb_frame)

                # Update compare time
                t0 = datetime.now().timestamp()

            if cv2.waitKey(1) & 0xff == ord('q'):
                break

GetData()
