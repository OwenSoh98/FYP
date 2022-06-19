import pyrealsense2 as rs
import numpy as np


class DepthCamera:
    def __init__(self, serial, width, height, fps):
        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        self.colorizer = rs.colorizer()
        self.align = rs.align(rs.stream.color)

        # Build config object and request pose data
        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

        # Start streaming
        self.pipeline.start(config)

    def get_frame(self):
        """ Returns RET, RGB, DEPTH, HEATMAP"""
        frames = self.pipeline.wait_for_frames()

        # Align the depth frame to color frame
        aligned_frames = self.align.process(frames)

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        depth_image = np.asanyarray(depth_frame.get_data())
        rgb_image = np.asanyarray(color_frame.get_data())
        heatmap_image = np.asanyarray(self.colorizer.colorize(depth_frame).get_data())

        if not depth_frame or not color_frame:
            return False, None, None, None

        return True, rgb_image, depth_image, heatmap_image

    def release(self):
        """ Stop Camera """
        self.pipeline.stop()


class TrackingCamera:
    def __init__(self, serial):
        # Configure depth and color streams
        self.pipeline = rs.pipeline()

        # Build config object and request pose data
        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.pose)

        # Start streaming
        self.pipeline.start(config)

    def get_position(self):
        # Wait for the next set of frames from the camera
        frames = self.pipeline.wait_for_frames()

        # Fetch pose frame
        pose = frames.get_pose_frame()

        if pose:
            data = pose.get_pose_data()
            return data.translation, data.velocity, data.acceleration
        else:
            return None, None, None

    def release(self):
        """ Stop Camera """
        self.pipeline.stop()
