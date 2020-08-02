"""
DeepLabCut Toolbox (deeplabcut.org)
© A. & M. Mathis Labs

Licensed under GNU Lesser General Public License v3.0
"""


import serial
import struct
import time
import numpy as np

from dlclive import Processor


class MouseLickLED(Processor):
    def __init__(self, com, lik_thresh=0.5, baudrate=int(9600)):

        super().__init__()
        self.ser = serial.Serial(com, baudrate, timeout=0)
        self.lik_thresh = lik_thresh
        self.lick_frame_time = []
        self.out_time = []
        self.in_time = []

    def close_serial(self):

        self.ser.close()

    def switch_led(self):

        ### flush input buffer ###

        self.ser.reset_input_buffer()

        ### turn on IR LED ###

        self.out_time.append(time.time())
        self.ser.write(b"I")

        ### wait for receiver ###

        while True:
            led_byte = self.ser.read()
            if len(led_byte) > 0:
                break
        self.in_time.append(time.time())

    def process(self, pose, **kwargs):

        ### bodyparts
        # 0. pupil-top
        # 1. pupil-left
        # 2. pupil-bottom
        # 3. pupil-right
        # 4. lip-upper
        # 5. lip-lower
        # 6. tongue
        # 7. tube

        if kwargs["record"]:
            if pose[6, 2] > self.lik_thresh:
                self.lick_frame_time.append(kwargs["frame_time"])
                self.switch_led()

        return pose

    def save(self, filename):

        ### save stim on and stim off times

        filename += ".npy"
        out_time = np.array(self.out_time)
        in_time = np.array(self.in_time)
        frame_time = np.array(self.lick_frame_time)
        try:
            np.savez(
                filename, out_time=out_time, in_time=in_time, frame_time=frame_time
            )
            save_code = True
        except Exception:
            save_code = False

        return save_code
