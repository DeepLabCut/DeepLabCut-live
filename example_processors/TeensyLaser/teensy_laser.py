"""
DeepLabCut Toolbox (deeplabcut.org)
© A. & M. Mathis Labs

Licensed under GNU Lesser General Public License v3.0
"""


from dlclive.processor.processor import Processor
import serial
import struct
import pickle
import time


class TeensyLaser(Processor):
    def __init__(
        self, com, baudrate=115200, pulse_freq=50, pulse_width=5, max_stim_dur=0
    ):

        super().__init__()
        self.ser = serial.Serial(com, baudrate)
        self.pulse_freq = pulse_freq
        self.pulse_width = pulse_width
        self.max_stim_dur = (
            max_stim_dur if (max_stim_dur >= 0) and (max_stim_dur < 65356) else 0
        )
        self.stim_on = False
        self.stim_on_time = []
        self.stim_off_time = []

    def close_serial(self):

        self.ser.close()

    def turn_stim_on(self):

        # command to activate PWM signal to laser is the letter 'O' followed by three 16 bit integers -- pulse frequency, pulse width, and max stim duration
        if not self.stim_on:
            self.ser.write(
                b"O"
                + struct.pack(
                    "HHH", self.pulse_freq, self.pulse_width, self.max_stim_dur
                )
            )
            self.stim_on = True
            self.stim_on_time.append(time.time())

    def turn_stim_off(self):

        # command to turn off PWM signal to laser is the letter 'X'
        if self.stim_on:
            self.ser.write(b"X")
            self.stim_on = False
            self.stim_off_time.append(time.time())

    def process(self, pose, **kwargs):

        # define criteria to stimulate (e.g. if first point is in a corner of the video)
        box = [[0, 100], [0, 100]]
        if (
            (pose[0][0] > box[0][0])
            and (pose[0][0] < box[0][1])
            and (pose[0][1] > box[1][0])
            and (pose[0][1] < box[1][1])
        ):
            self.turn_stim_on()
        else:
            self.turn_stim_off()

        return pose

    def save(self, file=None):

        ### save stim on and stim off times
        save_code = 0
        if file:
            try:
                pickle.dump(
                    {"stim_on": self.stim_on_time, "stim_off": self.stim_off_time},
                    open(file, "wb"),
                )
                save_code = 1
            except Exception:
                save_code = -1
        return save_code
