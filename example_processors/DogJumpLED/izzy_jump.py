"""
DeepLabCut Toolbox (deeplabcut.org)
© A. & M. Mathis Labs

Licensed under GNU Lesser General Public License v3.0
"""


import serial
import struct
import time
import numpy as np

from dlclive.processor import Processor, KalmanFilterPredictor


class IzzyJump(Processor):
    def __init__(self, com="", lik_thresh=0.5, baudrate=int(9600), **kwargs):

        super().__init__()
        self.ser = serial.Serial(com, baudrate, timeout=0)
        self.lik_thresh = lik_thresh
        self.led_times = []
        self.last_light = 0

    def close_serial(self):

        self.ser.close()

    def switch_led(self, val, frame_time):

        ### check status of led ###

        self.ser.write(b"R")

        led_byte = b""
        led_status = None
        while (len(led_byte) != 0) or (led_status is None):
            led_byte = self.ser.read()
            if len(led_byte) > 0:
                led_status = ord(led_byte)

        if led_status != val:
            ctime = time.time()
            if ctime - self.last_light > 0.25:
                self.ser.write(b"L")
                self.last_light = ctime
                self.led_times.append((val, frame_time, ctime))

    def process(self, pose, **kwargs):

        ### bodyparts
        # 0. nose
        # 1. L-eye
        # 2. R-eye
        # 3. L-ear
        # 4. R-ear
        # 5. Throat
        # 6. Withers
        # 7. Tailset
        # 8. L-front-paw
        # 9. R-front-paw
        # 10. L-front-wrist
        # 11. R-front-wrist
        # 12. L-front-elbow
        # 13. R-front-elbow
        # ...

        l_elbow = pose[12, 1] if pose[12, 2] > self.lik_thresh else None
        r_elbow = pose[13, 1] if pose[13, 2] > self.lik_thresh else None
        elbows = [l_elbow, r_elbow]
        this_elbow = (
            min([e for e in elbows if e is not None])
            if any([e is not None for e in elbows])
            else None
        )

        withers = pose[6, 1] if pose[6, 2] > self.lik_thresh else None

        if kwargs["record"]:
            if withers is not None and this_elbow is not None:
                if this_elbow < withers:
                    self.switch_led(True, kwargs["frame_time"])
                else:
                    self.switch_led(False, kwargs["frame_time"])

        return pose

    def save(self, filename):

        ### save stim on and stim off times

        if filename[-4:] != ".npy":
            filename += ".npy"
        arr = np.array(self.led_times, dtype=float)
        try:
            np.save(filename, arr)
            save_code = True
        except Exception:
            save_code = False

        return save_code


class IzzyJumpKF(KalmanFilterPredictor, IzzyJump):
    def __init__(
        self,
        com="",
        lik_thresh=0.5,
        baudrate=int(9600),
        adapt=True,
        forward=0.003,
        fps=30,
        nderiv=2,
        priors=[1, 1],
        initial_var=1,
        process_var=1,
        dlc_var=4,
    ):

        super().__init__(
            adapt=adapt,
            forward=forward,
            fps=fps,
            nderiv=nderiv,
            priors=priors,
            initial_var=initial_var,
            process_var=process_var,
            dlc_var=dlc_var,
            com=com,
            lik_thresh=lik_thresh,
            baudrate=baudrate,
        )

    def process(self, pose, **kwargs):

        future_pose = KalmanFilterPredictor.process(self, pose, **kwargs)
        final_pose = IzzyJump.process(self, future_pose, **kwargs)
        return final_pose

    def save(self, filename):

        return IzzyJump.save(self, filename)
