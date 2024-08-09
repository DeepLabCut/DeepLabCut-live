from dlclive import DLCLive, Processor
import dlclive
import cv2

dlc_proc = Processor()
dlc_live = DLCLive("/media1/data/dikra/dlc-live-tmp", processor=dlc_proc)
img = cv2.imread("/media1/data/dikra/fly-kevin/img001.png")
dlc_live.init_inference(frame=img)