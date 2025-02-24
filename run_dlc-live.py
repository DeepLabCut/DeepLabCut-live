import numpy as np
from PIL import Image

from dlclive import DLCLive, Processor

image = Image.open(
    "/Users/annastuckert/Downloads/exported DLC model for dlc-live/img049.png"
)
img = np.asarray(image)

dlc_proc = Processor()
dlc_live = DLCLive(
    "/Users/annastuckert/Downloads/exported DLC model for dlc-live/DLC_dev-single-animal_resnet_50_iteration-1_shuffle-1",
    processor=dlc_proc,
)
dlc_live.init_inference(img)
pose = dlc_live.get_pose(img)
print(pose)
