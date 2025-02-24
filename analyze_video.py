#!/usr/bin/env python3

import cv2
import numpy as np
import torch

from dlclive import DLCLive, Processor
from dlclive.display import Display


def analyze_video2(video_path: str, dlc_live):
    # Load video
    cap = cv2.VideoCapture(video_path)
    poses = []
    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        # Prepare the frame for the model
        frame = np.array(frame, dtype=np.float32)
        frame = np.transpose(frame, (2, 0, 1))
        frame = frame.reshape(1, frame.shape[0], frame.shape[1], frame.shape[2])
        frame = frame / 255.0

        # Analyze the frame using the get_pose function
        pose = dlc_live.get_pose(frame)

        # Store the pose for this frame
        poses.append(pose)

        frame_index += 1
        print(frame_index)

    # Release the video capture object
    cap.release()

    return poses


def main():
    # Paths provided by you
    video_path = "/Users/annastuckert/Documents/DLC_AI_Residency/DLC_AI2024/DeepLabCut-live/Ventral_gait_model/1_20cms_0degUP_first_1s.avi"
    model_dir = "/Users/annastuckert/Documents/DLC_AI_Residency/DLC_AI2024/DeepLabCut-live/Ventral_gait_model/train"
    snapshot = "/Users/annastuckert/Documents/DLC_AI_Residency/DLC_AI2024/DeepLabCut-live/Ventral_gait_model/train/snapshot-263.pt"
    model_type = "pytorch"

    # Initialize the DLCLive model
    dlc_proc = Processor()
    dlc_live = DLCLive(
        pytorch_cfg=model_dir,
        processor=dlc_proc,
        snapshot=snapshot,
        model_type=model_type,
    )

    # Analyze the video
    poses = analyze_video2(video_path, dlc_live)
    print("Pose analysis complete.")


if __name__ == "__main__":
    main()
