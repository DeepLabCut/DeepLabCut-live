import csv
import os

import colorcet as cc
import cv2
import h5py
import numpy as np
from PIL import ImageColor


def analyze_video(
    video_path: str,
    dlc_live,
    pcutoff=0.5,
    display_radius=5,
    resize=None,
    save_poses=False,
    save_dir="model_predictions",
    draw_keypoint_names=False,
    cmap="bmy",
):
    """
    Analyze a video to track keypoints using an imported DeepLabCut model, visualize keypoints on the video, and optionally save the keypoint data and the labelled video.

    Parameters:
    -----------
    video_path : str
        The path to the video file to be analyzed.
    dlc_live : DLCLive
        An instance of the DLCLive class.
    pcutoff : float, optional, default=0.5
        The probability cutoff value below which keypoints are not visualized.
    display_radius : int, optional, default=5
        The radius of the circles drawn to represent keypoints on the video frames.
    resize : tuple of int (width, height) or None, optional, default=None
        The size to which the frames should be resized. If None, the frames are not resized.
    save_poses : bool, optional, default=False
        Whether to save the detected poses to CSV and HDF5 files.
    save_dir : str, optional, default="model_predictions"
        The directory where the output video and pose data will be saved.
    draw_keypoint_names : bool, optional, default=False
        Whether to draw the names of the keypoints on the video frames.
    cmap : str, optional, default="bmy"
        The colormap from the colorcet library to use for keypoint visualization.

    Returns:
    --------
    poses : list of dict
        A list of dictionaries where each dictionary contains the frame number and the corresponding pose data.
    """
    # Ensure save directory exists
    os.makedirs(name=save_dir, exist_ok=True)

    # Load video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    # Start empty dict to save poses to for each frame
    poses = []
    # Create variable indicate current frame. Later in the code +1 is added to frame_index
    frame_index = 0

    # Load the DLC model
    try:
        pose_model = dlc_live.load_model()
    except Exception as e:
        print(f"Error: Could not load DLC model. Details: {e}")
        return

    # Retrieve bodypart names and number of keypoints
    bodyparts = dlc_live.cfg["metadata"]["bodyparts"]
    num_keypoints = len(bodyparts)

    # Set colors and convert to RGB
    cmap_colors = getattr(cc, cmap)
    colors = [
        ImageColor.getrgb(color)
        for color in cmap_colors[:: int(len(cmap_colors) / num_keypoints)]
    ]

    # Define output video path
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_video_path = os.path.join(save_dir, f"{video_name}_DLCLIVE_LABELLED.mp4")

    # Get video writer setup
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width, frame_height = (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )

    if resize:
        frame_width, frame_height = resize
    vwriter = cv2.VideoWriter(
        filename=output_video_path,
        fourcc=fourcc,
        fps=fps,
        frameSize=(frame_width, frame_height),
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        try:
            pose = dlc_live.get_pose(frame, pose_model=pose_model)
        except Exception as e:
            print(f"Error analyzing frame {frame_index}: {e}")
            continue

        poses.append({"frame": frame_index, "pose": pose})

        # Visualize keypoints
        this_pose = pose["poses"][0][0]
        for j in range(this_pose.shape[0]):
            if this_pose[j, 2] > pcutoff:
                x, y = map(int, this_pose[j, :2])
                cv2.circle(
                    frame,
                    center=(x, y),
                    radius=display_radius,
                    color=colors[j],
                    thickness=-1,
                )

                if draw_keypoint_names:
                    cv2.putText(
                        frame,
                        text=bodyparts[j],
                        org=(x + 10, y),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        color=colors[j],
                        thickness=1,
                        lineType=cv2.LINE_AA,
                    )

        if resize:
            frame = cv2.resize(src=frame, dsize=resize)

        vwriter.write(image=frame)
        frame_index += 1

    cap.release()
    vwriter.release()

    if save_poses:
        save_poses_to_files(video_path, save_dir, bodyparts, poses)

    return poses


def save_poses_to_files(video_path, save_dir, bodyparts, poses):
    """
    Save the keypoint poses detected in the video to CSV and HDF5 files.

    Parameters:
    -----------
    video_path : str
        The path to the video file that was analyzed.
    save_dir : str
        The directory where the pose data files will be saved.
    bodyparts : list of str
        A list of body part names corresponding to the keypoints.
    poses : list of dict
        A list of dictionaries where each dictionary contains the frame number and the corresponding pose data.

    Returns:
    --------
    None
    """
    base_filename = os.path.splitext(os.path.basename(video_path))[0]
    csv_save_path = os.path.join(save_dir, f"{base_filename}_poses.csv")
    h5_save_path = os.path.join(save_dir, f"{base_filename}_poses.h5")

    # Save to CSV
    with open(csv_save_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        header = ["frame"] + [
            f"{bp}_{axis}" for bp in bodyparts for axis in ["x", "y", "confidence"]
        ]
        writer.writerow(header)
        for entry in poses:
            frame_num = entry["frame"]
            pose = entry["pose"]["poses"][0][0]
            row = [frame_num] + [item for kp in pose for item in kp]
            writer.writerow(row)

    # Save to HDF5
    with h5py.File(h5_save_path, "w") as hf:
        hf.create_dataset(name="frames", data=[entry["frame"] for entry in poses])
        for i, bp in enumerate(bodyparts):
            hf.create_dataset(
                name=f"{bp}_x",
                data=[entry["pose"]["poses"][0][0][i, 0].item() for entry in poses],
            )
            hf.create_dataset(
                name=f"{bp}_y",
                data=[entry["pose"]["poses"][0][0][i, 1].item() for entry in poses],
            )
            hf.create_dataset(
                name=f"{bp}_confidence",
                data=[entry["pose"]["poses"][0][0][i, 2].item() for entry in poses],
            )
