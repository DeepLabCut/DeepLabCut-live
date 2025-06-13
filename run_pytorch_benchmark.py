from dlclive.pose_estimation_pytorch.runner import PyTorchRunner

import numpy as np

#single_animal = True
# Single Animal: Resnet
#model_path="/home/max/tmp/dog_single_animal-max-2025-06-06/exported-models-pytorch/DLC_dog_single_animal_resnet_50_iteration-0_shuffle-1/DLC_dog_single_animal_resnet_50_iteration-0_shuffle-1_snapshot-010.pt"
# Single Animal: Hrnet
#model_path="/home/max/tmp/dog_single_animal-max-2025-06-06/exported-models-pytorch/DLC_dog_single_animal_hrnet_w32_iteration-0_shuffle-2/DLC_dog_single_animal_hrnet_w32_iteration-0_shuffle-2_snapshot-030.pt"

single_animal = False
# Super Animal Hrnet 32
model_path="/home/max/tmp/export_torch_model/quadruped-max-2025-06-04/exported-models-pytorch/DLC_quadruped_hrnet_w32_iteration-0_shuffle-1/DLC_quadruped_hrnet_w32_iteration-0_shuffle-1_snapshot-detector000_snapshot-000.pt"
# Super Animal Resnet 50
#model_path="/home/max/tmp/export_torch_model/quadrupedresnet-max-2025-06-06/exported-models-pytorch/DLC_quadrupedresnet_resnet_50_iteration-0_shuffle-1/DLC_quadrupedresnet_resnet_50_iteration-0_shuffle-1_snapshot-detector000_snapshot-000.pt"

# Dog video
video_path="/home/max/dlc-live-tmp/dog_clip.avi"


#single_animal = False
# Multi Animal: Dekr
#model_path="/home/max/tmp/trimice-best/trimice-dlc-2021-06-22/exported-models-pytorch/DLC_trimice_dekr_w32_iteration-1_shuffle-25030/DLC_trimice_dekr_w32_iteration-1_shuffle-25030_snapshot-190.pt"
# Multi Animal: RTMPose
#model_path="/home/max/tmp/trimice-best/trimice-dlc-2021-06-22/exported-models-pytorch/DLC_trimice_rtmpose_m_iteration-1_shuffle-25043/DLC_trimice_rtmpose_m_iteration-1_shuffle-25043_snapshot-detector195_snapshot-390.pt"
# Multi Animal: BU Resnet
#model_path="/home/max/tmp/trimice-best/trimice-dlc-2021-06-22/exported-models-pytorch/DLC_trimice_resnet_50_iteration-1_shuffle-25053/DLC_trimice_resnet_50_iteration-1_shuffle-25053_snapshot-140.pt"
# Trimice video
#video_path="/home/max/tmp/trimice-best/trimice-dlc-2021-06-22/videos/trimouse_200frames.mp4"


model_type="pytorch"
device="cpu"

import dlclive.benchmark_pytorch

poses, times = dlclive.benchmark_pytorch.benchmark(
    model_path=model_path,
    video_path=video_path,
    model_type=model_type,
    device=device,
    single_animal=single_animal,
    save_video=False,
    draw_keypoint_names=False,
    save_poses=True,
    display=False,
)

print(np.array(times)[1:].mean())

print(f"Mean inference time: {np.array(times)[1:].mean()}[s]")
