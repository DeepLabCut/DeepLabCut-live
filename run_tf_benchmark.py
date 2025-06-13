single_animal = False

model_path = "/home/max/dlc-live-tmp/DLC_Dog_resnet_50_iteration-0_shuffle-0/"
video_path="/home/max/dlc-live-tmp/dog_clip.avi"


import dlclive.benchmark_tf

inf_times, im_size, TFGPUinference, meta = dlclive.benchmark_tf.benchmark(
    model_path=model_path,
    video_path=video_path,
    save_video=False,
    save_poses=True,
    display=False,
)

print(f"Mean inference time: {inf_times.mean()}[s]")
