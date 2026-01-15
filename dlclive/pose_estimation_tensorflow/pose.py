"""
DeepLabCut Toolbox (deeplabcut.org)
© A. & M. Mathis Labs

Licensed under GNU Lesser General Public License v3.0
"""


import numpy as np


def extract_cnn_output(outputs, cfg):
    """
    Extract location refinement and score map from DeepLabCut network

    Parameters
    -----------
    outputs : list
        List of outputs from DeepLabCut network.
        Requires 2 entries:
            index 0 is output from Sigmoid
            index 1 is output from pose/locref_pred/block4/BiasAdd

    cfg : dict
        Dictionary read from the pose_cfg.yaml file for the network.

    Returns
    --------
    scmap : ?
        score map

    locref : ?
        location refinement
    """

    scmap = outputs[0]
    scmap = np.squeeze(scmap)
    locref = None
    if cfg["location_refinement"]:
        locref = np.squeeze(outputs[1])
        shape = locref.shape
        locref = np.reshape(locref, (shape[0], shape[1], -1, 2))
        locref *= cfg["locref_stdev"]
    if len(scmap.shape) == 2:  # for single body part!
        scmap = np.expand_dims(scmap, axis=2)
    return scmap, locref


def argmax_pose_predict(scmap, offmat, stride):
    """
    Combines score map and offsets to the final pose

    Parameters
    -----------
    scmap : ?
        score map

    offmat : ?
        offsets

    stride : ?
        ?

    Returns
    --------
    pose :class:`numpy.ndarray`
        pose as a numpy array
    """

    num_joints = scmap.shape[2]
    pose = []
    for joint_idx in range(num_joints):
        maxloc = np.unravel_index(
            np.argmax(scmap[:, :, joint_idx]), scmap[:, :, joint_idx].shape
        )
        offset = np.array(offmat[maxloc][joint_idx])[::-1]
        pos_f8 = np.array(maxloc).astype("float") * stride + 0.5 * stride + offset
        pose.append(np.hstack((pos_f8[::-1], [scmap[maxloc][joint_idx]])))
    return np.array(pose)


def get_top_values(scmap, n_top=5):
    batchsize, ny, nx, num_joints = scmap.shape
    scmap_flat = scmap.reshape(batchsize, nx * ny, num_joints)
    if n_top == 1:
        scmap_top = np.argmax(scmap_flat, axis=1)[None]
    else:
        scmap_top = np.argpartition(scmap_flat, -n_top, axis=1)[:, -n_top:]
        for ix in range(batchsize):
            vals = scmap_flat[ix, scmap_top[ix], np.arange(num_joints)]
            arg = np.argsort(-vals, axis=0)
            scmap_top[ix] = scmap_top[ix, arg, np.arange(num_joints)]
        scmap_top = scmap_top.swapaxes(0, 1)

    Y, X = np.unravel_index(scmap_top, (ny, nx))
    return Y, X


def multi_pose_predict(scmap, locref, stride, num_outputs):
    Y, X = get_top_values(scmap[None], num_outputs)
    Y, X = Y[:, 0], X[:, 0]
    num_joints = scmap.shape[2]
    DZ = np.zeros((num_outputs, num_joints, 3))
    for m in range(num_outputs):
        for k in range(num_joints):
            x = X[m, k]
            y = Y[m, k]
            DZ[m, k, :2] = locref[y, x, k, :]
            DZ[m, k, 2] = scmap[y, x, k]

    X = X.astype("float32") * stride + 0.5 * stride + DZ[:, :, 0]
    Y = Y.astype("float32") * stride + 0.5 * stride + DZ[:, :, 1]
    P = DZ[:, :, 2]

    pose = np.empty((num_joints, num_outputs * 3), dtype="float32")
    pose[:, 0::3] = X.T
    pose[:, 1::3] = Y.T
    pose[:, 2::3] = P.T

    return pose
