import argparse
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import imageio
import os
import numpy as np
import skimage
import cv2
from PIL import Image
import glob
import time
from torchvision.transforms import GaussianBlur
from omegaconf import OmegaConf
import yaml

import matplotlib.pyplot as plt
plt.rcParams['axes.unicode_minus']=False
from scipy.ndimage import uniform_filter
from scipy.ndimage import median_filter

from sv_comp.dataset import MultiWarpDataset
from stitch_dynamic import MeshFlowStabilizer, seamcut
import stitch_utils
from multiband import multi_band_blending

mesh_row_count = 10
mesh_col_count = 16
multicore = 4
W = 512
H = 512
O = 100

def motion_field_filter(left_velocity, right_velocity):
    # 中值滤波器去噪
    left_velocity = median_filter(left_velocity, size=3)
    right_velocity = median_filter(right_velocity, size=3)

    # 运动场的平滑过程 调试中
    O = int((-np.ceil(np.median(left_velocity[:, 15, 0])) + np.floor(np.median(right_velocity[:, 1, 0]))) // 2)
    O_l = -int(np.ceil(np.median(left_velocity[:, 15, 0])))
    O_r = int(np.floor(np.median(right_velocity[:, 1, 0])))
    print(O, O_l, O_r)

    # 边缘的运动场平滑 根据平移量置固定值
    vertex_motion_x_l = left_velocity[:, :15, 0]
    vertex_motion_y_l = left_velocity[:, :15, 1]

    vertex_motion_x_r = right_velocity[:, -15:, 0]
    vertex_motion_y_r = right_velocity[:, -15:, 1]

    vertex_motion_y_l[:, :10] = 0
    vertex_motion_x_l[:, :10] = -O_l
    # # vertex_motion_x_l[:, 1] = -O 
    # vertex_motion_x_l[:, 2] = -O - 15

    vertex_motion_y_r[:, -10:] = 0
    vertex_motion_x_r[:, -10:] = O_r
    # vertex_motion_x_r[:, -2] = O 
    # vertex_motion_x_r[:, -3] = O + 15

    # 均值滤波器
    vertex_motion_x_l_filter = uniform_filter(vertex_motion_x_l, size=3)
    vertex_motion_y_l_filter = uniform_filter(vertex_motion_y_l, size=3)
    # vertex_motion_y_l_filter = vertex_motion_y_l

    vertex_motion_x_r_filter = uniform_filter(vertex_motion_x_r, size=3)
    vertex_motion_y_r_filter = uniform_filter(vertex_motion_y_r, size=3)
    # vertex_motion_y_r_filter = vertex_motion_y_r

    # vertex_motion_x_l_filter[:, :5] = -O
    # vertex_motion_x_r_filter[:, -5:] = O

    # vertex_motion_l_no = np.dstack((vertex_motion_x_l_filter, vertex_motion_y_l_filter))
    # left_velocity[:, :12, :] = vertex_motion_l_no

    # vertex_motion_r_no = np.dstack((vertex_motion_x_r_filter, vertex_motion_y_r_filter))
    # right_velocity[:, -12:, :] = vertex_motion_r_no

    vertex_motion_l_no = np.dstack((vertex_motion_x_l_filter, vertex_motion_y_l_filter))
    # vertex_motion_l_no = np.dstack((vertex_motion_x_l_filter, vertex_motion_y_l))
    left_velocity[:, :15, :] = vertex_motion_l_no

    vertex_motion_r_no = np.dstack((vertex_motion_x_r_filter, vertex_motion_y_r_filter))
    # vertex_motion_r_no = np.dstack((vertex_motion_x_r_filter, vertex_motion_y_r))
    right_velocity[:, -15:, :] = vertex_motion_r_no

    # vertex_motion_y_l[:, :10] = 0
    # vertex_motion_x_l[:, :10] = -O_l

    # vertex_motion_y_r[:, -10:] = 0
    # vertex_motion_x_r[:, -10:] = O_r

    return left_velocity, right_velocity, O_l, O_r, O

if __name__ == '__main__':
    # 加载数据集
    sv_comp_cfg = OmegaConf.load('sv_comp/sv_comp.yaml')
    with open('sv_comp/intrinsics.yaml', 'r', encoding='utf-8') as file:
        intrinsics = yaml.safe_load(file)
    dataset = MultiWarpDataset(config=sv_comp_cfg, intrinsics=intrinsics, is_train=False)
    num_frames = 1  # 一张图像复制若干次作为一个视频

    for idx in range(len(dataset)):
        sample = dataset[idx]
        input_imgs, input_masks = sample[0], sample[1]
        left_frames = [input_imgs[0]] * num_frames
        right_frames = [input_imgs[1]] * num_frames
        os.makedirs(f'data/{idx}', exist_ok=True)
        cv2.imwrite(f'data/{idx}/left.jpg', left_frames[0])
        cv2.imwrite(f'data/{idx}/right.jpg', right_frames[0])

        # Stablizer
        stabilizer = MeshFlowStabilizer(visualize=True)
        adaptive_weights_definition = MeshFlowStabilizer.ADAPTIVE_WEIGHTS_DEFINITION_FLIPPED

        if not (adaptive_weights_definition == MeshFlowStabilizer.ADAPTIVE_WEIGHTS_DEFINITION_ORIGINAL or
            adaptive_weights_definition == MeshFlowStabilizer.ADAPTIVE_WEIGHTS_DEFINITION_FLIPPED or
            adaptive_weights_definition == MeshFlowStabilizer.ADAPTIVE_WEIGHTS_DEFINITION_CONSTANT_HIGH or
            adaptive_weights_definition == MeshFlowStabilizer.ADAPTIVE_WEIGHTS_DEFINITION_CONSTANT_LOW):
            raise ValueError(
            'Invalid value for `adaptive_weights_definition`. Expecting value of '
            '`MeshFlowStabilizer.ADAPTIVE_WEIGHTS_DEFINITION_ORIGINAL`, '
            '`MeshFlowStabilizer.ADAPTIVE_WEIGHTS_DEFINITION_FLIPPED`, '
            '`MeshFlowStabilizer.ADAPTIVE_WEIGHTS_DEFINITION_CONSTANT_HIGH`, or'
            '`MeshFlowStabilizer.ADAPTIVE_WEIGHTS_DEFINITION_CONSTANT_LOW`.'
        )

        #前视图后视图,拼接运动场生成
        vertex_left, vertex_right = stabilizer._get_stitch_vertex_displacements_and_homographies(num_frames, left_frames, right_frames)
        #使用雅可比方法，计算网格顶点稳定后的运动场
        vertex_stabilized_stitched_by_frame_index_1 = stabilizer._get_stitch_vertex_displacements(
            num_frames, vertex_left
        )

        vertex_stabilized_stitched_by_frame_index_2 = stabilizer._get_stitch_vertex_displacements(
            num_frames, vertex_right
        )

        stitcher = stitch_utils.stitch_utils(mesh_row_count=mesh_row_count, mesh_col_count=mesh_col_count, 
                                            feature_ellipse_row_count=8, feature_ellipse_col_count=10)

        # stitched_frames = []
        # stitched_frames_multiband = []
        frame_index = 0
        left_frame_ = left_frames[frame_index]
        right_frame_ = right_frames[frame_index]
        
        left_velocity_ = vertex_stabilized_stitched_by_frame_index_1[frame_index]
        right_velocity_ = vertex_stabilized_stitched_by_frame_index_2[frame_index]
        left_velocity_1_filter, right_velocity_1_filter, O_l_1, O_r_1, O_1 = motion_field_filter(left_velocity_, right_velocity_)
        O_1 = O_1 + 5
        # 网格变形
        img_l_1 = stitcher.get_warped_frames_for_stitch(0, left_frame_, left_velocity_1_filter, O_l_1)
        img_r_1 = stitcher.get_warped_frames_for_stitch(1, right_frame_, right_velocity_1_filter, O_r_1)

        # 缝合线选取
        print(f"H = {H}, W = {W}, O_1 = {O_1}")
        print(f"W + 2*O_1 = {W + 2 * O_1}")
        l = np.zeros((H, W + 2 * O_1, 3), np.uint8)
        r = np.zeros((H, W + 2 * O_1, 3), np.uint8)
        l[:, :W, :] = img_l_1
        r[:, 2 * O_1:, :] = img_r_1
        stitched_seam_1 = seamcut(l, r)

        # 多频段融合
        flag_half = False
        mask = None
        need_mask =True     
        leveln = 5

        overlap_w = W-2*O_1
        stitched_band_1 = multi_band_blending(img_l_1, img_r_1, mask, overlap_w, leveln, flag_half, need_mask)

        cv2.imwrite(f'data/{idx}/{frame_index:03d}_seamcut.jpg', stitched_seam_1)
        cv2.imwrite(f'data/{idx}/{frame_index:03d}_multiband.jpg', stitched_band_1)
