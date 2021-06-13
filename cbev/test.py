"""
# -*- coding: utf-8 -*-
-----------------------------------------------------------------------------------
# Author: Nguyen Mau Dung
# DoC: 2020.08.17
# email: nguyenmaudung93.kstn@gmail.com
-----------------------------------------------------------------------------------
# Description: Testing script
"""

import argparse
import sys
import os
import time
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

from easydict import EasyDict as edict
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import torch
import numpy as np
import mayavi.mlab

src_dir = os.path.dirname(os.path.realpath(__file__))
while not src_dir.endswith("cbev"):
    src_dir = os.path.dirname(src_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

from data_process.kitti_dataloader import create_test_dataloader
from models.model_utils import create_model
from utils.misc import make_folder, time_synchronized
from utils.evaluation_utils import decode, post_processing, draw_predictions, convert_det_to_real_values
from utils.torch_utils import _sigmoid
import config.kitti_config as cnf
from data_process.transformation import lidar_to_camera_box
from utils.visualization_utils import merge_rgb_to_bev, show_rgb_image_with_boxes
from data_process.kitti_data_utils import Calibration, get_filtered_lidar
from data_process.kitti_bev_utils import makeBEVView


def parse_test_configs():
    parser = argparse.ArgumentParser(description='Testing config for the Implementation')
    parser.add_argument('--saved_fn', type=str, default='dla34', metavar='FN',
                        help='The name using for saving logs, models,...')
    parser.add_argument('-a', '--arch', type=str, default='dla34', metavar='ARCH',
                        help='The name of the model architecture')
    parser.add_argument('--pretrained_path', type=str,
                        default='../checkpoints/dla34/dla34_hdiff_epoch_70.pth', metavar='PATH',
                        help='the path of the pretrained checkpoint')
    parser.add_argument('--K', type=int, default=50,
                        help='the number of top K')
    parser.add_argument('--no_cuda', action='store_true',
                        help='If true, cuda is not used.')
    parser.add_argument('--gpu_idx', default=0, type=int,
                        help='GPU index to use.')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Take a subset of the dataset to run and debug')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='Number of threads for loading data')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='mini-batch size (default: 4)')
    parser.add_argument('--peak_thresh', type=float, default=0.2)
    parser.add_argument('--save_test_output', action='store_true',
                        help='If true, the output image of the testing phase will be saved')
    parser.add_argument('--output_format', type=str, default='image', metavar='PATH',
                        help='the type of the test output (support image or video)')
    parser.add_argument('--output_video_fn', type=str, default='out_fpn_resnet_18', metavar='PATH',
                        help='the video filename if the output format is video')
    parser.add_argument('--output-width', type=int, default=608,
                        help='the width of showing output, the height maybe vary')

    configs = edict(vars(parser.parse_args()))
    configs.pin_memory = True
    configs.distributed = False  # For testing on 1 GPU only

    configs.input_size = (608, 608)
    configs.hm_size = (152, 152)
    configs.down_ratio = 4
    configs.max_objects = 50

    configs.imagenet_pretrained = False
    configs.head_conv = 256
    configs.num_classes = 3
    configs.num_center_offset = 2
    configs.num_z = 1
    configs.num_dim = 3
    configs.num_direction = 2  # sin, cos

    configs.heads = {
        'hm_cen': configs.num_classes,
        'cen_offset': configs.num_center_offset,
        'direction': configs.num_direction,
        'z_coor': configs.num_z,
        'dim': configs.num_dim
    }
    configs.num_input_features = 4

    ####################################################################
    ##############Dataset, Checkpoints, and results dir configs#########
    ####################################################################
    configs.root_dir = '../'
    configs.dataset_dir = '/media/wx/File/DATA/kittidatabase'

    if configs.save_test_output:
        configs.results_dir = os.path.join(configs.root_dir, 'results', configs.saved_fn)
        make_folder(configs.results_dir)

    return configs


def inverse_rigid_trans(Tr):
    ''' Inverse a rigid body transform matrix (3x4 as [R|t])
        [R'|-R't; 0|1]
    '''
    inv_Tr = np.zeros_like(Tr) # 3x4
    inv_Tr[0:3,0:3] = np.transpose(Tr[0:3,0:3])
    inv_Tr[0:3,3] = np.dot(-np.transpose(Tr[0:3,0:3]), Tr[0:3,3])
    return inv_Tr


V2C= np.array([7.533745000000e-03, -9.999714000000e-01, -6.166020000000e-04,
                           -4.069766000000e-03, 1.480249000000e-02, 7.280733000000e-04,
                           -9.998902000000e-01, -7.631618000000e-02, 9.998621000000e-01,
                           7.523790000000e-03, 1.480755000000e-02, -2.717806000000e-01])
V2C = np.reshape(V2C, [3, 4])

C2V = inverse_rigid_trans(V2C)

R0 = np.array([9.999239000000e-01, 9.837760000000e-03, -7.445048000000e-03,  -9.869795000000e-03,
                    9.999421000000e-01, -4.278459000000e-03, 7.402527000000e-03, 4.351614000000e-03,
                    9.999631000000e-01])
R0 = np.reshape(R0, [3, 3])


def cart2hom(pts_3d):
    ''' Input: nx3 points in Cartesian
        Oupput: nx4 points in Homogeneous by pending 1
    '''
    n = pts_3d.shape[0]
    pts_3d_hom = np.hstack((pts_3d, np.ones((n, 1))))
    return pts_3d_hom

def project_ref_to_velo(pts_3d_ref):
    pts_3d_ref = cart2hom(pts_3d_ref)  # nx4
    return np.dot(pts_3d_ref, np.transpose(C2V))


def project_rect_to_ref(pts_3d_rect):
    ''' Input and Output are nx3 points '''
    return np.transpose(np.dot(np.linalg.inv(R0), np.transpose(pts_3d_rect)))


def project_rect_to_velo(pts_3d_rect):
    ''' Input: nx3 points in rect camera coord.
        Output: nx3 points in velodyne coord.
    '''
    pts_3d_ref = project_rect_to_ref(pts_3d_rect)
    return project_ref_to_velo(pts_3d_ref)

def rotz(t):
    ''' Rotation about the y-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c,  -s,  0],
                     [s,  c,  0],
                     [0, 0,  1]])


def roty(t):
    ''' Rotation about the y-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c,  0,  s],
                     [0,  1,  0],
                     [-s, 0,  c]])

def draw_gt_boxes3d(gt_boxes3d, score,fig,  color=(1,1,1), line_width=1, draw_text=True, text_scale=(1,1,1), color_list=None, ):
    ''' Draw 3D bounding boxes
    Args:
        gt_boxes3d: numpy array (n,8,3) for XYZs of the box corners
        fig: mayavi figure handler
        color: RGB value tuple in range (0,1), box line color
        line_width: box line width
        draw_text: boolean, if true, write box indices beside boxes
        text_scale: three number tuple
        color_list: a list of RGB tuple, if not None, overwrite color.
    Returns:
        fig: updated fig
    '''
    num = len(gt_boxes3d)
    for n in range(num):
        b = gt_boxes3d[n]
        if color_list is not None:
            color = color_list[n]
        #if draw_text: mayavi.mlab.text3d(b[4,0], b[4,1], b[4,2], 'car'+"{:.2f}".format(float(score)), scale=text_scale, color=(1,1,1), figure=fig)
        for k in range(0,4):
            #http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
            i,j=k,(k+1)%4
            mayavi.mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=color, tube_radius=None, line_width=line_width, figure=fig)

            i,j=k+4,(k+1)%4 + 4
            mayavi.mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=color, tube_radius=None, line_width=line_width, figure=fig)

            i,j=k,k+4
            mayavi.mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=color, tube_radius=None, line_width=line_width, figure=fig)
    #mlab.show(1)
    #mlab.view(azimuth=180, elevation=70, focalpoint=[ 12.0909996 , -1.04700089, -2.03249991], distance=62.0, figure=fig)
    return fig

def show3dlidar(pointpaht, detections):
    pointcloud = np.fromfile(pointpaht, dtype=np.float32).reshape(-1, 4)
    x = pointcloud[:, 0]  # x position of point
    xmin = np.amin(x, axis=0)
    xmax = np.amax(x, axis=0 )
    y = pointcloud[:, 1]  # y position of point
    ymin = np.amin(y, axis=0)
    ymax = np.amax(y, axis=0)
    z = pointcloud[:, 2]  # z position of point
    zmin = np.amin(z, axis=0)
    zmax = np.amax(z, axis=0)
    d = np.sqrt(x ** 2 + y ** 2)  # Map Distance from sensor
    vals = 'height'
    if vals == "height":
        col = z
    else:
        col = d
    fig = mayavi.mlab.figure(bgcolor=(0, 0, 0), size=(640, 500))
    mayavi.mlab.points3d(x, y, z,
                         col,  # Values used for Color
                         mode="point",
                         # 灰度图的伪彩映射
                         colormap='Blues',  # 'bone', 'copper', 'gnuplot'
                         # color=(0, 1, 0),   # Used a fixed (r,g,b) instead
                         figure=fig,
                         )
    # 绘制原点
    '''mayavi.mlab.points3d(0, 0, 0, color=(1, 1, 1), mode="sphere",scale_factor=0.2)

    print(detections.shape)

    detections[:, 1:] = lidar_to_camera_box(detections[:, 1:], calib.V2C, calib.R0, calib.P2)

    for i in range(detections.shape[0]):

        h = float(detections[i][4])
        w = float(detections[i][5])
        l = float(detections[i][6])

        x = float(detections[i][1])
        y = float(detections[i][2])
        z = float(detections[i][3])
        x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2] ;
        y_corners = [0, 0, 0, 0, -h, -h, -h, -h] ;
        z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2];
        #print(x_corners)
        #print(detections[i])
        R = roty(float(detections[i][7]))
        corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
        # print corners_3d.shape
        #corners_3d = np.zeros((3,8))
        corners_3d[0, :] = corners_3d[0, :] + x;
        corners_3d[1, :] = corners_3d[1, :] + y;
        corners_3d[2, :] = corners_3d[2, :] + z;
        corners_3d = np.transpose(corners_3d)
        box3d_pts_3d_velo = project_rect_to_velo(corners_3d)
        #x1, y1, z1 = box3d_pts_3d_velo[0, :]
        #x2, y2, z2 = box3d_pts_3d_velo[1, :]
        if detections[i][0] == 1.0:
            draw_gt_boxes3d([box3d_pts_3d_velo],1,color=(1,0,0), fig=fig)
        else:
            draw_gt_boxes3d([box3d_pts_3d_velo], 1, color=(0, 1, 0), fig=fig)
    mayavi.mlab.show()'''

if __name__ == '__main__':
    configs = parse_test_configs()

    model = create_model(configs)
    print('\n\n' + '-*=' * 30 + '\n\n')
    assert os.path.isfile(configs.pretrained_path), "No file at {}".format(configs.pretrained_path)
    model.load_state_dict(torch.load(configs.pretrained_path, map_location='cpu'))
    print('Loaded weights from {}\n'.format(configs.pretrained_path))

    configs.device = torch.device('cpu' if configs.no_cuda else 'cuda:{}'.format(configs.gpu_idx))
    model = model.to(device=configs.device)

    out_cap = None

    model.eval()

    test_dataloader = create_test_dataloader(configs)
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(test_dataloader):
            metadatas, bev_maps, img_rgbs = batch_data
            input_bev_maps = bev_maps.to(configs.device, non_blocking=True).float()
            t1 = time_synchronized()
            outputs = model(input_bev_maps)
            outputs['hm_cen'] = _sigmoid(outputs['hm_cen'])
            outputs['cen_offset'] = _sigmoid(outputs['cen_offset'])
            # detections size (batch_size, K, 10)
            detections = decode(outputs['hm_cen'], outputs['cen_offset'], outputs['direction'], outputs['z_coor'],
                                outputs['dim'], K=configs.K)
            detections = detections.cpu().numpy().astype(np.float32)
            detections = post_processing(detections, configs.num_classes, configs.down_ratio, configs.peak_thresh)
            t2 = time_synchronized()

            detections = detections[0]  # only first batch
            # Draw prediction in the image
            bev_map = (bev_maps.squeeze().permute(1, 2, 0).numpy() * 255).astype(np.uint8)

            zero = np.sum(bev_map, axis=2)
            indices = np.where(zero== 0)
            bev_map[indices[0], indices[1], :] = [240, 240, 240]

            bev_map = cv2.resize(bev_map, (cnf.BEV_WIDTH, cnf.BEV_HEIGHT))
            bev_map = draw_predictions(bev_map, detections.copy(), configs.num_classes)

            # Rotate the bev_map
            bev_map = cv2.rotate(bev_map, cv2.ROTATE_180)

            img_path = metadatas['img_path'][0]

            img_rgb = img_rgbs[0].numpy()
            img_rgb = cv2.resize(img_rgb, (img_rgb.shape[1], img_rgb.shape[0]))
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            calib = Calibration(img_path.replace(".png", ".txt").replace("image_2", "calib"))
            kitti_dets = convert_det_to_real_values(detections)
            #if len(kitti_dets) > 0:
            #    show3dlidar(lidar_path, kitti_dets)

            if len(kitti_dets) > 0:
                kitti_dets[:, 1:] = lidar_to_camera_box(kitti_dets[:, 1:], calib.V2C, calib.R0, calib.P2)
                img_bgr = show_rgb_image_with_boxes(img_bgr, kitti_dets, calib)

            out_img = merge_rgb_to_bev(img_bgr, bev_map, output_width=configs.output_width)

            print('\tDone testing the {}th sample, time: {:.1f}ms, speed {:.2f}FPS'.format(batch_idx, (t2 - t1) * 1000,
                                                                                           1 / (t2 - t1)))
            if configs.save_test_output:
                if configs.output_format == 'image':
                    img_fn = os.path.basename(metadatas['img_path'][0])[:-4]
                    cv2.imwrite(os.path.join(configs.results_dir, '{}.jpg'.format(img_fn)), out_img)
                elif configs.output_format == 'video':
                    if out_cap is None:
                        out_cap_h, out_cap_w = out_img.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                        out_cap = cv2.VideoWriter(
                            os.path.join(configs.results_dir, '{}.avi'.format(configs.output_video_fn)),
                            fourcc, 30, (out_cap_w, out_cap_h))

                    out_cap.write(out_img)
                else:
                    raise TypeError

            cv2.imshow('test-img', out_img)
            print('\n[INFO] Press n to see the next sample >>> Press Esc to quit...\n')
            if cv2.waitKey(0) & 0xFF == 27:
                break
    if out_cap:
        out_cap.release()
    cv2.destroyAllWindows()
