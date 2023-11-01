#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from scene.cameras import Camera
import numpy as np
import torch
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp

WARNED = False

def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.width, cam_info.height
    print("orig_w:",orig_w,"orig_h:",orig_h)
    # TODO: check this problem later

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    gt_image = None
    loaded_mask = None
    
    if cam_info.image is not None:
        resized_image_rgb = PILtoTorch(cam_info.image, resolution)
        gt_image = resized_image_rgb[:3, ...]

        if resized_image_rgb.shape[1] == 4:
            loaded_mask = resized_image_rgb[3:4, ...]
        
    opengl_proj = None
    if hasattr(cam_info, "opengl_proj"):
        opengl_proj = cam_info.opengl_proj
        
    w2c = None
    if hasattr(cam_info, "w2c"):
        w2c = cam_info.w2c
        
    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device, 
                  opengl_proj=opengl_proj, w2c=w2c, width=resolution[0], height=resolution[1])

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    
    # here there might be a problem with the w2c camera matrix, it should be c2w camera as I can TELL....
    
    if hasattr(camera, "w2c"):
        W2C = camera.w2c
        W2C = np.linalg.inv(W2C)
    else:
        Rt = np.zeros((4, 4))
        Rt[:3, :3] = camera.R.transpose()
        Rt[:3, 3] = camera.T
        Rt[3, 3] = 1.0
        W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry


def interpolate(cam1, cam2, ratio):
    # cam1, cam2: 4*4 transformation matrix
    # ratio: a float number between 0 and 1
    # return: a 4*4 transformation matrix
    
    Rot_mat = [cam[:3,:3] for cam in [cam1, cam2]]
    rots = Rot.from_matrix(np.stack([Rot_mat[0], Rot_mat[1]]))
    key_times = [0, 1]
    slerp = Slerp(key_times, rots)
    rot = slerp(ratio)
    
    # get the translation vectors
    Tran_mat = [cam[:3,3] for cam in [cam1, cam2]]
    
    interpolated_translations = (1 - ratio)[:, np.newaxis] * Tran_mat[0] + ratio[:, np.newaxis] * Tran_mat[1]
        
    # Now we should combine the rotation and translation to get the transformation matrix
    interpolated_w2cs = []

    for i in range(ratio.shape[0]):
        interpolated_w2c = np.eye(4)
        interpolated_w2c[:3,:3] = rot[i].as_matrix()
        interpolated_w2c[:3,3] = interpolated_translations[i]
        interpolated_w2cs.append(interpolated_w2c)
        
    return interpolated_w2cs
            
def interpolate_all(w2cs, ratio):
    # w2cs: a list of 4*4 transformation matrix
    # ratio: a float number between 0 and 1
    # return: a list of 4*4 transformation matrix
    interp_w2cs_all = []
    # we need to combine the lists not append the lists
    for i in range(len(w2cs)-1):
        interp_w2cs = interpolate(w2cs[i], w2cs[i+1], ratio)
        interp_w2cs_all += interp_w2cs
        
    # print(len(interp_w2cs_all))
    
    return interp_w2cs_all