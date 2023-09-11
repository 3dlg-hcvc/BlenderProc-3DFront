#  Copyright (c) 1.2022. Yinyu Nie
#  License: MIT

import vtk
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import seaborn as sns
import json
from PIL import Image, ImageDraw, ImageFont
from typing import List, Union
from visualization.utils.tools import binary_mask_to_polygon
import cv2

from visualization.vis_base import VIS_BASE
from visualization.front3d.tools.threed_front_scene import rotation_matrix

golden = (1 + 5 ** 0.5) / 2


def read_3dfront_obj2vtk(instance):
    '''Read and transform mesh from 3d front to vtk'''
    '''Read mesh to vtk'''
    vtk_object = vtk.vtkOBJReader()
    vtk_object.SetFileName(instance.raw_model_path)
    vtk_object.Update()

    '''Transform mesh'''
    # get points from object
    polydata = vtk_object.GetOutput()
    # read points using vtk_to_numpy
    obj_points = vtk_to_numpy(polydata.GetPoints().GetData()).astype(float)
    obj_points_transformed = instance._transform(obj_points)
    points_array = numpy_to_vtk(obj_points_transformed[..., :3], deep=True)
    polydata.GetPoints().SetData(points_array)
    vtk_object.Update()

    return vtk_object


def read_3dfront_extra(instance):
    '''Read and transform mesh from 3d front to vtk'''
    '''Transform vertices'''
    obj_points_transformed = instance._transform(instance.xyz)
    return obj_points_transformed, instance.faces


def get_point_cloud(depth_maps, cam_K, cam_RTs, rgb_imgs=None):
    '''
    get point cloud from depth maps
    :param depth_maps: depth map list
    :param cam_K: camera intrinsics
    :param cam_RTs: corresponding camera rotations and translations
    :param rgb_imgs: corresponding rgb images
    :return: aligned point clouds in the canonical system with color intensities.
    '''
    point_list_canonical = []
    color_intensities = []
    cam_RTs = np.copy(cam_RTs)
    if not isinstance(rgb_imgs, np.ndarray) and not isinstance(rgb_imgs, List):
        rgb_imgs = 32 * np.ones([depth_maps.shape[0], depth_maps.shape[1], depth_maps.shape[2], 3], dtype=np.uint8)

    for depth_map, rgb_img, cam_RT in zip(depth_maps, rgb_imgs, cam_RTs):
        u, v = np.meshgrid(range(depth_map.shape[1]), range(depth_map.shape[0]))
        u = u.reshape([1, -1])[0]
        v = v.reshape([1, -1])[0]

        z = depth_map[v, u]

        color_indices = rgb_img[v, u]

        # calculate coordinates
        x = (u - cam_K[0][2]) * z / cam_K[0][0]
        y = (v - cam_K[1][2]) * z / cam_K[1][1]

        point_cam = np.vstack([x, y, z]).T

        # opengl camera to opencv camera
        R = cam_RT[:3, :3]
        T = cam_RT[:3, 3]
        R[:, 1] *= -1
        R[:, 2] *= -1

        points_world = point_cam.dot(R.T) + T

        point_list_canonical.append(points_world)
        color_intensities.append(color_indices)

    return {'points': point_list_canonical, 'colors': color_intensities}


def image_grid(imgs: Union[List[np.ndarray], np.ndarray]):
    h, w = imgs[0].shape[:2]

    cols = np.floor(np.sqrt(h * golden * len(imgs) / w)).astype(np.uint16)
    rows = np.ceil(len(imgs) / cols).astype(np.uint16)

    grid = Image.new('RGB', size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(Image.fromarray(img), box=(i % cols * w, i // cols * h))
    return grid


class PROCESS_3DFRONT_2D(object):
    '''This class is to visualize the renderings of 3DFRONT scenes.'''

    def __init__(self, color_maps, inst_info, cls_maps, **kwargs):
        self.color_maps = np.array(color_maps, dtype=color_maps[0].dtype)
        self.inst_info = inst_info
        self.cls_maps = np.array(cls_maps, dtype=cls_maps[0].dtype)
        self.projected_inst_boxes = kwargs.get('projected_inst_boxes', None)
        if 'class_names' in kwargs:
            self.class_names = kwargs['class_names']
        self.cls_palette = (np.array(sns.color_palette('hls', len(self.class_names))) * 255).astype(np.uint8)

    def draw_box2d_from_3d(self):
        masked_images = self.color_maps.copy()
        font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeSans.ttf", 25, encoding="unic")
        inst_maps = []
        width = 5
        for im_id in range(len(masked_images)):
            insts_per_img = self.inst_info[im_id]
            projected_insts_per_img = self.projected_inst_boxes[im_id]
            source_img = Image.fromarray(masked_images[im_id]).convert("RGB")
            img_draw = ImageDraw.Draw(source_img)
            # Number of instances
            if not len(insts_per_img):
                print("\n*** No instances to display *** \n")
                continue
            for inst_info, proj_corners in zip(insts_per_img, projected_insts_per_img):
                if proj_corners is None: continue
                color = tuple(self.cls_palette[inst_info['category_id']])
                proj_corners = [tuple(corner) for corner in proj_corners]
                img_draw.line([proj_corners[0], proj_corners[1], proj_corners[3], proj_corners[2], proj_corners[0]],
                              fill=color, width=width)
                img_draw.line([proj_corners[4], proj_corners[5], proj_corners[7], proj_corners[6], proj_corners[4]],
                              fill=color, width=width)
                img_draw.line([proj_corners[0], proj_corners[4]],
                              fill=color, width=width)
                img_draw.line([proj_corners[1], proj_corners[5]],
                              fill=color, width=width)
                img_draw.line([proj_corners[2], proj_corners[6]],
                              fill=color, width=width)
                img_draw.line([proj_corners[3], proj_corners[7]],
                              fill=color, width=width)
            inst_maps.append(np.array(source_img))
        image_grid(inst_maps).show()

    def draw_colors(self):
        image_grid(self.color_maps).show()

    def draw_inst_maps(self, type=(), output_dir='output'):
        masked_image = self.color_maps[0].astype(np.uint8).copy()
        inst_map = np.zeros((masked_image.shape[0], masked_image.shape[1], 3), dtype=np.uint8)

        labeled_img = np.zeros(masked_image.shape[:2], dtype=np.int32)  # Image to store category IDs

        insts_per_img = self.inst_info[0]

        objects_info = []  # List to store objects info for the image

        # Number of instances
        if not len(insts_per_img):
            print("\n*** No instances to display *** \n")
            return

        for inst in insts_per_img:
            color = tuple(self.cls_palette[inst['category_id']])

            mask = np.zeros(masked_image.shape[:2], dtype=bool)
            x_min, y_min, width, height = inst['bbox2d']
            x_max = x_min + width - 1
            y_max = y_min + height - 1
            mask[y_min: y_max + 1, x_min: x_max + 1] = inst['mask']

            inst_map[mask] = color
            labeled_img[mask] = inst['category_id']

            if 'mask' in type:
                inst_mask = binary_mask_to_polygon(mask, tolerance=2)

                # Collect all polygons of the instance in a single list
                polygons = []
                for verts in inst_mask:
                    polygons.append(verts)

                obj_info = {
                    "label": self.class_names[inst['category_id']],
                    "polygons": polygons  # Using "polygons" here to indicate it can be a list of polygons
                }
                objects_info.append(obj_info)

        # Save the results to disk
        inst_map_img = Image.fromarray(inst_map)
        inst_map_img.save(f'{output_dir}/inst_map.png')

        lbl_img = Image.fromarray(labeled_img.astype(np.uint8))
        lbl_img.save(f'{output_dir}/label_map.png')

        with open(f'{output_dir}/mask.json', 'w') as file:
            json.dump(objects_info, file, indent=4)
