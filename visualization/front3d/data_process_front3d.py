import sys

sys.path.append('.')
import argparse
import h5py
import numpy as np
import json
from visualization.front3d import Threed_Front_Config
from visualization.front3d.tools.threed_front import ThreedFront
from visualization.front3d.data_process_classes import PROCESS_3DFRONT_2D
from visualization.front3d.tools.utils import parse_inst_from_3dfront, project_insts_to_2d
from visualization.utils.tools import label_mapping_2D
from pathlib import Path

import os
# from concurrent.futures import ProcessPoolExecutor
# import concurrent
from tqdm import tqdm
from functools import partial
from tqdm.contrib.concurrent import process_map
import traceback
import cv2


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize a 3D-FRONT room.")
    parser.add_argument("--output_dir", type=str, default='../../datasets/output/prosessed_3dfront_data_V5',
                        help="The output directory")
    parser.add_argument("--debug", default=False, action="store",
                        help="The output directory")
    parser.add_argument("--floor", default=True, action="store",
                        help="The output directory")
    return parser.parse_args()


def scale_to_0_255(arr):
    # Compute the current minimum and maximum values of the array
    min_val = np.min(arr)
    max_val = np.max(arr)

    # Scale the array to [0, 255]
    if max_val-min_val != 0:
        scaled_arr = 255 * (arr - min_val) / (max_val - min_val)
    else:
        scaled_arr = 255 * arr

    # Convert the scaled array to unsigned 8-bit integer type
    return scaled_arr.astype(np.uint8)


def mask_to_coco_polygon(binary_mask):
    """
    Convert a binary mask to COCO polygon representation.

    Args:
    - binary_mask (ndarray): A 2D binary numpy array where mask is represented by `True` values.

    Returns:
    - coco_polygons (list): A list of lists containing the polygon points.
    """
    # Convert binary mask to uint8
    mask_uint8 = np.uint8(binary_mask)

    # Find contours in the binary mask
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    coco_polygons = []

    for contour in contours:
        point_list = contour.flatten().tolist()
        if len(point_list) > 4:  # Ensure the polygon has more than 2 points (4 values)
            coco_polygons.append(point_list)
        # # Flatten list of points
        # for shape in simplified_contour:
        #     point_list = shape.flatten().tolist()
        #     if len(point_list) > 4:  # Ensure the polygon has more than 2 points (4 values)
        #         coco_polygons.append(point_list)

    return coco_polygons


def process_scene(dataset_config, output_dir, floor_slice, scene_render_dir):
    try:
        # initialize category labels and mapping dict for specific room type.
        dataset_config.init_generic_categories_by_room_type('all')

        '''Read 3D-Front Data'''
        room_id = scene_render_dir.parts[-1]
        scene_id = scene_render_dir.parts[-1].split("_")[0]
        # print("processsing room ", room_id)
        json_path = f"{scene_id}.json"
        json_files = [json_path]
        d = ThreedFront.from_dataset_directory(
            str(dataset_config.threed_front_dir),
            str(dataset_config.model_info_path),
            str(dataset_config.threed_future_dir),
            str(dataset_config.dump_dir_to_scenes),
            path_to_room_masks_dir=None,
            path_to_bounds=None,
            json_files=json_files,
            filter_fn=lambda s: s)
        # print(d)

        # print("processing room ", room_id)

        output_dir = f"{output_dir}/{room_id}"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        '''Read rendering information'''
        cam_K = dataset_config.cam_K

        room_imgs = []
        room_heights = []
        room_orientations = []
        cam_Ts = []
        class_maps = []
        instance_attrs = []
        projected_inst_boxes = []
        plane_names = []
        for render_path in scene_render_dir.iterdir():
            if floor_slice and "Bottom" in str(render_path):
                with h5py.File(render_path) as f:
                    colors = np.array(f["colors"])[:, ::-1]
                    cam_T = np.array(f["cam_Ts"])
                    class_segmap = np.array(f["class_segmaps"])[:, ::-1]
                    instance_segmap = np.array(f["instance_segmaps"])[:, ::-1]
                    instance_attribute_mapping = json.loads(f["instance_attribute_maps"][()])
            elif floor_slice and "Bottom" not in str(render_path):
                continue
            else:
                with h5py.File(render_path) as f:
                    colors = np.array(f["colors"])[:, ::-1]
                    cam_T = np.array(f["cam_Ts"])
                    class_segmap = np.array(f["class_segmaps"])[:, ::-1]
                    instance_segmap = np.array(f["instance_segmaps"])[:, ::-1]
                    instance_attribute_mapping = json.loads(f["instance_attribute_maps"][()])



            plane_name = str(render_path).split("/")[-1].split(".")[0]

            ### get scene_name
            scene_json = render_path.parent.name

            #### class mapping
            class_segmap = label_mapping_2D(class_segmap, dataset_config.label_mapping)

            #### get instance info
            # inst_marks = set([inst['inst_mark'] for inst in instance_attribute_mapping if
            #                   inst['inst_mark'] != '' and 'layout' not in inst['inst_mark']])

            inst_marks = set([inst['inst_mark'] for inst in instance_attribute_mapping if
                              inst['inst_mark'] != ''])

            inst_info = []
            instance_annotation = []
            inst_id = 0
            # Initialize maps outside the loop
            height_map_all = np.zeros(instance_segmap.shape,
                                      dtype=np.float32)  # Assuming float32 is suitable for height
            orientation_map_all = np.zeros((*instance_segmap.shape, 3),
                                           dtype=np.float32)  # Assuming 3 values for orientation and float32 type

            for inst_mark in inst_marks:
                parts = [part for part in instance_attribute_mapping if part['inst_mark'] == inst_mark]

                # remove background objects.
                category_id = dataset_config.label_mapping[parts[0]['category_id']]
                if category_id == 0:
                    continue
                inst_anno = {"category": category_id}
                # get 2D masks
                part_indices = [part['idx'] for part in parts]
                inst_mask = np.sum([instance_segmap == idx for idx in part_indices], axis=0, dtype=bool)

                # get 2D bbox
                mask_mat = np.argwhere(inst_mask)
                y_min, x_min = mask_mat.min(axis=0)
                y_max, x_max = mask_mat.max(axis=0)
                bbox = [x_min, y_min, x_max - x_min + 1, y_max - y_min + 1]  # [x,y,width,height]
                if min(bbox[2:]) <= dataset_config.min_bbox_edge_len:
                    continue

                inst_dict = {key: parts[0][key] for key in ['inst_mark', 'uid', 'jid', 'room_id', 'location']}
                inst_dict['category_id'] = category_id
                inst_dict['mask'] = inst_mask[y_min:y_max + 1, x_min:x_max + 1]
                inst_dict['bbox2d'] = bbox

                valid_part = 0
                for idx, part in enumerate(parts):
                    # Get 2D masks
                    part_mask = (instance_segmap == part['idx'])

                    # Use this mask to assign the height and orientation values to the global height_map and orientation_map
                    # height_map_all[part_mask] = part['height']
                    # orientation_map_all[part_mask] = part['orientation']

                    current_vol = part["size"][0] * part["size"][1] * part["size"][2]
                    previous_vol = parts[valid_part]["size"][0] * parts[valid_part]["size"][1] * parts[valid_part]["size"][2]

                    if current_vol > previous_vol:
                        valid_part = idx

                # get 3D bbox
                inst_rm_uid = "_".join([scene_json, inst_dict['room_id']])
                inst_3d_info = parse_inst_from_3dfront(inst_dict, d.rooms, inst_rm_uid)
                inst_dict = {**inst_dict, **inst_3d_info, **{'room_uid': inst_rm_uid}}

                polygon_mask = mask_to_coco_polygon(inst_mask)
                inst_anno["mask"] = polygon_mask
                inst_anno["size"] = parts[valid_part]["size"]
                inst_anno["orientation"] = parts[valid_part]["orientation"]
                inst_anno["scale"] = parts[valid_part]["scale"]
                inst_anno["offset"] = parts[valid_part]["height"]
                inst_anno["inst_id"] = inst_id
                inst_anno["model_id"] = parts[valid_part]["jid"]
                inst_id += 1

                instance_annotation.append(inst_anno)

                inst_info.append(inst_dict)

            # Concatenate the height and orientation maps to form the object info map
            # object_info_map = np.concatenate([height_map_all[..., np.newaxis], orientation_map_all], axis=-1)

            # process cam_T from blender to ours
            cam_T = dataset_config.blender2opengl_cam(cam_T)
            room_imgs.append(colors)
            cam_Ts.append(cam_T)
            class_maps.append(class_segmap)
            instance_attrs.append(inst_info)
            plane_names.append(plane_name)

            '''Project objects 3D boxes to image planes'''
            projected_box2d_list = project_insts_to_2d(inst_info, cam_K, cam_T)
            projected_inst_boxes.append(projected_box2d_list)

        # # get room layout information
        # layout_boxes = []
        # for rm in d.rooms:
        #     layout_boxes.append(rm.layout_box)

            # save height and orientation map
            depth_ori_output_path = os.path.join(output_dir, plane_name + "_depth_ori_map.npy")
            vis_output_path = os.path.join(output_dir, plane_name)
            # cv2.imwrite(
            #     f'{vis_output_path}_height.png',
            #     scale_to_0_255(height_map_all))
            # cv2.imwrite(
            #     f'{vis_output_path}_orientation.png',
            #     scale_to_0_255(orientation_map_all))
            # np.save(depth_ori_output_path, object_info_map)

            with open(f'{vis_output_path}_inst_anno.json', "w") as outfile:
                json.dump(instance_annotation, outfile, indent=4)

        process_2D = PROCESS_3DFRONT_2D(color_maps=room_imgs, inst_info=instance_attrs,
                                        cls_maps=class_maps, class_names=dataset_config.label_names,
                                        projected_inst_boxes=projected_inst_boxes, plane_names=plane_names)

        process_2D.draw_inst_maps(type=('mask'), output_dir=output_dir)
        process_2D.draw_colors(output_dir=output_dir)

    except Exception as e:
        print(f"Error in scene {scene_render_dir}: {e}")
        traceback.print_exc()

    return


if __name__ == '__main__':
    args = parse_args()
    # Create a list of directories.
    base_rendering_path = "/localhome/xsa55/Xiaohao/SemDiffLayout/datasets/front_3d_with_improved_mat/renderings_V5"
    scene_dirs = [d for d in Path(base_rendering_path).iterdir() if d.is_dir()]

    # Define the output directory
    output_directory = args.output_dir
    dataset_config = Threed_Front_Config()
    floor_slice = args.floor

    if args.debug:
        process_scene(dataset_config, output_directory, floor_slice, scene_dirs[0])
    else:
        partial_process = partial(process_scene, dataset_config, output_directory, floor_slice)
        process_map(partial_process, scene_dirs, chunksize=1)
