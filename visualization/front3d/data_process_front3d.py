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
    parser.add_argument("--output_dir", type=str, default='output/processed_front3d_data',
                        help="The output directory")
    parser.add_argument("--debug", default=True, action="store",
                        help="The output directory")
    return parser.parse_args()


def scale_to_0_255(arr):
    # Compute the current minimum and maximum values of the array
    min_val = np.min(arr)
    max_val = np.max(arr)

    # Scale the array to [0, 255]
    scaled_arr = 255 * (arr - min_val) / (max_val - min_val)

    # Convert the scaled array to unsigned 8-bit integer type
    return scaled_arr.astype(np.uint8)


def process_scene(dataset_config, output_dir, scene_render_dir):
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
        for render_path in scene_render_dir.iterdir():
            with h5py.File(render_path) as f:
                colors = np.array(f["colors"])[:, ::-1]
                cam_T = np.array(f["cam_Ts"])
                class_segmap = np.array(f["class_segmaps"])[:, ::-1]
                instance_segmap = np.array(f["instance_segmaps"])[:, ::-1]
                instance_attribute_mapping = json.loads(f["instance_attribute_maps"][()])

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

                for part in parts:
                    # Get 2D masks
                    part_mask = (instance_segmap == part['idx'])

                    # Use this mask to assign the height and orientation values to the global height_map and orientation_map
                    height_map_all[part_mask] = part['height']
                    orientation_map_all[part_mask] = part['orientation']

                # get 3D bbox
                inst_rm_uid = "_".join([scene_json, inst_dict['room_id']])
                inst_3d_info = parse_inst_from_3dfront(inst_dict, d.rooms, inst_rm_uid)
                inst_dict = {**inst_dict, **inst_3d_info, **{'room_uid': inst_rm_uid}}

                inst_info.append(inst_dict)

            # Concatenate the height and orientation maps to form the object info map
            object_info_map = np.concatenate([height_map_all[..., np.newaxis], orientation_map_all], axis=-1)

            # process cam_T from blender to ours
            cam_T = dataset_config.blender2opengl_cam(cam_T)
            room_imgs.append(colors)
            cam_Ts.append(cam_T)
            class_maps.append(class_segmap)
            instance_attrs.append(inst_info)

            '''Project objects 3D boxes to image planes'''
            projected_box2d_list = project_insts_to_2d(inst_info, cam_K, cam_T)
            projected_inst_boxes.append(projected_box2d_list)

        # # get room layout information
        # layout_boxes = []
        # for rm in d.rooms:
        #     layout_boxes.append(rm.layout_box)

        # save height and orientation map
        depth_ori_output_path = os.path.join(output_dir, "depth_ori_map.npy")
        cv2.imwrite(
            '/home/sunxh/Xiaohao/slice_layout_gen/BlenderProc-3DFront/output/processed_front3d_data/6a0e73bc-d0c4-4a38-bfb6-e083ce05ebe9_SecondBedroom-1415/height.png',
            scale_to_0_255(height_map_all))
        cv2.imwrite(
            '/home/sunxh/Xiaohao/slice_layout_gen/BlenderProc-3DFront/output/processed_front3d_data/6a0e73bc-d0c4-4a38-bfb6-e083ce05ebe9_SecondBedroom-1415/orientation.png',
            scale_to_0_255(orientation_map_all))
        np.save(depth_ori_output_path, object_info_map)

        process_2D = PROCESS_3DFRONT_2D(color_maps=room_imgs, inst_info=instance_attrs,
                                        cls_maps=class_maps, class_names=dataset_config.label_names,
                                        projected_inst_boxes=projected_inst_boxes)

        process_2D.draw_inst_maps(type=('mask'), output_dir=output_dir)
        process_2D.draw_colors()

    except Exception as e:
        print(f"Error in scene {scene_render_dir}: {e}")
        traceback.print_exc()

    return


if __name__ == '__main__':
    args = parse_args()
    # Create a list of directories.
    base_rendering_path = "/home/sunxh/Xiaohao/slice_layout_gen/BlenderProc-3DFront/examples/datasets/front_3d_with_improved_mat/renderings"
    scene_dirs = [d for d in Path(base_rendering_path).iterdir() if d.is_dir()]

    # Define the output directory
    output_directory = args.output_dir
    dataset_config = Threed_Front_Config()

    if args.debug:
        process_scene(dataset_config, output_directory, scene_dirs[0])
    else:
        partial_process = partial(process_scene, dataset_config, output_directory)
        process_map(partial_process, scene_dirs, chunksize=1)
