import blenderproc as bproc
import sys
import argparse
import os
import numpy as np
import random
from pathlib import Path
import json
import signal
from contextlib import contextmanager
import blenderproc.python.renderer.RendererUtility as RendererUtility
from time import time

import sys

sys.path.append('./')
from visualization.front3d import Threed_Front_Config
from visualization.front3d.tools.threed_front import ThreedFront

# import pydevd_pycharm
# pydevd_pycharm.settrace('localhost', port=12345, stdoutToServer=True, stderrToServer=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("front_folder", help="Path to the 3D front file")
    parser.add_argument("future_folder", help="Path to the 3D Future Model folder.")
    parser.add_argument("front_3D_texture_folder", help="Path to the 3D FRONT texture folder.")
    parser.add_argument("front_json",
                        help="Path to a 3D FRONT scene json file, e.g.6a0e73bc-d0c4-4a38-bfb6-e083ce05ebe9.json.")
    parser.add_argument('cc_material_folder', nargs='?', default="resources/cctextures",
                        help="Path to CCTextures folder, see the /scripts for the download script.")
    parser.add_argument("output_folder", nargs='?', default="examples/datasets/front_3d_with_improved_mat/renderings",
                        help="Path to where the data should be saved")
    parser.add_argument("--n_views_per_scene", type=int, default=100,
                        help="The number of views to render in each scene.")
    parser.add_argument("--append_to_existing_output", type=bool, default=True,
                        help="If append new renderings to the existing ones.")
    parser.add_argument("--fov", type=int, default=90, help="Field of view of camera.")
    parser.add_argument("--res_x", type=int, default=480, help="Image width.")
    parser.add_argument("--res_y", type=int, default=360, help="Image height.")
    return parser.parse_args()


class TimeoutException(Exception): pass


@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


def get_folders(args):
    front_folder = Path(args.front_folder)
    future_folder = Path(args.future_folder)
    front_3D_texture_folder = Path(args.front_3D_texture_folder)
    cc_material_folder = Path(args.cc_material_folder)
    output_folder = Path(args.output_folder)
    if not output_folder.exists():
        output_folder.mkdir()
    return front_folder, future_folder, front_3D_texture_folder, cc_material_folder, output_folder


def get_bbox_of_all_objects(loaded_objects):
    min_x, min_y, min_z = float('inf'), float('inf'), float('inf')
    max_x, max_y, max_z = float('-inf'), float('-inf'), float('-inf')
    for obj in loaded_objects:
        if isinstance(obj, bproc.types.MeshObject):
            bbox_corners = obj.get_bound_box()
            for corner in bbox_corners:
                min_x = min(min_x, corner[0])
                min_y = min(min_y, corner[1])
                min_z = min(min_z, corner[2])
                max_x = max(max_x, corner[0])
                max_y = max(max_y, corner[1])
                max_z = max(max_z, corner[2])
    return (min_x, min_y, min_z), (max_x, max_y, max_z)


def is_point_inside_box(point, box_corners, margin=0.01):
    # Get the min and max coordinates of the box in x, y, and z directions
    min_corner = [min([corner[i] for corner in box_corners]) - margin for i in range(3)]
    max_corner = [max([corner[i] for corner in box_corners]) + margin for i in range(3)]

    # Check if point lies inside the box
    for i in range(3):
        if point[i] < min_corner[i] or point[i] > max_corner[i]:
            return False
    return True


def get_box_corners(center, vectors):
    '''
    Convert box center and vectors to the corner-form.
    Note x0<x1, y0<y1, z0<z1, then the 8 corners are concatenated by:
    [[x0, y0, z0], [x0, y0, z1], [x0, y1, z0], [x0, y1, z1],
     [x1, y0, z0], [x1, y0, z1], [x1, y1, z0], [x1, y1, z1]]
    :return: corner points and faces related to the box
    '''
    corner_pnts = [None] * 8
    corner_pnts[0] = tuple(center - vectors[0] - vectors[1] - vectors[2])
    corner_pnts[1] = tuple(center - vectors[0] - vectors[1] + vectors[2])
    corner_pnts[2] = tuple(center - vectors[0] + vectors[1] - vectors[2])
    corner_pnts[3] = tuple(center - vectors[0] + vectors[1] + vectors[2])

    corner_pnts[4] = tuple(center + vectors[0] - vectors[1] - vectors[2])
    corner_pnts[5] = tuple(center + vectors[0] - vectors[1] + vectors[2])
    corner_pnts[6] = tuple(center + vectors[0] + vectors[1] - vectors[2])
    corner_pnts[7] = tuple(center + vectors[0] + vectors[1] + vectors[2])

    return corner_pnts


def adjust_coordinates(corners):
    '''
    Convert from layout_bbox's coordinate system (where y is up)
    to scene's coordinate system (where z is up).
    '''
    adjusted_corners = []
    for corner in corners:
        x, y, z = corner
        adjusted_corners.append([x, z, y])  # Swap y and z values
    return np.array(adjusted_corners)


def get_corners(layout_box):
    floor_center, x_vec, y_vec, z_vec = layout_box[:3], layout_box[3:6], layout_box[6:9], layout_box[9:12]
    centroid = floor_center + y_vec / 2
    vectors = np.array([x_vec, y_vec / 2, z_vec])
    corners = get_box_corners(centroid, vectors)

    corners = adjust_coordinates(corners)

    return corners


def get_centroid(bbox_corners):
    min_x, min_y, min_z = float('inf'), float('inf'), float('inf')
    max_x, max_y, max_z = float('-inf'), float('-inf'), float('-inf')
    for corner in bbox_corners:
        min_x = min(min_x, corner[0])
        min_y = min(min_y, corner[1])
        min_z = min(min_z, corner[2])
        max_x = max(max_x, corner[0])
        max_y = max(max_y, corner[1])
        max_z = max(max_z, corner[2])

    centroid_point = [
        (min_x + max_x) / 2,  # Center in X
        (min_y + max_y) / 2,  # Center in Y
        (min_z + max_z) / 2  # Center in Z
    ]  # Center in Z

    return centroid_point


def check_name(name, category_name):
    return True if category_name in name.lower() else False


if __name__ == '__main__':
    '''Parse folders / file paths'''
    args = parse_args()
    front_folder, future_folder, front_3D_texture_folder, cc_material_folder, output_folder = get_folders(args)
    front_json = front_folder.joinpath(args.front_json)
    # n_cameras = args.n_views_per_scene
    n_cameras = 1

    dataset_config = Threed_Front_Config()
    dataset_config.init_generic_categories_by_room_type('all')
    json_file = [str(front_json).split("/")[-1]]
    d = ThreedFront.from_dataset_directory(
        str(dataset_config.threed_front_dir),
        str(dataset_config.model_info_path),
        str(dataset_config.threed_future_dir),
        str(dataset_config.dump_dir_to_scenes),
        path_to_room_masks_dir=None,
        path_to_bounds=None,
        json_files=json_file,
        filter_fn=lambda s: s)

    layout_boxes = {}
    for rm in d.rooms:
        layout_boxes[rm.room_id] = rm.layout_box

    failed_scene_name_file = output_folder.parent.joinpath('failed_scene_names.txt')

    cam_intrinsic_path = output_folder.joinpath('cam_K.npy')

    if not front_folder.exists() or not future_folder.exists() \
            or not front_3D_texture_folder.exists() or not cc_material_folder.exists():
        raise Exception("One of these folders does not exist!")

    scene_name = front_json.name[:-len(front_json.suffix)]
    print('Processing scene name: %s.' % (scene_name))

    '''Pass those failure cases'''
    if failed_scene_name_file.is_file():
        with open(failed_scene_name_file, 'r') as file:
            failure_scenes = file.read().splitlines()
        if scene_name in failure_scenes:
            print('File in failure log: %s. Continue.' % (scene_name))
            sys.exit(0)

    '''Pass already generated scenes.'''
    scene_output_folder = output_folder.joinpath(scene_name)
    existing_n_renderings = 0

    if scene_output_folder.is_dir():
        existing_n_renderings = len(list(scene_output_folder.iterdir()))
        if existing_n_renderings >= n_cameras:
            print('Scene %s is already generated.' % (scene_output_folder.name))
            sys.exit(0)

    if args.append_to_existing_output:
        n_cameras = n_cameras - existing_n_renderings

    try:
        with time_limit(10000):  # per scene generation would not exceeds X seconds.
            start_time = time()

            bproc.init()
            RendererUtility.set_max_amount_of_samples(32)

            mapping_file = bproc.utility.resolve_resource(os.path.join("front_3D", "blender_label_mapping.csv"))
            mapping = bproc.utility.LabelIdMapping.from_csv(mapping_file)

            # set the light bounces
            bproc.renderer.set_light_bounces(diffuse_bounces=200, glossy_bounces=200, max_bounces=200,
                                             transmission_bounces=200, transparent_max_bounces=200)
            # set intrinsic parameters
            bproc.camera.set_intrinsics_from_blender_params(lens=args.fov / 180 * np.pi, image_width=args.res_x,
                                                            image_height=args.res_y,
                                                            lens_unit="FOV")

            cam_K = bproc.camera.get_intrinsics_as_K_matrix()

            # write camera intrinsics
            if not cam_intrinsic_path.exists():
                np.save(str(cam_intrinsic_path), cam_K)

            # read 3d future model info
            with open(future_folder.joinpath('model_info_revised.json'), 'r') as f:
                model_info_data = json.load(f)
            model_id_to_label = {m["model_id"]: m["category"].lower().replace(" / ", "/") if m["category"] else 'others'
                                 for
                                 m in
                                 model_info_data}

            # load the front 3D objects
            loaded_objects = bproc.loader.load_front3d(
                json_path=str(front_json),
                future_model_path=str(future_folder),
                front_3D_texture_path=str(front_3D_texture_folder),
                label_mapping=mapping,
                model_id_to_label=model_id_to_label)

            bedroom_ids = set(obj.get_cp("room_id") for obj in loaded_objects if
                              obj.has_cp("room_id") and "Bedroom" in obj.get_cp("room_id"))

            # bproc.renderer.enable_normals_output()

            for current_bedroom_id in bedroom_ids:
                room_output_folder = f"{scene_output_folder}_{current_bedroom_id}"

                bproc.init()
                RendererUtility.set_max_amount_of_samples(32)

                mapping_file = bproc.utility.resolve_resource(os.path.join("front_3D", "blender_label_mapping.csv"))
                mapping = bproc.utility.LabelIdMapping.from_csv(mapping_file)

                # set the light bounces
                bproc.renderer.set_light_bounces(diffuse_bounces=200, glossy_bounces=200, max_bounces=200,
                                                 transmission_bounces=200, transparent_max_bounces=200)
                # set intrinsic parameters
                bproc.camera.set_intrinsics_from_blender_params(lens=args.fov / 180 * np.pi, image_width=args.res_x,
                                                                image_height=args.res_y,
                                                                lens_unit="FOV")

                cam_K = bproc.camera.get_intrinsics_as_K_matrix()

                # load the front 3D objects again
                loaded_objects = bproc.loader.load_front3d(
                    json_path=str(front_json),
                    future_model_path=str(future_folder),
                    front_3D_texture_path=str(front_3D_texture_folder),
                    label_mapping=mapping,
                    model_id_to_label=model_id_to_label)

                objects_ceiling = []
                not_target_objects = []
                traget_objects = []
                # only select objects from the current bedroom:
                for tmp_object in loaded_objects:
                    bbox = tmp_object.get_bound_box()
                    centroid = get_centroid(bbox)
                    layout_bbox = get_corners(layout_boxes[current_bedroom_id])
                    origin = tmp_object.get_origin()

                    if "Ceiling" in tmp_object.get_name():
                        objects_ceiling.append(tmp_object)
                        continue
                    if tmp_object.has_cp("room_id"):
                        # is_point_inside_box(origin, layout_bbox)
                        if tmp_object.get_cp("room_id") == current_bedroom_id:
                            traget_objects.append(tmp_object)

                            # tmp_obj = bproc.object.create_primitive("SPHERE")
                            # tmp_obj.set_scale([0.1, 0.1, 0.1])
                            # tmp_obj.set_location(centroid)
                        else:
                            not_target_objects.append(tmp_object)
                    else:
                        if ("Wall" in tmp_object.get_name() or "Floor" in tmp_object.get_name()):
                            if is_point_inside_box(centroid, layout_bbox, margin=0.07) and "Outer" not in tmp_object.get_name() and "Top" not in tmp_object.get_name():
                                traget_objects.append(tmp_object)
                            else:
                                not_target_objects.append(tmp_object)
                            # traget_objects.append(tmp_object)
                            # not_target_objects.append(tmp_object)
                            #
                            # # to adebug draw the center of all wall and floor objects
                            # tmp_obj = bproc.object.create_primitive("SPHERE")
                            # tmp_obj.set_scale([0.1, 0.1, 0.1])
                            # tmp_obj.set_location(centroid)
                        else:
                            not_target_objects.append(tmp_object)

                # for corner in layout_bbox:
                #     tmp_obj = bproc.object.create_primitive("SPHERE")
                #     tmp_obj.set_scale([0.1, 0.1, 0.1])
                #     tmp_obj.set_location(corner)

                # -------------------------------------------------------------------------
                #          Sample materials
                # -------------------------------------------------------------------------
                cc_materials = bproc.loader.load_ccmaterials(args.cc_material_folder,
                                                             ["Bricks", "Wood", "Carpet", "Tile", "Marble"])

                floors = bproc.filter.by_attr(loaded_objects, "name", "Floor.*", regex=True)
                for floor in floors:
                    # For each material of the object
                    for i in range(len(floor.get_materials())):
                        floor.set_material(i, random.choice(cc_materials))

                baseboards_and_doors = bproc.filter.by_attr(loaded_objects, "name", "Baseboard.*|Door.*", regex=True)
                wood_floor_materials = bproc.filter.by_cp(cc_materials, "asset_name", "WoodFloor.*", regex=True)
                for obj in baseboards_and_doors:
                    # For each material of the object
                    for i in range(len(obj.get_materials())):
                        # Replace the material with a random one
                        obj.set_material(i, random.choice(wood_floor_materials))

                walls = bproc.filter.by_attr(loaded_objects, "name", "Wall.*", regex=True)
                marble_materials = bproc.filter.by_cp(cc_materials, "asset_name", "Marble.*", regex=True)
                for wall in walls:
                    # For each material of the object
                    for i in range(len(wall.get_materials())):
                        wall.set_material(i, random.choice(marble_materials))

                # -------------------------------------------------------------------------
                #          Sample camera extrinsics
                # -------------------------------------------------------------------------
                # Init sampler for sampling locations inside the loaded front3D house
                point_sampler = bproc.sampler.Front3DPointInRoomSampler(loaded_objects)

                # Init bvh tree containing all mesh objects
                bvh_tree = bproc.object.create_bvh_tree_multi_objects(
                    [o for o in loaded_objects if isinstance(o, bproc.types.MeshObject)])

                # filter some objects from the loaded objects, which are later used in calculating an interesting score
                interest_score_setting = {'ceiling': 0, 'column': 0, 'customizedpersonalizedmodel': 0, 'beam': 0,
                                          'wallinner': 0,
                                          'slabside': 0, 'customizedfixedfurniture': 0, 'cabinet/lightband': 0,
                                          'window': 0,
                                          'hole': 0, 'customizedplatform': 0, 'baseboard': 0,
                                          'customizedbackgroundmodel': 0,
                                          'front': 0, 'walltop': 0, 'wallouter': 0, 'cornice': 0, 'sewerpipe': 0,
                                          'smartcustomizedceiling': 0, 'customizedfeaturewall': 0,
                                          'customizedfurniture': 0,
                                          'slabtop': 0, 'baywindow': 0, 'door': 0, 'customized_wainscot': 0,
                                          'slabbottom': 0,
                                          'back': 0, 'flue': 0, 'extrusioncustomizedceilingmodel': 0,
                                          'extrusioncustomizedbackgroundwall': 0, 'floor': 0, 'lightband': 0,
                                          'customizedceiling': 0, 'void': 0, 'pocket': 0, 'wallbottom': 0, 'chair': 10,
                                          'sofa': 10,
                                          'table': 10, 'bed': 10}
                special_objects = []
                special_object_scores = {}
                for category_name, category_score in interest_score_setting.items():
                    special_objects_per_category = [obj.get_cp("category_id") for obj in loaded_objects if
                                                    check_name(obj.get_name(), category_name)]
                    special_objects.extend(special_objects_per_category)
                    unique_cat_ids = set(special_objects_per_category)
                    for cat_id in unique_cat_ids:
                        special_object_scores[cat_id] = category_score

                # sample camera poses
                proximity_checks = {}
                cam_Ts = []
                floor_areas = np.array(point_sampler.get_floor_areas())
                cam_nums = np.ceil(floor_areas / floor_areas.sum() * n_cameras).astype(np.int16)
                n_tries = cam_nums * 3

                for floor_id, cam_num_per_scene in enumerate(cam_nums):
                    cam2world_matrices = []
                    coverage_scores = []
                    tries = 0
                    while tries < n_tries[floor_id]:
                        # sample cam loc inside house
                        height = np.random.uniform(1.4, 1.8)
                        # location = point_sampler.sample_by_floor_id(height, floor_id=floor_id)
                        # # Sample rotation (fix around X and Y axis)
                        # rotation = np.random.uniform([1.2217, 0, 0], [1.338, 0, np.pi * 2])  # pitch, roll, yaw

                        bounding_box = get_bbox_of_all_objects(traget_objects)
                        highest_point = bounding_box[1][2]  # Z-coordinate of the top of the bounding box
                        bird_eye_height = highest_point + 5  # for example,5 units/meters above the highest point

                        location = [bounding_box[0][0] + (bounding_box[1][0] - bounding_box[0][0]) / 2,  # Center in X
                                    bounding_box[0][1] + (bounding_box[1][1] - bounding_box[0][1]) / 2,  # Center in Y
                                    bird_eye_height]  # Above the room
                        rotation = [0, 0, np.pi / 2]  # pitch, roll, yaw for bird's eye view

                        cam2world_matrix = bproc.math.build_transformation_mat(location, rotation)

                        # Check that obstacles are at least 1 meter away from the camera and have an average distance between 2.5 and 3.5
                        # meters and make sure that no background is visible, finally make sure the view is interesting enough
                        obstacle_check = bproc.camera.perform_obstacle_in_view_check(cam2world_matrix, proximity_checks,
                                                                                     bvh_tree)
                        coverage_score = bproc.camera.scene_coverage_score(cam2world_matrix, special_objects,
                                                                           special_objects_weight=special_object_scores)
                        # for sanity check
                        if obstacle_check and coverage_score >= 0.0:
                            cam2world_matrices.append(cam2world_matrix)
                            coverage_scores.append(coverage_score)
                            tries += 1
                        cam2world_matrices.append(cam2world_matrix)
                    cam_ids = np.argsort(coverage_scores)[-cam_num_per_scene:]
                    for cam_id, cam2world_matrix in enumerate(cam2world_matrices):
                        if cam_id in cam_ids:
                            bproc.camera.add_camera_pose(cam2world_matrix)
                            cam_Ts.append(cam2world_matrix)

                # render the whole pipeline

                bproc.object.delete_multiple(objects_ceiling)
                bproc.object.delete_multiple(not_target_objects)

                # bproc.renderer.enable_depth_output(activate_antialiasing=False)
                # tmp_output_dir = f"output/temp/{current_bedroom_id}"
                data = bproc.renderer.render()
                # default_values = {"location": [0, 0, 0], "cp_inst_mark": '', "cp_uid": '', "cp_jid": '',
                #                   "cp_room_id": ""}
                # data.update(bproc.renderer.render_segmap(
                #     map_by=["instance", "class", "cp_uid", "cp_jid", "cp_inst_mark", "cp_room_id", "location", "height",
                #             "orientation"],
                #     default_values=default_values))
                #
                # # write camera extrinsics
                # data['cam_Ts'] = cam_Ts
                # # write the data to a .hdf5 container
                bproc.writer.write_hdf5(str(room_output_folder), data,
                                        append_to_existing_output=args.append_to_existing_output)
                print('Time elapsed: %f.' % (time() - start_time))

                break

    except TimeoutException as e:
        print('Time is out: %s.' % scene_name)
        with open(failed_scene_name_file, 'a') as file:
            file.write(scene_name + "\n")
        sys.exit(0)
    except Exception as e:
        print('Failed scene name: %s.' % scene_name)
        with open(failed_scene_name_file, 'a') as file:
            file.write(scene_name + "\n")
        sys.exit(0)
