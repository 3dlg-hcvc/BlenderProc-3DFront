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
from examples.datasets.front_3d_with_improved_mat.view_tool import ViewGenerator


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
    parser.add_argument("--n_views_per_scene", type=int, default=1,
                        help="The number of views to render in each scene.")
    parser.add_argument("--bound_slice", default=True, type=bool,
                        help="If we want to get boundary slices for all rooms")
    parser.add_argument("--append_to_existing_output", type=bool, default=True,
                        help="If append new renderings to the existing ones.")
    parser.add_argument("--fov", type=int, default=120, help="Field of view of camera.")
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


def is_majority_of_vertices_inside_bbox(mesh, bbox_corners, threshold=0.9):
    """
    Check if a majority of vertices of the given mesh are inside the bounding box defined by bbox_corners.

    Args:
    - mesh (bpy.types.Object): The Blender mesh object.
    - bbox_corners (list of list of float): List of 8 corner coordinates of the bounding box.
    - threshold (float): Fraction of vertices that need to be inside the bbox for the mesh to be considered inside.

    Returns:
    - bool: True if the majority of mesh vertices are inside the bounding box, False otherwise.
    """

    # Convert bbox_corners to numpy array for easy calculations
    bbox_corners = np.array(bbox_corners)

    # Compute the min and max corners of the bounding box
    bbox_min = np.min(bbox_corners, axis=0)
    bbox_max = np.max(bbox_corners, axis=0)

    inside_count = 0

    # Check every vertex of the mesh
    for vertex in mesh.vertices:
        # Convert the vertex coordinate to world space
        coord = vertex.co
        # Check if the vertex is inside the bounding box
        if all(bbox_min[i] - 1e-2 <= coord[i] <= bbox_max[i] + 1e-2 for i in range(3)):
            inside_count += 1

    return inside_count / len(mesh.vertices) >= threshold


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


def get_plane_name(plane_normal):
    normals = [
        np.array([0, 0, -1]),  # Bottom: z0
        np.array([0, 0, 1]),  # Top: z1
        np.array([0, -1, 0]),  # Front: y0
        np.array([0, 1, 0]),  # Back: y1
        np.array([-1, 0, 0]),  # Left: x0
        np.array([1, 0, 0])  # Right: x1
    ]

    # Creating a dictionary to map normals to plane names
    normal_to_name = {
        tuple(norm): name for norm, name in zip(normals, ["Bottom", "Top", "Front", "Back", "Left", "Right"])
    }

    plane_name = normal_to_name.get(tuple(plane_normal), "Unknown")

    return plane_name


def get_planes_from_bbox(bbox):
    # Assuming Z is up, using the right-hand rule for normals
    normals = [
        np.array([0, 0, -1]),  # Bottom: z0
        np.array([0, 0, 1]),  # Top: z1
        np.array([0, -1, 0]),  # Front: y0
        np.array([0, 1, 0]),  # Back: y1
        np.array([-1, 0, 0]),  # Left: x0
        np.array([1, 0, 0])  # Right: x1
    ]

    # One point from each plane
    points = [
        bbox[0],  # Bottom
        bbox[3],  # Top
        bbox[0],  # Front
        bbox[7],  # Back
        bbox[0],  # Left
        bbox[4]  # Right
    ]

    return list(zip(points, normals))


def distance_to_plane(point, plane_point, plane_normal):
    return plane_normal.dot(point - plane_point)


def plane_normal_to_rotation(normal):
    # Ensure the normal is a unit vector for precision
    normal = normal / np.linalg.norm(normal)

    x, y, z = normal

    # Using ZYX Euler angles convention

    # Plane normal (0, 0, 1)
    if np.allclose(normal, [0, 0, -1]):
        return 0, 0, np.pi

    # Plane normal (0, 0, -1)
    elif np.allclose(normal, [0, 0, 1]):
        return 0, np.pi, np.pi

    # Plane normal (0, 1, 0)
    elif np.allclose(normal, [0, 1, 0]):
        return np.pi / 2, 0, 0

    # Plane normal (0, -1, 0)
    elif np.allclose(normal, [0, -1, 0]):
        return np.pi / 2, 0, np.pi

    # Plane normal (1, 0, 0)
    elif np.allclose(normal, [1, 0, 0]):
        return np.pi / 2, 0, 3 * np.pi / 2

    # Plane normal (-1, 0, 0)
    elif np.allclose(normal, [-1, 0, 0]):
        return np.pi / 2, 0, np.pi / 2

    else:
        raise ValueError("Unsupported plane normal")


def is_close_to_plane(obj, plane):
    plane_point, plane_normal = plane[0], plane[1]
    bbox = obj.get_bound_box()
    min_distance = float('inf')
    idx = 0
    for tmp_idx, corner in enumerate(bbox):
        distance = np.abs(distance_to_plane(corner, plane_point, plane_normal))
        if distance < min_distance:
            min_distance = distance
            idx = tmp_idx
    # tmp_obj = bproc.object.create_primitive("SPHERE")
    # tmp_obj.set_scale([0.1, 0.1, 0.1])
    # tmp_obj.set_location(bbox[idx])
    if np.abs(min_distance) < 0.5:
        print(obj.get_name(), "distance_to_plane: ", min_distance)
        return True
    else:
        print(obj.get_name(), "distance_to_plane: ", min_distance)
        return False


def min_max_distance_to_plane(plane, bbox):
    plane_point, plane_normal = plane[0], plane[1]
    max_distance = float('-inf')
    min_distance = float('inf')
    idx = 0
    for tmp_idx, corner in enumerate(bbox):
        distance = np.abs(distance_to_plane(corner, plane_point, plane_normal))
        if distance > max_distance:
            max_distance = distance
        if distance < min_distance:
            min_distance = distance
    return max_distance, min_distance


def oriented_bbox_dimensions(bbox_corners, plane_normal):
    # # Normalize the plane normal
    # plane_normal = np.array(plane_normal, dtype=np.float64) / np.linalg.norm(plane_normal)
    #
    # # Find the bbox corner with the largest projection onto the plane's normal
    # projections = [np.dot(p, plane_normal) for p in bbox_corners]
    # idx_max_proj = np.argmax(projections)
    # corner1 = np.array(bbox_corners[idx_max_proj])
    #
    # # Find the other corners of the oriented bbox face
    # diffs = [np.linalg.norm(corner1 - np.array(c)) for c in bbox_corners]
    #
    # # The largest two differences will give the diagonally opposite corner on the bbox
    # idx_diag_opposite = np.argmax(diffs)
    # corner2 = np.array(bbox_corners[idx_diag_opposite])
    #
    # # Remove the selected corners to find the remaining corners
    # remaining_corners = [c for i, c in enumerate(bbox_corners) if i not in [idx_max_proj, idx_diag_opposite]]
    # diffs_remaining = [np.linalg.norm(corner1 - np.array(c)) for c in remaining_corners]
    # idx_third_corner = np.argmax(diffs_remaining)
    # corner3 = np.array(remaining_corners[idx_third_corner])
    #
    # # Compute the dimensions
    # length = np.linalg.norm(corner1 - corner2)
    # width = np.linalg.norm(corner1 - corner3)
    # height = np.linalg.norm(corner2 - corner3)

    length = max(bbox_corners[:, 0]) - min(bbox_corners[:, 0])
    width = max(bbox_corners[:, 1]) - min(bbox_corners[:, 1])
    height = max(bbox_corners[:, 2]) - min(bbox_corners[:, 2])

    return length, width, height


def check_name(name, category_name):
    return True if category_name in name.lower() else False


def room_process_by_plane(plane, target_objects, loaded_objects, only_floor, args):
    bbox_height = []
    bbox_height_not_target_furniture = []
    plane_name = get_plane_name(plane[1])
    # only select objects from the current bedroom:

    distance_info = {}
    size_info = {}
    for idx, tmp_object in enumerate(target_objects):
        bbox = tmp_object.get_bound_box()
        if tmp_object.has_cp("room_id"):
            size = oriented_bbox_dimensions(bbox, plane[1])
            size_info[tmp_object.get_name()] = size
            if is_close_to_plane(tmp_object, plane) or only_floor:
                # bbox_height.append(max(bbox[:, 2]))
                max_distance, min_distance = min_max_distance_to_plane(plane, bbox)
                bbox_height.append(max_distance)
                distance_info[tmp_object.get_name()] = tmp_object.get_location()[-1]
            else:
                max_distance, min_distance = min_max_distance_to_plane(plane, bbox)
                # bbox_height_not_target_furniture.append(min(bbox[:, 2]))
                bbox_height_not_target_furniture.append(min_distance)
                distance_info[tmp_object.get_name()] = min_distance
        else:
            distance_info[tmp_object.get_name()] = 0
            size_info[tmp_object.get_name()] = (0.0, 0.0, 0.0)

    if len(bbox_height) == 0:
        bbox_height.append(0.5)
    # -------------------------------------------------------------------------
    #          Sample camera extrinsics
    # -------------------------------------------------------------------------
    # Init sampler for sampling locations inside the loaded front3D house
    point_sampler = bproc.sampler.Front3DPointInRoomSampler(target_objects)

    # Init bvh tree containing all mesh objects
    bvh_tree = bproc.object.create_bvh_tree_multi_objects(
        [o for o in target_objects if isinstance(o, bproc.types.MeshObject)])

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
        special_objects_per_category = [obj.get_cp("category_id") for obj in target_objects if
                                        check_name(obj.get_name(), category_name)]
        special_objects.extend(special_objects_per_category)
        unique_cat_ids = set(special_objects_per_category)
        for cat_id in unique_cat_ids:
            special_object_scores[cat_id] = category_score

    # sample camera poses
    proximity_checks = {}
    cam_Ts = []
    floor_areas = np.array(point_sampler.get_floor_areas())
    # cam_nums = np.ceil(floor_areas / floor_areas.sum() * n_cameras).astype(np.int16)
    cam_nums = [1]
    n_tries = cam_nums

    bbox = {}
    for floor_id, cam_num_per_scene in enumerate(cam_nums):
        cam2world_matrices = []
        coverage_scores = []
        tries = 0
        while tries < n_tries[floor_id]:
            # sample cam loc inside house
            # height = np.random.uniform(1.4, 1.8)
            # location = point_sampler.sample_by_floor_id(height, floor_id=floor_id)
            # # Sample rotation (fix around X and Y axis)
            # rotation = np.random.uniform([1.2217, 0, 0], [1.338, 0, np.pi * 2])  # pitch, roll, yaw

            bounding_box = get_bbox_of_all_objects(target_objects)
            # highest_point = bounding_box[1][2]  # Z-coordinate of the top of the bounding box
            # bird_eye_height = highest_point + 2  # for example,5 units/meters above the highest point
            if only_floor:
                plane_height = max(bbox_height)
            else:
                if min(bbox_height_not_target_furniture) > max(bbox_height):
                    plane_height = max(bbox_height)
                else:
                    plane_height = min(bbox_height_not_target_furniture)

            print("plane height ====================", plane_height)

            center_x = bounding_box[0][0] + (bounding_box[1][0] - bounding_box[0][0]) / 2
            center_y = bounding_box[0][1] + (bounding_box[1][1] - bounding_box[0][1]) / 2
            center_z = bounding_box[0][2] + (bounding_box[1][2] - bounding_box[0][2]) / 2
            location = [center_x,  # Center in X
                        center_y,  # Center in Y
                        center_z]  # Above the room
            bbox["centroid"] = location
            bbox["dimensions"] = [bounding_box[1][0] - bounding_box[0][0], bounding_box[1][1] - bounding_box[0][1],
                                  bounding_box[1][2] - bounding_box[0][2]]
            plane_height = plane[0][np.nonzero(plane[1])[0][0]] - plane_height * plane[1][np.nonzero(plane[1])[0][0]]
            location[np.nonzero(plane[1])[0][0]] = plane_height

            normal = plane[1]
            rotation = plane_normal_to_rotation(normal)

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

    view_generator = ViewGenerator()
    view_opts = view_generator.get_fixed_resolution_view_for_bbox(bbox, pixel_width=0.1, theta=0, phi=0,
                                                                  use_square_image=False)

    # render the whole pipeline
    # bproc.camera.set_intrinsics_from_blender_params(lens=args.fov / 180 * np.pi, image_width=args.res_x,
    #                                                 image_height=args.res_y, clip_start=0.01, clip_end=5,
    #                                                 lens_unit="FOV", ortho_scale=8)
    dim_x = bbox["dimensions"][0]
    dim_y = bbox["dimensions"][1]
    pixel_width = 0.01
    res_x = dim_x / pixel_width
    res_y = dim_y / pixel_width
    orthographic_scale = max(res_x, res_y) * pixel_width
    bproc.camera.set_intrinsics_from_blender_params(lens=view_opts["fov"], image_width=res_x,
                                                    image_height=res_y, clip_start=0, clip_end=10,
                                                    lens_unit="FOV", ortho_scale=orthographic_scale)

    # bproc.renderer.enable_depth_output(activate_antialiasing=False)
    # tmp_output_dir = f"output/temp/{current_bedroom_id}"
    data = bproc.renderer.render()
    default_values = {"location": [0, 0, 0], "cp_inst_mark": '', "cp_uid": '', "cp_jid": '',
                      "cp_room_id": ""}
    data.update(bproc.renderer.render_segmap(
        map_by=["instance", "class", "cp_uid", "cp_jid", "cp_inst_mark", "cp_room_id", "cf_basename", "location",
                "height",
                "orientation", "size", "scale"],
        default_values=default_values))

    # # write camera extrinsics
    if "cam_Ts" in data:
        data['cam_Ts'].append(cam_Ts)
    else:
        data['cam_Ts'] = cam_Ts
    sub_data = {key: [value[-1]] for key, value in data.items()}

    # # write the data to a .hdf5 container
    bproc.writer.write_hdf5(str(room_output_folder), sub_data, plane_name=plane_name,
                            append_to_existing_output=args.append_to_existing_output)


if __name__ == '__main__':
    '''Parse folders / file paths'''
    args = parse_args()
    front_folder, future_folder, front_3D_texture_folder, cc_material_folder, output_folder = get_folders(args)
    front_json = front_folder.joinpath(args.front_json)
    # n_cameras = args.n_views_per_scene
    n_cameras = 1
    room_type = "livingroom"

    split_path = "/localhome/xsa55/Xiaohao/SemDiffLayout/scripts/visualization/config/livingroom_threed_front_splits.csv"
    valid_room_ids = []
    with open(split_path, "r") as f:
        for line in f:
            valid_room_ids.append(line.strip().split(",")[0])

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
                try:
                    layout_boxes[rm.room_id] = rm.layout_box
                except:
                    continue

            # read 3d future model info
            with open(future_folder.joinpath('model_info_revised.json'), 'r') as f:
                model_info_data = json.load(f)
            model_id_to_label = {m["model_id"]: m["category"].lower().replace(" / ", "/") if m["category"] else 'others'
                                 for
                                 m in
                                 model_info_data}

            if room_type == "bedroom":
                room_ids = set(room.room_id for room in d.rooms if "Bedroom" in room.room_id)
            elif room_type == "livingroom":
                room_ids = set(room.room_id for room in d.rooms if "Living" in room.room_id)
            elif room_type == "dinigroom":
                room_ids = set(room.room_id for room in d.rooms if "Dining" in room.room_id)

            # bproc.renderer.enable_normals_output()

            for current_bedroom_id in room_ids:
                # if current_bedroom_id != "MasterBedroom-18030":
                #     continue
                if current_bedroom_id not in valid_room_ids:
                    continue

                room_output_folder = f"{scene_output_folder}_{current_bedroom_id}"

                if Path(room_output_folder).exists():
                    continue

                bproc.init()
                RendererUtility.set_max_amount_of_samples(32)

                mapping_file = bproc.utility.resolve_resource(os.path.join("front_3D", "blender_label_mapping.csv"))
                mapping = bproc.utility.LabelIdMapping.from_csv(mapping_file)

                # set the light bounces
                bproc.renderer.set_light_bounces(diffuse_bounces=200, glossy_bounces=200, max_bounces=200,
                                                 transmission_bounces=200, transparent_max_bounces=200)
                # # set intrinsic parameters
                # bproc.camera.set_intrinsics_from_blender_params(lens=args.fov / 180 * np.pi, image_width=args.res_x,
                #                                                 image_height=args.res_y, clip_start=0.01, clip_end=5,
                #                                                 lens_unit="FOV", ortho_scale=10)

                # cam_K = bproc.camera.get_intrinsics_as_K_matrix()

                # load the front 3D objects again
                loaded_objects = bproc.loader.load_front3d(
                    json_path=str(front_json),
                    future_model_path=str(future_folder),
                    front_3D_texture_path=str(front_3D_texture_folder),
                    label_mapping=mapping,
                    model_id_to_label=model_id_to_label)

                layout_bbox = get_corners(layout_boxes[current_bedroom_id])

                planes = get_planes_from_bbox(layout_bbox)

                not_target_objects = []
                target_objects = []
                excluded_terms = ["Outer", "Top", "Bottom"]

                for tmp_object in bproc.object.get_all_mesh_objects():
                    bbox = tmp_object.get_bound_box()
                    centroid = get_centroid(bbox)
                    origin = tmp_object.get_origin()

                    if args.bound_slice:
                        if tmp_object.has_cp("room_id"):
                            if tmp_object.get_cp(
                                    "room_id") == current_bedroom_id:
                                # if is_close_to_plane(tmp_object, planes[0]):
                                #     traget_objects.append(tmp_object)
                                # else:
                                #     not_target_objects.append(tmp_object)
                                target_objects.append(tmp_object)
                            else:
                                not_target_objects.append(tmp_object)
                        elif "Floor" in tmp_object.get_name():
                            if is_majority_of_vertices_inside_bbox(tmp_object.get_mesh(),
                                                                   layout_bbox) and all(term not in tmp_object.get_name() for term in excluded_terms):
                                target_objects.append(tmp_object)
                            else:
                                not_target_objects.append(tmp_object)
                        # if "Wall" in tmp_object.get_name() or "Floor" in tmp_object.get_name() or "Ceiling" in tmp_object.get_name():
                        #     if is_majority_of_vertices_inside_bbox(tmp_object.get_mesh(),
                        #                                            layout_bbox) and all(term not in tmp_object.get_name() for term in excluded_terms):
                        #         target_objects.append(tmp_object)
                        #     else:
                        #         not_target_objects.append(tmp_object)
                        # if "Floor" in tmp_object.get_name():
                        #     if is_majority_of_vertices_inside_bbox(tmp_object.get_mesh(),
                        #                                            layout_bbox) and all(
                        #         term not in tmp_object.get_name() for term in excluded_terms):
                        #         target_objects.append(tmp_object)
                        #     else:
                        #         not_target_objects.append(tmp_object)
                        else:
                            not_target_objects.append(tmp_object)

                #          Sample materials
                # -------------------------------------------------------------------------
                # -------------------------------------------------------------------------
                cc_materials = bproc.loader.load_ccmaterials(args.cc_material_folder,
                                                             ["Bricks", "Wood", "Carpet", "Tile", "Marble"])

                floors = bproc.filter.by_attr(loaded_objects, "name", "Floor.*", regex=True)
                for floor in floors:
                    # For each material of the object
                    for i in range(len(floor.get_materials())):
                        floor.set_material(i, random.choice(cc_materials))

                baseboards_and_doors = bproc.filter.by_attr(loaded_objects, "name", "Baseboard.*|Door.*",
                                                            regex=True)
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

                bproc.object.delete_multiple(not_target_objects)

                if len(planes) > 6:
                    continue

                debug = False
                only_floor = True
                print("process room: ", current_bedroom_id)
                for plane in planes:
                    if debug:
                        if np.array_equal(plane[1], np.array([0, 0, -1])):
                            room_process_by_plane(plane, target_objects, loaded_objects, only_floor, args)
                            print("===================== finish one plane =======================")
                            break
                        else:
                            continue
                    if only_floor:
                        if np.array_equal(plane[1], np.array([0, 0, -1])):
                            room_process_by_plane(plane, target_objects, loaded_objects, only_floor, args)
                            break
                    else:
                        room_process_by_plane(plane, target_objects, loaded_objects, only_floor, args)

                # room_process_by_plane(planes[5], target_objects, loaded_objects, args)

            print('Time elapsed: %f.' % (time() - start_time))

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
