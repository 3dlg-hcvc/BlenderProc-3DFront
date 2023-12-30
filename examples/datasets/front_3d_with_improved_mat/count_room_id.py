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
from tqdm import tqdm

sys.path.append('./')
from visualization.front3d import Threed_Front_Config
from visualization.front3d.tools.threed_front import ThreedFront
from examples.datasets.front_3d_with_improved_mat.view_tool import ViewGenerator


# import pydevd_pycharm
# pydevd_pycharm.settrace('localhost', port=12345, stdoutToServer=True, stderrToServer=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("front_folder", default="/localhome/xsa55/Xiaohao/SemDiffLayout/datasets/front_3d_with_improved_mat/3D-FRONT",
                        help="Path to the 3D front file")
    parser.add_argument("future_folder", default="/localhome/xsa55/Xiaohao/SemDiffLayout/datasets/front_3d_with_improved_mat/3D-FUTURE-model",
                        help="Path to the 3D Future Model folder.")
    parser.add_argument("front_3D_texture_folder", default="/localhome/xsa55/Xiaohao/SemDiffLayout/datasets/front_3d_with_improved_mat/3D-FRONT-texture",
                        help="Path to the 3D FRONT texture folder.")
    parser.add_argument("front_json",
                        help="Path to a 3D FRONT scene json file, e.g.6a0e73bc-d0c4-4a38-bfb6-e083ce05ebe9.json.")
    parser.add_argument('cc_material_folder', nargs='?', default="/localhome/xsa55/Xiaohao/SemDiffLayout/dependencies/BlenderProc-3DFront/resources/cctextures",
                        help="Path to CCTextures folder, see the /scripts for the download script.")
    parser.add_argument("output_folder", nargs='?', default="/localhome/xsa55/Xiaohao/SemDiffLayout/datasets/front_3d_with_improved_mat/renderings",
                        help="Path to where the data should be saved")
    parser.add_argument("--n_views_per_scene", type=int, default=100,
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

def is_point_inside_box(point, box_corners, margin=0.01):
    # Get the min and max coordinates of the box in x, y, and z directions
    min_corner = [min([corner[i] for corner in box_corners]) - margin for i in range(3)]
    max_corner = [max([corner[i] for corner in box_corners]) + margin for i in range(3)]

    # Check if point lies inside the box
    for i in range(3):
        if point[i] < min_corner[i] or point[i] > max_corner[i]:
            return False
    return True


def is_majority_of_vertices_inside_bbox(mesh, bbox_corners, threshold=0.6):
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


def check_name(name, category_name):
    return True if category_name in name.lower() else False


if __name__ == '__main__':
    data_path = "/localhome/xsa55/Xiaohao/SemDiffLayout/datasets/front_3d_with_improved_mat/3D-FRONT"
    # read all files under 3d-front folder
    all_files = os.listdir(data_path)
    room_ids = []
    scene_ids = []
    match_count = 0

    split_path = "/localhome/xsa55/Xiaohao/SemDiffLayout/scripts/visualization/config/bedroom_threed_front_splits.csv"
    train_room_ids = []
    with open(split_path, "r") as f:
        for line in f:
            if "train" in line:
                train_room_ids.append(line.strip().split(",")[0])

    for scene_json in tqdm(all_files):
        '''Parse folders / file paths'''
        args = parse_args()
        front_folder, future_folder, front_3D_texture_folder, cc_material_folder, output_folder = get_folders(args)
        front_json = front_folder.joinpath(scene_json)
        # n_cameras = args.n_views_per_scene

        scene_name = front_json.name[:-len(front_json.suffix)]
        # print('Processing scene name: %s.' % (scene_name))

        dataset_config = Threed_Front_Config()
        dataset_config.init_generic_categories_by_room_type('all')
        json_file = [str(front_json).split("/")[-1]]
        scene_rooms = []
        try:
            d = ThreedFront.from_dataset_directory(
                str(dataset_config.threed_front_dir),
                str(dataset_config.model_info_path),
                str(dataset_config.threed_future_dir),
                str(dataset_config.dump_dir_to_scenes),
                path_to_room_masks_dir=None,
                path_to_bounds=None,
                json_files=json_file,
                filter_fn=lambda s: s)
            if len(d.rooms) == 0:
                continue

            for room in d.rooms:
                room_id = room.room_id
                room_ids.append(room_id)
                scene_ids.append(scene_name)
                if room_id in train_room_ids:
                    match_count += 1
                scene_rooms.append(room_id)
            # if len(scene_rooms) != len(set(scene_rooms)):
            #     import pdb
            #     pdb.set_trace()


        except:
            print("error scene_id: ", scene_name)

    print("total number of rooms in the training set: ", match_count)
    print("total number of rooms in the train split: ", len(train_room_ids))
    print("total number of rooms in the 3d-front dataset: ", len(room_ids))
    # save room_ids in a txt file
    with open("room_ids.txt", "w") as f:
        for room_id in room_ids:
            f.write(room_id + "\n")

    # save room_ids and scene_ids in a csv file
    with open("room_ids_and_scene_ids.csv", "w") as f:
        for room_id, scene_id in zip(room_ids, scene_ids):
            f.write(room_id + "," + scene_id + "\n")

