import math
import numpy as np


class ViewGenerator:
    meters_to_virtual_unit = 1  # Define a constant for meters to virtual unit conversion
    world_up = np.array([0, 1, 0])  # Assuming Y is up
    world_front = np.array([0, 0, -1])  # Assuming -Z is front

    def __init__(self):
        self.default_distance_scale = 1  # Define a default distance scale

    def get_frustum_params(self, object_depth, object_width, pixel_width, eps=0.01):
        fov = 2 * math.atan(pixel_width / object_depth)
        object_width_far = 2 * (pixel_width + (object_width / 2))
        far = object_width_far * object_depth / (2 * pixel_width) + eps
        near = far - object_depth - eps
        return {'fov': math.degrees(fov), 'near': near, 'far': far}

    def __get_view_bbox_dims(self, dims, theta, phi):
        st = abs(math.sin(theta))
        ct = abs(math.cos(theta))
        sp = abs(math.sin(phi))
        cp = abs(math.cos(phi))
        w = cp * dims[0] + sp * dims[2]
        h = ct * dims[1] + st * (sp * dims[0] + cp * dims[2])
        d = st * dims[1] + ct * (sp * dims[0] + cp * dims[2])
        return [w, h, d]

    def get_fixed_resolution_view_for_bbox(self, bbox, pixel_width, theta=0, phi=0,
                                           use_square_image=False):
        pixel_width = pixel_width or 0.01
        theta = theta if theta is not None else math.pi / 2
        phi = phi if phi is not None else 0

        dims = bbox["dimensions"]
        view_dims = self.__get_view_bbox_dims(dims, theta, phi)
        image_height_vu = max(view_dims) * self.default_distance_scale
        object_depth_vu = view_dims[2] * self.default_distance_scale

        params = self.get_frustum_params(object_depth_vu, image_height_vu, pixel_width * self.meters_to_virtual_unit)
        dists = np.array([params['far'] - object_depth_vu / 2] * 3)
        view_opts = self.get_view_for_point(bbox["centroid"], theta, phi, dists, params['fov'], params['near'],
                                            params['far'])
        view_opts['image_height_meters'] = image_height_vu / self.meters_to_virtual_unit
        view_opts['object_depth_meters'] = object_depth_vu / self.meters_to_virtual_unit
        view_opts['pixel_width_meters'] = pixel_width
        view_opts['image_size'] = [math.ceil(
            view_opts['image_height_meters'] / view_opts['pixel_width_meters'])] * 2 if use_square_image else [
            math.ceil(view_dims[0] / self.meters_to_virtual_unit / view_opts['pixel_width_meters']),
            math.ceil(view_dims[1] / self.meters_to_virtual_unit / view_opts['pixel_width_meters'])
        ]
        return view_opts

    def get_view_for_point(self, target, theta, phi, dists, fov, near, far):
        ry = dists[1]
        rz = dists[2] * math.cos(phi) * -1
        rx = dists[0] * math.sin(phi)
        cam_x = target[0] + (rx * math.cos(theta))
        cam_y = target[1] + (ry * math.sin(theta))
        cam_z = target[2] + (rz * math.cos(theta))

        eye = np.array([cam_x, cam_y, cam_z])
        up = self.world_up.copy()
        cam_vec = np.array(target) - eye
        lookat_up = None
        dot = abs(np.linalg.norm(cam_vec) * np.linalg.norm(up))
        if dot > 0.95:
            lookat_up = -self.world_front.copy()
        return {'position': eye.tolist(), 'target': target, 'lookat_up': lookat_up, 'up': up.tolist(),
                'fov': fov, 'near': near, 'far': far}
