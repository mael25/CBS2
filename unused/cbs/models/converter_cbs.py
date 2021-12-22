import numpy as np
import torch

class CoordinateConverterCBS():
    def __init__(self, sensor_transform, img_width=480, img_height=224, fov=120):
        self.sensor_transform = sensor_transform
        self.img_size = np.array([img_width, img_height])
        f = img_width / (2 * np.tan(fov * np.pi / 360))
        self.K = np.array([[f, 0., img_width/2],
                           [0, f, img_height/2],
                           [0., 0., 1.]])

    @staticmethod
    def camera_coords_downscale(coords):
        raise NotImplemented("TODO")

    def _world_to_sensor(self, cords):
        """
        Transforms world coordinates to sensor.
        """
        sensor_world_matrix = self.get_matrix(self.sensor_transform)
        world_sensor_matrix = np.linalg.inv(sensor_world_matrix)
        sensor_cords = np.dot(world_sensor_matrix, cords)
        return sensor_cords

    @staticmethod
    def get_matrix(transform):
        """
        Creates matrix from carla transform.
        """
        rotation = transform.rotation
        location = transform.location
        c_y = np.cos(np.radians(rotation.yaw))
        s_y = np.sin(np.radians(rotation.yaw))
        c_r = np.cos(np.radians(rotation.roll))
        s_r = np.sin(np.radians(rotation.roll))
        c_p = np.cos(np.radians(rotation.pitch))
        s_p = np.sin(np.radians(rotation.pitch))
        matrix = np.matrix(np.identity(4))
        matrix[0, 3] = location.x
        matrix[1, 3] = location.y
        matrix[2, 3] = location.z
        matrix[0, 0] = c_p * c_y
        matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
        matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
        matrix[1, 0] = s_y * c_p
        matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
        matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
        matrix[2, 0] = s_p
        matrix[2, 1] = -c_p * s_r
        matrix[2, 2] = c_p * c_r

        return matrix

    def convert(self, world_coords):
        # Convert to homogenous coordinates
        world_coords = np.hstack((world_coords, np.ones((world_coords.shape[0], 1))))

        # Convert world to sensor
        cords_x_y_z = self._world_to_sensor(world_coords.T).T

        # Convert sensor to camera
        cords_y_minus_z_x = np.concatenate([cords_x_y_z[:, 1], -cords_x_y_z[:, 2], cords_x_y_z[:, 0]],axis=1).T
        temp = np.transpose(np.dot(self.K, cords_y_minus_z_x))
        camera_coords = np.concatenate([temp[:, 0] / temp[:, 2], temp[:, 1] / temp[:, 2], temp[:, 2]],axis=1)
        camera_coords = np.array(camera_coords)

        # Mask invalid coordinates
        mask = np.all([np.all(camera_coords >= 0, axis=1), camera_coords[:, 0] <= 480,camera_coords[:, 1] <= 240], axis=0)
        return camera_coords[mask]

    def unproject(self, output, world_y=1.5, fov=120, fixed_offset=0):
        cx, cy = self.img_size / 2

        w, h = self.img_size

        f = w / (2 * np.tan(fov * np.pi / 360))

        xt = (output[..., 0:1] - cx) / f
        yt = (output[..., 1:2] - cy) / f

        world_z = world_y / yt
        world_x = world_z * xt

        world_output = np.stack([world_x, world_z], axis=-1)

        if fixed_offset:
            world_output[..., 1] -= fixed_offset

        world_output = world_output.squeeze()

        return world_output


class Location():
    def __init__(self, x=1.5, y=0, z=2.4):
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self, ):
        return self.__str__()

    def __str__(self, ):
        return "Location(x={}, y={}, z={})".format(self.x, self.y, self.z)


class Rotation():
    def __init__(self, p=0, y=0, r=0):
        self.pitch = p
        self.yaw = y
        self.roll = r

    def __repr__(self, ):
        return self.__str__()

    def __str__(self):
        return "Rotation(pitch={}, yaw={}, roll={})".format(self.pitch, self.yaw, self.roll)


class Transform():
    def __init__(self, loc, rot):
        self.location = loc
        self.rotation = rot

    def __repr__(self, ):
        return self.__str__()

    def __str__(self, ):
        return "Transform({}, {})".format(self.location, self.rotation)
