import lmdb
import glob
import yaml
import math
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
from common.augmenter import augment
#from utils import filter_sem
from utils import filter_sem, filter_sem_cbs_per_channel

MAP_SIZE = 96
PIXELS_PER_METER = 3

class CBSDataset(Dataset):
    def __init__(self, data_dir, config_path, jitter=False):
        super().__init__()
        self.augmenter = augment(0.5)

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        self.crop_size = 64
        self.margin = (MAP_SIZE - self.crop_size)//2

        self.T = config['num_plan']
        self.x_jitter  = config['x_jitter'] if jitter else 0
        self.a_jitter  = config['a_jitter'] if jitter else 0
        self.seg_channels = config['seg_channels']
        self.camera_yaws = config['camera_yaws']
        self.crop_top = config['crop_top']
        self.crop_bottom = config['crop_bottom']

        self.num_frames = 0
        self.txn_map = dict()
        self.idx_map = dict()
        self.yaw_map = dict()
        self.file_map = dict()

        # Load dataset
        print(data_dir)
        for full_path in glob.glob(f'{data_dir}/**'):
            txn = lmdb.open(
                full_path,
                max_readers=1, readonly=True,
                lock=False, readahead=False, meminit=False).begin(write=False)

            n = int(txn.get('len'.encode()))
            if n < self.T+1:
                print (full_path, ' is too small. consider deleting it.')
                txn.__exit__()
            else:
                offset = self.num_frames
                for i in range(n-self.T):
                    self.num_frames += 1
                    for j in range(len(self.camera_yaws)):
                        self.txn_map[(offset+i)*len(self.camera_yaws)+j] = txn
                        self.idx_map[(offset+i)*len(self.camera_yaws)+j] = i
                        self.yaw_map[(offset+i)*len(self.camera_yaws)+j] = j
                        self.file_map[(offset+i)*len(self.camera_yaws)+j] = full_path

        print(f'{data_dir}: {self.num_frames} frames (x{len(self.camera_yaws)})')

    def __len__(self):
        return self.num_frames*len(self.camera_yaws)

    def __getitem__(self, idx):

        lmdb_txn = self.txn_map[idx]
        index = self.idx_map[idx]
        cam_index = self.yaw_map[idx]

        locs = self.__class__.access('loc', lmdb_txn, index, self.T+1, dtype=np.float32)[:,:2]
        rot = self.__class__.access('rot', lmdb_txn, index, 1, dtype=np.float32)
        spd = self.__class__.access('spd', lmdb_txn, index, 1, dtype=np.float32)
        lbl = self.__class__.access('lbl', lmdb_txn, index, 1, dtype=np.uint8).reshape(96,96,12)
        cmd = self.__class__.access('cmd', lmdb_txn, index, 1, dtype=np.float32).flatten()
        tls = self.__class__.access('tls', lmdb_txn, index, 1, dtype=np.uint8)
        ego_locs = self.__class__.access('ego_location', lmdb_txn, index, self.T+1, dtype=np.float32)
        ego_rots = self.__class__.access('ego_rotation', lmdb_txn, index, self.T+1, dtype=np.float32)

        rgb = self.__class__.access('wide_rgb_{}'.format(cam_index),  lmdb_txn, index, 1, dtype=np.uint8).reshape(240,480,3)
        sem = self.__class__.access('wide_sem_{}'.format(cam_index),  lmdb_txn, index, 1, dtype=np.uint8).reshape(240,480)

        yaw = float(rot)
        spd = float(spd)

        # Jitter
        x_jitter = np.random.randint(-self.x_jitter,self.x_jitter+1)
        #a_jitter = np.random.randint(-self.a_jitter,self.a_jitter+1) + self.camera_yaws[cam_index]


        # Rotate BEV
        # Cannot perform this data augmentation anymore a_jitter+yaw+90=angle=0
        #lbl = rotate_image(lbl, a_jitter+yaw+90)
        lbl = lbl[self.margin:self.margin+self.crop_size,self.margin+x_jitter:self.margin+x_jitter+self.crop_size]

        # Rotate locs
        # Thus locations must not be adapted according to rotated birdview anymore...
        #dloc = rotate_points(locs[1:] - locs[0:1], -a_jitter-yaw-90)*PIXELS_PER_METER - [x_jitter-self.crop_size/2,-self.crop_size/2]
        dloc = rotate_points(locs[1:] - locs[0:1], 0)*PIXELS_PER_METER - [x_jitter-self.crop_size/2,-self.crop_size/2]
        #sensor_transform = Transform(Location(cam_x, cam_y, cam_z), Rotation(cam_yaw, cam_roll,cam_pitch))
        #sensor_transform = Transform(Location(), Rotation())
        #self.converter = CoordinateConverter(sensor_transform, fov=120)
        #locs_sem = self.get_waypoints(ego_locs, ego_rots, tls)
        locs_sem = locs #TODO

        # Augment RGB
        rgb = self.augmenter(images=rgb[None,...,::-1])[0]
        sem = filter_sem(sem, labels=self.seg_channels)
        sem_channels_tls = filter_sem_cbs_per_channel(sem, tls, labels=self.seg_channels)

        #print(f'before: {rgb.shape}')
        rgb = rgb[self.crop_top:-self.crop_bottom]
        #print(f'after: {rgb.shape}')
        sem = sem[self.crop_top:-self.crop_bottom]
        #print(f'before: {sem_channels_tls.shape}')
        sem_channels_tls = sem_channels_tls[self.crop_top:-self.crop_bottom]
        #print(f'after: {sem_channels_tls.shape}')

        return rgb, lbl, sem, dloc, spd, int(cmd), sem_channels_tls, locs_sem

    @staticmethod
    def access(tag, lmdb_txn, index, T, dtype=np.float32):
        return np.stack([np.frombuffer(lmdb_txn.get((f'{tag}_{t:05d}').encode()), dtype) for t in range(index,index+T)])


    '''#Added for CBS
    #def project_vehicle(self, x, y, z, ori_x, ori_y, ori_z):
    def project_vehicle(self, pos, ori):
            pos = np.array([x, y, z])
            ori = np.array([ori_x, ori_y, ori_z])
            ori /= np.linalg.norm(ori)  # Make unit vector

            new_pos = pos + 4 * ori
            return self.converter.convert(np.array([new_pos]))

    @staticmethod
    def interpolate_waypoints(points):
        points = points[:, :2]

        # Fit first or second function through points
        n_degree = 2 if points.shape[0] > 2 else 1
        z = np.polyfit(points[:, 0], points[:, 1], n_degree)
        p = np.poly1d(z)

        # Keep interpolating until we have 5 points
        while points.shape[0] < 5:
            points_2 = np.vstack([points[0], points[:-1]])
            max_id = np.argmax(np.linalg.norm(points - points_2, axis=1))
            _x = np.mean([points[max_id], points_2[max_id]], axis=0)[0]
            points = np.insert(points, max_id, np.array([_x, p(_x)]), 0)

        return points

    def get_waypoints(self, pos, ori, tls):
        if tls:
            # Leads to a stop: the 5 waypoints are just in front of the vehicle (= project_vehicle)
            vehicle_proj = self.project_vehicle(pos[0], ori[0])
            output = np.array([vehicle_proj[0] for _ in range(self.T)])
            return output

        output = []
        for i in range(self.T+1):
            if len(output) == self.T:
                break
            image_coords = self.converter.convert(np.array([[pos[i,0], pos[i,1], pos[i,2]]]))
            if len(image_coords) > 0:
                output.append(image_coords[0])

        if len(output) < 2:
            # Leads to a stop
            vehicle_proj = self.project_vehicle(pos[0], ori[0])
            output = np.array([vehicle_proj[0] for _ in range(self.T)])
            return output

        if 2 <= len(output) < self.T:
            return self.interpolate_waypoints(np.array(output))
        return np.array(output)'''


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)

    return result

def rotate_points(points, angle):
    radian = angle * math.pi/180
    return points @ np.array([[math.cos(radian), math.sin(radian)], [-math.sin(radian), math.cos(radian)]])


def data_loader(memory, batch_size):
    return DataLoader(memory, batch_size=batch_size, num_workers=8, drop_last=True, shuffle=True)



'''class CoordinateConverter():
    def __init__(self, sensor_transform, img_width=480, img_height=240, fov=120):
        self.sensor_transform = sensor_transform
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
        return "Transform({}, {})".format(self.location, self.rotation)'''
