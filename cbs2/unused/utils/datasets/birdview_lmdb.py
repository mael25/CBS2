from pathlib import Path

import torch
import lmdb
import os
import glob
import numpy as np
import cv2

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from utils.image_utils import draw_msra_gaussian, gaussian_radius, CoordinateConverter
from utils.carla_utils import visualize_birdview
import imgaug.augmenters as iaa
import imgaug as ia
from imgaug.augmentables import Keypoint, KeypointsOnImage

import math
import random

PIXEL_OFFSET = 10
N_CLASSES = 13
N_TRAFFIC_LIGHT_STATES = 1
N_CLASSES_COMBINED = 5
KEEP_CLASSES = {4, 6, 7, 10, 12}  # pedestrians, roadlines, roads, vehicles, trafficsigns


class Location():
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self, ):
        return self.__str__()

    def __str__(self, ):
        return "Location(x={}, y={}, z={})".format(self.x, self.y, self.z)


class Rotation():
    def __init__(self, p, y, r):
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


def one_hot_tl(tl: int):
    """
    Converts categorical traffic light to one hot encoded vector.
    @param tl: categorical traffic light value
    returns: one hot encoded vector
    """
    # Add one to number of states for 'no traffic lights' option
    out = np.zeros(N_TRAFFIC_LIGHT_STATES+1)
    out[tl] = 1

    return out


def seg2D_to_ND_combined(seg, tl_info, walker_info, vehicle_info):
    seg = seg[:, :, 0]  # CARLA stores segmentation values in R channel
    mask = np.zeros((*seg.shape, N_CLASSES_COMBINED))

    for i, seg_class in enumerate(KEEP_CLASSES):
        if seg_class == 12:
            mask[..., i][seg == seg_class] = tl_info
        elif seg_class == 4:
            mask[..., i][seg == seg_class] = walker_info
        elif seg_class == 10:
            mask[..., i][seg == seg_class] = vehicle_info
        else:
            mask[..., i][seg == seg_class] = 1

    # TO_COMBINE = set(range(N_CLASSES)) - KEEP_CLASSES
    # for i in TO_COMBINE:
    #     mask[..., len(KEEP_CLASSES)] += seg == i

    return mask


def seg2D_to_ND(seg, tl_info, walker_info, vehicle_info, combine=False):
    """Converts 2D segmentation image to ND array with N boolean masks.
    Where N corresponds to number of segmentation classes."""
    if combine:
        return seg2D_to_ND_combined(seg, tl_info, walker_info, vehicle_info)

    seg = seg[:, :, 0]  # CARLA stores segmentation values in R channel
    mask = np.zeros((*seg.shape, N_CLASSES))

    # Do not add traffic light state yet
    for i in range(N_CLASSES-1):
        mask[..., i][seg == i] = 1

    # Add traffic light state, 12 is traffic sign class

    ############################
    # MODIF (11-09-2021)
    #mask[..., 12][seg == 12] = tl_info #original
    #print('\n\n Using modified segmentation for tl')
    mask[..., 12] = tl_info
    ############################
    return mask


def world_to_pixel(
        x,y,ox,oy,ori_ox, ori_oy,
        pixels_per_meter=5, offset=(-80,160), size=320, angle_jitter=15):
    pixel_dx, pixel_dy = (x-ox)*pixels_per_meter, (y-oy)*pixels_per_meter

    pixel_x = pixel_dx*ori_ox+pixel_dy*ori_oy
    pixel_y = -pixel_dx*ori_oy+pixel_dy*ori_ox

    pixel_x = size-pixel_x

    return np.array([pixel_x, pixel_y]) + offset


class BirdViewDataset(Dataset):
    def __init__(
            self, dataset_path,
            img_size=320, crop_size=192, gap=5, n_step=5,
            crop_x_jitter=15, crop_y_jitter=5, angle_jitter=5, scale=1.,
            down_ratio=4, gaussian_radius=1.0, buffer=40, max_frames=None,
            combine_seg=False, segmentation=False, is_train=False):

        # These typically don't change.
        self.img_size = img_size
        self.crop_size = crop_size
        self.down_ratio = down_ratio
        self.gap = gap
        self.ori_gap = gap
        self.n_step = n_step
        self.buffer = buffer
        self.crop_size_y = 80
        self.is_train = is_train

        self.max_frames = max_frames

        self.crop_x_jitter = crop_x_jitter
        self.crop_y_jitter = crop_y_jitter
        self.angle_jitter = angle_jitter

        self.scale = scale
        self.gaussian_radius = gaussian_radius

        self._name_map = {}
        self.file_map = {}
        self.idx_map = {}

        self.combine_seg = combine_seg
        self.segmentation = segmentation

        self.bird_view_transform = transforms.ToTensor()

        self.seq = self.init_aug()

        n_episodes = 0

        for full_path in sorted(glob.glob('%s/**' % dataset_path), reverse=True):
            txn = lmdb.open(
                    full_path,
                    max_readers=1, readonly=True,
                    lock=False, readahead=False, meminit=False).begin(write=False)

            n = int(txn.get('len'.encode())) - (self.gap + self.buffer) * self.n_step
            offset = len(self._name_map)

            for i in range(n):
                if max_frames and len(self) >= max_frames:
                    break

                self._name_map[offset+i] = full_path
                self.file_map[offset+i] = txn
                self.idx_map[offset+i] = i

            n_episodes += 1

            if max_frames and len(self) >= max_frames:
                break

        print('%s: %d frames, %d episodes.' % (dataset_path, len(self), n_episodes))

    @staticmethod
    def points_to_keypoints(points, shape):
        kps = KeypointsOnImage([
            Keypoint(x=points[0][0], y=points[0][1]),
            Keypoint(x=points[1][0], y=points[1][1]),
            Keypoint(x=points[2][0], y=points[2][1]),
            Keypoint(x=points[3][0], y=points[3][1]),
            Keypoint(x=points[4][0], y=points[4][1])
        ], shape=shape)

        return kps

    def init_aug(self):
        sometimes = lambda aug: iaa.Sometimes(0.33, aug)
        seq = iaa.Sequential(
            [
                sometimes(
                    iaa.Affine(
                        translate_px={"x": (-self.crop_x_jitter,
                                            self.crop_x_jitter),
                                      "y": (-self.crop_y_jitter, 0)},
                        scale={"x": (1, self.scale), "y": (1, self.scale)},
                        mode='reflect')
                ),
                sometimes(
                    iaa.Dropout(
                        p=(0, 0.1),
                        per_channel=0.5)
                ),
                sometimes(
                    iaa.MotionBlur(
                        k=(3, 5),
                        angle=[-90, 0, 90])
                )
            ],
            random_order=True)

        print("Using x translate {}, y translate {}, x and y scale {}".format(self.crop_x_jitter, self.crop_y_jitter, self.scale))
        return seq

    @staticmethod
    def down_scale(img):
        new_shape = (img.shape[0] // 2, img.shape[1] // 2)
        img = np.moveaxis(img, 1, 0)
        img = cv2.resize(img.astype(np.float32), new_shape)
        img = np.moveaxis(img, 1, 0)

        return img

    def augment_image(self, image, points, redo=0):
        if redo >= 5:
            return image, points

        image_aug, p_aug = self.seq(image=image,
                                    keypoints=self.points_to_keypoints(points / 2, (80, 192)))
        temp = p_aug.to_xy_array()
        if not np.all((temp[:, 0] <= 192) & (temp[:, 1] <= 80)):
            return self.augment_image(image, points, redo=redo + 1)

        return image_aug, temp

    def __len__(self):
        return len(self.file_map)

    def project_vehicle(self, x, y, z, ori_x, ori_y, ori_z):
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

    def get_waypoints(self, index, lmdb_txn, world_x, world_y, world_z, ori_x, ori_y, ori_z):
        tl = int.from_bytes(lmdb_txn.get(('trafficlights_%04d' % index).encode()), 'little')

        ############################
        # MODIF (11-09-2021)
        if self.combine_seg:
            vehicle = int.from_bytes(lmdb_txn.get(('vehicles_%04d' % index).encode()), 'little')
            walker = int.from_bytes(lmdb_txn.get(('walkers_%04d' % index).encode()), 'little')
        else:
            #These variables won't be used if we are not using combined segmentation
            walker = 0
            vehicle = 0
        ############################

        if tl or vehicle or walker:
            vehicle_proj = self.project_vehicle(world_x, world_y, world_z, ori_x, ori_y, ori_z)
            output = np.array([vehicle_proj[0] for _ in range(5)])
            return output, True

        output = []
        for i in range(index, (index + (self.n_step + 1 + self.buffer * self.gap)), self.gap):
            if len(output) == self.n_step:
                break

            x, y, z = np.frombuffer(lmdb_txn.get(('measurements_%04d' % i).encode()), np.float32)[:3]
            image_coords = self.converter.convert(np.array([[x, y, z]]))
            if len(image_coords) > 0:
                output.append(image_coords[0])

        if len(output) < 2:
            # First try with smaller GAP
            if self.gap == self.ori_gap:
                self.gap = 1
                return self.get_waypoints(index, lmdb_txn, world_x, world_y, world_z,ori_x, ori_y, ori_z)

            vehicle_proj = self.project_vehicle(world_x, world_y, world_z, ori_x,ori_y, ori_z)
            output = np.array([vehicle_proj[0] for _ in range(5)])
            return output, True

        if 2 <= len(output) < 5:
            return self.interpolate_waypoints(np.array(output)), False

        return np.array(output), False

    def __getitem__(self, idx):
        lmdb_txn = self.file_map[idx]
        index = self.idx_map[idx]

        bird_view = np.frombuffer(lmdb_txn.get(('birdview_%04d'%index).encode()), np.uint8).reshape(320,320,7)
        segmentation = np.frombuffer(lmdb_txn.get(('segmentation_%04d'%index).encode()), np.uint8).reshape(160, 384, 3)

        # Resize
        segmentation = self.down_scale(segmentation)
        assert_shape = (80, 192, 3)
        assert segmentation.shape == assert_shape, "Incorrect shape ({}), got {}".format(assert_shape, segmentation.shape)

        # tl_info_old = int.from_bytes(lmdb_txn.get(('OLD_trafficlights_%04d' % index).encode()), 'little')
        tl_info = int.from_bytes(lmdb_txn.get(('trafficlights_%04d' % index).encode()), 'little')

        ############################
        # MODIF (11-09-2021)
        if self.combine_seg:
            walker_info = int.from_bytes(lmdb_txn.get(('walkers_%04d' % index).encode()), 'little')
            vehicle_info = int.from_bytes(lmdb_txn.get(('vehicles_%04d' % index).encode()), 'little')
        else:
            #These variables won't be used if we are not using combined segmentation (seg2D_to_ND_combined not called), but still need to give values
            walker_info = 0
            vehicle_info = 0
        ############################
        segmentation = seg2D_to_ND(segmentation, tl_info,
                                   walker_info, vehicle_info,
                                   combine=self.combine_seg).astype(np.float32)
        measurement = np.frombuffer(lmdb_txn.get(('measurements_%04d'%index).encode()), np.float32)
        rgb_image = None

        ox, oy, oz, ori_ox, ori_oy, ori_oz, vx, vy, vz, cam_x, cam_y, cam_z, cam_yaw, cam_roll, cam_pitch, ax, ay, az, cmd, steer, throttle, brake, manual, gear  = measurement
        speed = np.linalg.norm([vx,vy,vz])

        # oangle = np.arctan2(ori_oy, ori_ox)
        delta_angle = np.random.randint(-self.angle_jitter,self.angle_jitter+1)
        dx = np.random.randint(-self.crop_x_jitter,self.crop_x_jitter+1)
        dy = np.random.randint(0,self.crop_y_jitter+1) - PIXEL_OFFSET

        # o_camx = ox + ori_ox*2
        # o_camy = oy + ori_oy*2

        if self.segmentation:
            # Create coordinate transformer
            sensor_transform = Transform(Location(cam_x, cam_y, cam_z),
                                         Rotation(cam_yaw, cam_roll,cam_pitch))
            self.converter = CoordinateConverter(sensor_transform, fov=120)

            # Get waypoints in image coordinates (x, y)
            image_coord_wp, full_stop = self.get_waypoints(index, lmdb_txn, ox, oy, oz,ori_ox, ori_oy,ori_oz)
            image_coord_wp = image_coord_wp[:,:2].astype(np.float32)

            self.gap = self.ori_gap  # Reset gap to its original value

            # Augment image
            if self.is_train:
                img_aug, points_aug = self.augment_image(segmentation, image_coord_wp)
            else:
                img_aug = segmentation
                points_aug = image_coord_wp / 2

            img_aug = np.moveaxis(img_aug, -1, 0)
            img_aug = img_aug.astype(np.float32)
            points_aug = np.array(points_aug, dtype=np.float32)

            assert_shape = (5, 80, 192) if self.combine_seg else (13, 80, 192)
            assert img_aug.shape == assert_shape, "Incorrect shape ({}), got {}".format(assert_shape, img_aug.shape)
            assert len(points_aug) == 5, "Not enough points, got {}".format(points_aug.shape)

            ###############################################
            # if full_stop and speed != 0.0:
            #    speed = float(0)
            ## MODIF 12/10/21 REVERT TO 9JUNE
            # if full_stop and speed != 0.0:
            #    speed = float(0)
            ###############################################

            return img_aug, points_aug, cmd, speed

        pixel_ox = 160
        pixel_oy = 260

        bird_view = cv2.warpAffine(
                bird_view,
                cv2.getRotationMatrix2D((pixel_ox,pixel_oy), delta_angle, 1.0),
                bird_view.shape[1::-1], flags=cv2.INTER_LINEAR)

        # random cropping
        center_x, center_y = 160, 260-self.crop_size//2
        bird_view = bird_view[
                dy+center_y-self.crop_size//2:dy+center_y+self.crop_size//2,
                dx+center_x-self.crop_size//2:dx+center_x+self.crop_size//2]

        angle = np.arctan2(ori_oy, ori_ox) + np.deg2rad(delta_angle)
        ori_ox, ori_oy = np.cos(angle), np.sin(angle)

        locations = []
        orientations = []

        for dt in range(self.gap, self.gap*(self.n_step+1), self.gap):
            lmdb_txn = self.file_map[idx]
            index =self.idx_map[idx]+dt

            f_measurement = np.frombuffer(lmdb_txn.get(("measurements_%04d"%index).encode()), np.float32)
            x, y, z, ori_x, ori_y = f_measurement[:5]

            pixel_y, pixel_x = world_to_pixel(x,y,ox,oy,ori_ox,ori_oy,size=self.img_size)
            pixel_x = pixel_x - (self.img_size-self.crop_size)//2
            pixel_y = self.crop_size - (self.img_size-pixel_y)+70

            pixel_x -= dx
            pixel_y -= dy

            angle = np.arctan2(ori_y, ori_x) - np.arctan2(ori_oy, ori_ox)
            ori_dx, ori_dy = np.cos(angle), np.sin(angle)

            locations.append([pixel_x, pixel_y])
            orientations.append([ori_dx, ori_dy])

        bird_view = self.bird_view_transform(bird_view)

        # Create mask
        output_size = self.crop_size // self.down_ratio
        heatmap_mask = np.zeros((self.n_step, output_size, output_size), dtype=np.float32)
        regression_offset = np.zeros((self.n_step,2), np.float32)
        indices = np.zeros((self.n_step), dtype=np.int64)

        for i, (pixel_x, pixel_y) in enumerate(locations):
            center = np.array(
                    [pixel_x / self.down_ratio, pixel_y / self.down_ratio],
                    dtype=np.float32)
            center = np.clip(center, 0, output_size-1)
            center_int = np.rint(center)

            draw_msra_gaussian(heatmap_mask[i], center_int, self.gaussian_radius)
            regression_offset[i] = center - center_int
            indices[i] = center_int[1] * output_size + center_int[0]

        return bird_view, np.array(locations), cmd, speed



class BiasedBirdViewDataset(BirdViewDataset):
    def __init__(self, dataset_path, left_ratio=0.25, right_ratio=0.25, straight_ratio=0.25, **kwargs):
        super().__init__(dataset_path, **kwargs)

        print ("Doing biased: %.2f,%.2f,%.2f"%(left_ratio, right_ratio, straight_ratio))

        self._choices = [1,2,3,4]
        self._weights = [left_ratio,right_ratio,straight_ratio,1-left_ratio-right_ratio-straight_ratio]
        # Separately save data on different cmd
        self.cmd_map = { i : set([]) for i in range(1,5)}

        for idx in range(len(self.file_map)):
            lmdb_txn = self.file_map[idx]
            index = self.idx_map[idx]

            measurement = np.frombuffer(lmdb_txn.get(('measurements_%04d'%index).encode()), np.float32)
            ox, oy, oz, ori_ox, ori_oy, vx, vy, vz, ax, ay, az, cmd, steer, throttle, brake, manual, gear = measurement
            speed = np.linalg.norm([vx,vy,vz])

            if cmd != 4 and speed > 1.0:
                self.cmd_map[cmd].add(idx)
            else:
                self.cmd_map[4].add(idx)

        for cmd, nums in self.cmd_map.items():
            print (cmd, len(nums))

    def __getitem__(self, idx):
        cmd = np.random.choice(self._choices, p=self._weights)
        [_idx] = random.sample(self.cmd_map[cmd], 1)
        return super(BiasedBirdViewDataset, self).__getitem__(_idx)



def load_birdview_data(
        dataset_dir,
        batch_size=32, num_workers=0, shuffle=True,
        crop_x_jitter=0, crop_y_jitter=0, angle_jitter=0, n_step=5, gap=5,
        max_frames=None, cmd_biased=False):
    if cmd_biased:
        dataset_cls = BiasedBirdViewDataset
    else:
        dataset_cls = BirdViewDataset

    dataset = dataset_cls(
        dataset_path,
        crop_x_jitter=crop_x_jitter,
        crop_y_jitter=crop_y_jitter,
        angle_jitter=angle_jitter,
        n_step=n_step,
        gap=gap,
        data_ratio=data_ratio,
    )

    return DataLoader(
            dataset,
            batch_size=batch_size, num_workers=num_workers,
            shuffle=shuffle, drop_last=True, pin_memory=True)


class Wrap(Dataset):
    def __init__(self, data, batch_size, samples):
        self.data = data
        self.batch_size = batch_size
        self.samples = samples

    def __len__(self):
        return self.batch_size * self.samples

    def __getitem__(self, i):
        return self.data[np.random.randint(len(self.data))]


def _dataloader(data, batch_size, num_workers):
    return DataLoader(
            data, batch_size=batch_size, num_workers=num_workers,
            shuffle=True, drop_last=True, pin_memory=True)


def get_birdview(
        dataset_dir,
        batch_size=32, num_workers=12, shuffle=True,
        crop_x_jitter=0, crop_y_jitter=0, angle_jitter=0, n_step=5, gap=5,
        max_frames=None, cmd_biased=False, combine_seg=False,
        segmentation=False, scale=1.):

    def make_dataset(dir_name, is_train):
        _dataset_dir = str(Path(dataset_dir) / dir_name)
        _samples = 1000 if is_train else 10
        _crop_x_jitter = crop_x_jitter if is_train else 0
        _crop_y_jitter = crop_y_jitter if is_train else 0
        _angle_jitter = angle_jitter if is_train else 0
        _max_frames = max_frames if is_train else None
        _num_workers = num_workers if is_train else 0
        _scale = scale if is_train else 1.

        if is_train and cmd_biased:
            dataset_cls = BiasedBirdViewDataset
        else:
            dataset_cls = BirdViewDataset

        data = dataset_cls(
                _dataset_dir, gap=gap, n_step=n_step,
                crop_x_jitter=_crop_x_jitter, crop_y_jitter=_crop_y_jitter,
                angle_jitter=_angle_jitter,
                max_frames=_max_frames, combine_seg=combine_seg,
            segmentation=segmentation, scale=_scale, is_train=is_train)
        data = Wrap(data, batch_size, _samples)
        data = _dataloader(data, batch_size, _num_workers)

        return data

    train = make_dataset('train', True)
    val = make_dataset('val', False)

    return train, val