from pathlib import Path

import torch
import lmdb
import os
import glob
import numpy as np
import cv2

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from utils.datasets.birdview_lmdb import seg2D_to_ND, Transform, Rotation, Location
import math
import random
from utils.image_utils import CoordinateConverter

#import augmenter
#from common.augmenter import augment

PIXEL_OFFSET = 10
PIXELS_PER_METER = 5
SEG_CLASSES = {4, 6, 7, 10, 18}  # pedestrians, roadlines, roads, vehicles, tl

def world_to_pixel(x,y,ox,oy,ori_ox, ori_oy, offset=(-80,160), size=320, angle_jitter=15):
    pixel_dx, pixel_dy = (x-ox)*PIXELS_PER_METER, (y-oy)*PIXELS_PER_METER

    pixel_x = pixel_dx*ori_ox+pixel_dy*ori_oy
    pixel_y = -pixel_dx*ori_oy+pixel_dy*ori_ox

    pixel_x = size-pixel_x

    return np.array([pixel_x, pixel_y]) + offset


def project_to_image(pixel_x, pixel_y, tran=[0.,0.,0.], rot=[0.,0.,0.], fov=90, w=384, h=160, camera_world_z=1.4, crop_size=192):
    # Apply fixed offset tp pixel_y
    pixel_y -= 2*PIXELS_PER_METER

    pixel_y = crop_size - pixel_y
    pixel_x = pixel_x - crop_size/2

    world_x = pixel_x / PIXELS_PER_METER
    world_y = pixel_y / PIXELS_PER_METER

    xyz = np.zeros((1,3))
    xyz[0,0] = world_x
    xyz[0,1] = camera_world_z
    xyz[0,2] = world_y

    f = w /(2 * np.tan(fov * np.pi / 360))
    A = np.array([
        [f, 0., w/2],
        [0, f, h/2],
        [0., 0., 1.]
    ])
    image_xy, _ = cv2.projectPoints(xyz, np.array(tran), np.array(rot), A, None)
    image_xy[...,0] = np.clip(image_xy[...,0], 0, w)
    image_xy[...,1] = np.clip(image_xy[...,1], 0, h)

    return image_xy[0,0]

class ImageDataset(Dataset):
    def __init__(self,
        dataset_path,
        rgb_shape=(160,384,3),
        img_size=320,
        crop_size=192,
        gap=5,
        n_step=5,
        gaussian_radius=1.,
        down_ratio=4,
        augment_strategy=None,
        batch_read_number=819200,
        batch_aug=1,
        buffer=40,
        combine_seg=False,
):
        self._name_map = {}

        self.file_map = {}
        self.idx_map = {}

        self.bird_view_transform = transforms.ToTensor()
        self.rgb_transform = transforms.ToTensor()

        self.rgb_shape = rgb_shape
        self.img_size = img_size
        self.crop_size = crop_size

        self.gap = gap
        self.ori_gap = gap
        self.buffer = buffer
        self.combine_seg = combine_seg

        self.n_step = n_step
        self.down_ratio = down_ratio
        self.batch_aug = batch_aug

        self.gaussian_radius = gaussian_radius

        # CBS RGB image augmentation
        # print("augment with ", augment_strategy)
        # if augment_strategy is not None and augment_strategy != 'None':
        #     self.augmenter = getattr(augmenter, augment_strategy)
        # else:
        #     self.augmenter = None

        # For CBS2, we use WoR RGB image augmentation
        print(f'WoR data augmentation approach, with batch_aug of {self.batch_aug} and p= 0.5')
        self.augmenter = augment(0.5)

        count = 0
        for full_path in glob.glob('%s/**'%dataset_path):
            lmdb_file = lmdb.open(full_path,
                 max_readers=1,
                 readonly=True,
                 lock=False,
                 readahead=False,
                 meminit=False
            )

            txn = lmdb_file.begin(write=False)

            N = int(txn.get('len'.encode())) - (self.gap + self.buffer) * self.n_step

            for _ in range(N):
                self._name_map[_+count] = full_path
                self.file_map[_+count] = txn
                self.idx_map[_+count] = _

            count += N

        print("Finished loading %s. Length: %d"%(dataset_path, count))
        self.batch_read_number = batch_read_number

    def __len__(self):
        return len(self.file_map)

    def project_vehicle(self, x, y, z, ori_x, ori_y, ori_z):
        pos = np.array([x, y, z])
        ori = np.array([ori_x, ori_y, ori_z])
        ori /= np.linalg.norm(ori)  # Make unit vector

        #new_pos = pos + 4 * ori
        fwd_2d_angle = np.deg2rad(ori_y) #yaw to rad
        new_pos = pos + 5.5 * np.array([np.cos(fwd_2d_angle), np.sin(fwd_2d_angle), 0])
        new_pos_cam_coords = self.converter.convert(np.array([new_pos]))
        if(new_pos_cam_coords.shape[0] == 0):
            return np.array([[192, 147, 0]]) # In the center of the image, almost at the bottom --> stop waypoint
        return new_pos_cam_coords

    @staticmethod
    def interpolate_waypoints(points):
        points = points[:, :2]

        # Fit first or second function through points
        n_degree = 2 if points.shape[0] > 2 else 1
        z = np.polyfit(points[:, 0], points[:, 1], n_degree)
        p = np.poly1d(z)

        # Keep interpolating until we have n_step points
        while points.shape[0] < self.n_step:
            points_2 = np.vstack([points[0], points[:-1]])
            max_id = np.argmax(np.linalg.norm(points - points_2, axis=1))
            _x = np.mean([points[max_id], points_2[max_id]], axis=0)[0]
            points = np.insert(points, max_id, np.array([_x, p(_x)]), 0)

        return points

    def get_waypoints(self, index, lmdb_txn, world_x, world_y, world_z, ori_x, ori_y, ori_z):
        tl = int.from_bytes(lmdb_txn.get(('trafficlights_%04d' % index).encode()), 'little')

        #if tl or vehicle or walker:
        if tl:
            vehicle_proj = self.project_vehicle(world_x, world_y, world_z, ori_x, ori_y, ori_z)
            output = np.array([vehicle_proj[0] for _ in range(self.n_step)])
            return output, True

        output = []
        for i in range(index, (index + (self.n_step + 1 + self.buffer * self.gap)), self.gap):
            if len(output) == self.n_step:
                break

            x, y, z = np.frombuffer(lmdb_txn.get(('loc_%04d' % i).encode()), np.float32)
            image_coords = self.converter.convert(np.array([[x, y, z]]))
            if len(image_coords) > 0:
                output.append(image_coords[0])

        if len(output) < 2:
            # First try with smaller GAP
            if self.gap == self.ori_gap:
                self.gap = 1
                return self.get_waypoints(index, lmdb_txn, world_x, world_y, world_z,ori_x, ori_y, ori_z)

            vehicle_proj = self.project_vehicle(world_x, world_y, world_z, ori_x,ori_y, ori_z)
            output = np.array([vehicle_proj[0] for _ in range(self.n_step)])
            return output, True

        if 2 <= len(output) < self.n_step:
            return self.interpolate_waypoints(np.array(output)), False

        return np.array(output), False

    @staticmethod
    def down_scale(img):
        new_shape = (img.shape[0] // 2, img.shape[1] // 2)
        img = np.moveaxis(img, 1, 0)
        img = cv2.resize(img.astype(np.float32), new_shape)
        img = np.moveaxis(img, 1, 0)

        return img

    def __getitem__(self, idx):

        lmdb_txn = self.file_map[idx]
        index = self.idx_map[idx]

        # bird_view = np.frombuffer(lmdb_txn.get(('birdview_%04d'%index).encode()), np.uint8).reshape(320,320,7)
        segmentation = np.frombuffer(lmdb_txn.get(('segmentation_%04d'%index).encode()), np.uint8).reshape(160, 384)
        segmentation = self.down_scale(segmentation)
        assert_shape = (80, 192)
        assert segmentation.shape == assert_shape, "Incorrect shape ({}), got {}".format(assert_shape, segmentation.shape)

        tl_info = int.from_bytes(lmdb_txn.get(('trafficlights_%04d' % index).encode()), 'little')

        segmentation = seg2D_to_ND(segmentation, tl_info).astype(np.float32)

        ox, oy, oz = np.frombuffer(lmdb_txn.get(('loc_%04d'%index).encode()), np.float32)
        ori_ox, ori_oy, ori_oz = np.frombuffer(lmdb_txn.get(('rot_%04d'%index).encode()), np.float32)
        speed = np.frombuffer(lmdb_txn.get(('spd_%04d'%index).encode()), np.float32)[0]
        cmd = int(np.frombuffer(lmdb_txn.get(('cmd_%04d'%index).encode()), np.float32)[0])
        cam_x, cam_y, cam_z = np.frombuffer(lmdb_txn.get(('cam_location_%04d'%index).encode()), np.float32)
        cam_pitch, cam_yaw, cam_roll = np.frombuffer(lmdb_txn.get(('cam_rotation_%04d'%index).encode()), np.float32)

        rgb_image = np.fromstring(lmdb_txn.get(('rgb_%04d'%index).encode()), np.uint8).reshape(160,384,3)

        # if self.augmenter:
        #     rgb_images = [self.augmenter(self.batch_read_number).augment_image(rgb_image) for i in range(self.batch_aug)]
        # else:
        #     rgb_images = [rgb_image for i in range(self.batch_aug)]
        #
        # if self.batch_aug == 1:
        #     rgb_images = rgb_images[0]

        if self.augmenter:
            rgb_images = [self.augmenter(rgb_image) for i in range(self.batch_aug)]
        else:
            rgb_images = [rgb_image for i in range(self.batch_aug)]

        if self.batch_aug == 1:
            rgb_images = rgb_images[0]

        # Create coordinate transformer
        sensor_transform = Transform(Location(cam_x, cam_y, cam_z),
                                     Rotation(cam_pitch, cam_yaw, cam_roll))
        self.converter = CoordinateConverter(sensor_transform, fov=120)

        # Get waypoints in image coordinates (x, y)
        image_coord_wp, full_stop = self.get_waypoints(index, lmdb_txn, ox, oy, oz,ori_ox, ori_oy,ori_oz)
        image_coord_wp = image_coord_wp[:,:2].astype(np.float32)

        self.gap = self.ori_gap  # Reset gap to its original value

        segmentation = segmentation.astype(np.float32)
        image_coord_wp = np.array(image_coord_wp, dtype=np.float32)

        assert_shape = (80, 192, len(SEG_CLASSES))
        assert segmentation.shape == assert_shape, "Incorrect shape ({}), got {}".format(assert_shape, segmentation.shape)
        assert len(image_coord_wp) == self.n_step, "Not enough points, got {}".format(image_coord_wp.shape)

        if self.batch_aug == 1:
            rgb_images = self.rgb_transform(rgb_images)
        else:
            rgb_images = torch.stack([self.rgb_transform(img) for img in rgb_images])

        segmentation = self.bird_view_transform(segmentation)

        self.batch_read_number += 1

        return rgb_images, segmentation, image_coord_wp, cmd, speed


def load_image_data(dataset_path,
        batch_size=32,
        num_workers=8,
        shuffle=True,
        n_step=5,
        gap=10,
        augment=None,
        **kwargs
    ):

    dataset = ImageDataset(
        dataset_path,
        n_step=n_step,
        gap=gap,
        augment_strategy=augment,
        **kwargs,
    )

    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, drop_last=True, pin_memory=True)


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


def get_image(
        dataset_dir,
        batch_size=32, num_workers=0, shuffle=True, augment=None,
        n_step=5, gap=5, batch_aug=1):

    def make_dataset(dir_name, is_train):
        _dataset_dir = str(Path(dataset_dir) / dir_name)
        _samples = 1000 if is_train else 10
        _num_workers = num_workers if is_train else 0
        _batch_aug = batch_aug if is_train else 1
        _augment = augment if is_train else None

        data = ImageDataset(
                _dataset_dir, gap=gap, n_step=n_step, augment_strategy=_augment, batch_aug=_batch_aug)
        data = Wrap(data, batch_size, _samples)
        data = _dataloader(data, batch_size, _num_workers)

        return data

    train = make_dataset('train', True)
    val = make_dataset('val', False)

    return train, val


if __name__ == '__main__':
    batch_size = 256
    import tqdm
    dataset = ImageDataset('/raid0/dian/carla_0.9.6_data/train')
    loader = _dataloader(dataset, batch_size=batch_size, num_workers=16)
    mean = []
    for rgb_img, bird_view, locations, cmd, speed in tqdm.tqdm(loader):
        mean.append(rgb_img.mean(dim=(0,2,3)).numpy())

    print ("Mean: ", np.mean(mean, axis=0))
    print ("Std: ", np.std(mean, axis=0)*np.sqrt(batch_size))
