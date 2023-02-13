from __future__ import division
from torch.utils.data import Dataset
import os
import numpy as np
import pickle
import imp
import trimesh
import torch
import json


def get_subject_names(data_source, split):
    subject_names = []
    for class_name in split:
        for instance_name in split[class_name]:
            if not os.path.exists(
                    os.path.join(data_source, instance_name)
            ):
                logging.warning(
                    "Requested non-existent directories '{}'".format(os.path.join(data_source, instance_name))
                )
            else:
                subject_names.append(instance_name)
    return subject_names



class VoxelizedDataset(Dataset):


    def __init__(self, mode, res = 32,  voxelized_pointcloud = False, pointcloud_samples = 3000, data_path = '/shared/storage/cs/staffstore/yg1390/HEADSPACE_PREPROCESSED/WoNeck/', split_file = '../headspace_sdf/splits/',
                 batch_size = 64, num_sample_points = 1024, num_workers = 12, sample_distribution = [1], sample_sigmas = [0.015], **kwargs):

        self.sample_distribution = np.array(sample_distribution)
        self.sample_sigmas = np.array(sample_sigmas)

        assert np.sum(self.sample_distribution) == 1
        assert np.any(self.sample_distribution < 0) == False
        assert len(self.sample_distribution) == len(self.sample_sigmas)

        self.path = data_path

        if mode == "val":
            with open(os.path.join(split_file, "headspace_train.json"), "r") as f:
                self.splits = json.load(f)
        else:
            with open(os.path.join(split_file, "headspace_{}.json".format(mode)), "r") as f:
                self.splits = json.load(f)

        self.data = get_subject_names(self.path, self.splits)

        # (training set + val set) : test set = 9:1, training set : val set = 9:1
        if mode == "train":
            self.data = self.data[:int(len(self.data)*0.9)]
        elif mode == "val":
            self.data = self.data[int(len(self.data)*0.9):]

        self.res = res

        self.num_sample_points = num_sample_points
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.voxelized_pointcloud = voxelized_pointcloud
        self.pointcloud_samples = pointcloud_samples

        # compute number of samples per sampling method
        self.num_samples = np.rint(self.sample_distribution * num_sample_points).astype(np.uint32)



    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path = self.path + self.data[idx]

        if not self.voxelized_pointcloud:
            occupancies = np.load(path + '/{}.voxelization_{}.npy'.format(self.data[idx], self.res))
            occupancies = np.unpackbits(occupancies)
            input = np.reshape(occupancies, (self.res,)*3)
        else:
            voxel_path = path + '/voxelized_point_cloud_{}res_{}points.npz'.format(self.res, self.pointcloud_samples)
            occupancies = np.unpackbits(np.load(voxel_path)['compressed_occupancies'])
            input = np.reshape(occupancies, (self.res,)*3)

        points = []
        coords = []
        occupancies = []

        for i, num in enumerate(self.num_samples):
            boundary_samples_path = path + '/boundary_{}_samples.npz'.format(self.sample_sigmas[i])
            boundary_samples_npz = np.load(boundary_samples_path)
            boundary_sample_points = boundary_samples_npz['points']
            boundary_sample_coords = boundary_samples_npz['grid_coords']
            boundary_sample_occupancies = boundary_samples_npz['occupancies']
            subsample_indices = np.random.randint(0, len(boundary_sample_points), num)
            points.extend(boundary_sample_points[subsample_indices])
            coords.extend(boundary_sample_coords[subsample_indices])
            occupancies.extend(boundary_sample_occupancies[subsample_indices])

        assert len(points) == self.num_sample_points
        assert len(occupancies) == self.num_sample_points
        assert len(coords) == self.num_sample_points

        return {'grid_coords':np.array(coords, dtype=np.float32),'occupancies': np.array(occupancies, dtype=np.float32),'points':np.array(points, dtype=np.float32), 'inputs': np.array(input, dtype=np.float32), 'path' : path}

    def get_loader(self, shuffle =True):

        return torch.utils.data.DataLoader(
                self, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=shuffle,
                worker_init_fn=self.worker_init_fn)

    def worker_init_fn(self, worker_id):
        random_data = os.urandom(4)
        base_seed = int.from_bytes(random_data, byteorder="big")
        np.random.seed(base_seed + worker_id)
