import trimesh
import numpy as np
import implicit_waterproofing as iw
import glob
import multiprocessing as mp
from multiprocessing import Pool
import argparse
import os
import traceback

ROOT = '/shared/storage/cs/staffstore/yg1390/HEADSPACE_PREPROCESSED/WoNeck/'


def boundary_sampling(path):
    for off_file in os.listdir(path):
        if ".off" in off_file:
            try:

                if os.path.exists(os.path.join(path, off_file.split("_")[0], '/boundary_{}_samples.npz'.format(args.sigma))):
                    return

                off_path = path + off_file
                out_file = path + off_file.split("_")[0] + '/boundary_{}_samples.npz'.format(args.sigma)

                mesh = trimesh.load(off_path)
                points = mesh.sample(sample_num)

                boundary_points = points + args.sigma * np.random.randn(sample_num, 3)
                grid_coords = boundary_points.copy()
                grid_coords[:, 0], grid_coords[:, 2] = boundary_points[:, 2], boundary_points[:, 0]

                grid_coords = 2 * grid_coords

                occupancies = iw.implicit_waterproofing(mesh, boundary_points)[0]

                np.savez(out_file, points=boundary_points, occupancies = occupancies, grid_coords= grid_coords)
                print('Finished {}'.format(off_path))
            except:
                print('Error with {}: {}'.format(path, traceback.format_exc()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run boundary sampling'
    )
    parser.add_argument('-sigma', default=0.00025, type=float)

    args = parser.parse_args()


    sample_num = 100000


    p = Pool(mp.cpu_count())
    p.map(boundary_sampling, glob.glob( ROOT))
