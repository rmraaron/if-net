import os
import glob
import multiprocessing as mp
from multiprocessing import Pool
import trimesh
import traceback
import sys

INPUT_PATH = '/shared/storage/cs/staffstore/yg1390/HEADSPACE_PREPROCESSED/WoNeck/'

def to_off(path):

    for obj_file in os.listdir(path):
        if "obj" in obj_file:

            input_file  = path + obj_file
            output_file = path + '{}_scaled.off'.format(obj_file.split(".")[0])

            if os.path.exists(output_file):
                print('Exists: {}'.format(output_file))
                return


            try:
                with HiddenPrints():
                    mesh = trimesh.load(input_file)
                    total_size = (mesh.bounds[1] - mesh.bounds[0]).max()
                    centers = (mesh.bounds[1] + mesh.bounds[0]) / 2

                    mesh.apply_translation(-centers)
                    mesh.apply_scale(1 / total_size)
                    mesh.export(output_file)

                print('Finished: {}'.format(input_file))
            except:
                print('Error with {}: {}'.format(input_file, traceback.format_exc()))


#             cmd = 'meshlabserver -i {} -o {}'.format(input_file, output_file)
#             # if you run this script on a server: comment out above line and uncomment the next line
#             # cmd = 'xvfb-run -a -s "-screen 0 800x600x24" meshlabserver -i {} -o {}'.format(input_file,output_file)
#             os.system(cmd)

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

#
p = Pool(mp.cpu_count())
p.map(to_off, glob.glob( INPUT_PATH ))
#
# def scale(path):
#
#     for off_file in os.listdir(path):
#         if "off" in off_file:
#             if os.path.exists(path + '{}_scaled.off'.format(off_file.split(".")[0])):
#                 return
#
#             try:
#                 mesh = trimesh.load(path + '{}_scaled.off'.format(off_file.split(".")[0]), process=False)
#                 total_size = (mesh.bounds[1] - mesh.bounds[0]).max()
#                 centers = (mesh.bounds[1] + mesh.bounds[0]) /2
#
#                 mesh.apply_translation(-centers)
#                 mesh.apply_scale(1/total_size)
#                 mesh.export(path + '{}_scaled.off'.format(off_file.split(".")[0]))
#             except:
#                 print('Error with {}'.format(path))
#             print('Finished {}'.format(path))

# p = Pool(mp.cpu_count())
# p.map(scale, glob.glob(INPUT_PATH))