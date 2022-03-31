from glob import glob
from data_preprocess.utils import mkdir, write_json
import os
import shutil
import numpy as np


class VIPeR:
    def __init__(self, root, test_ratio=5):
        self.raw_dir = os.path.join(root, 'VIPeR')
        self.test_ratio = test_ratio

    def prepare(self):
        output_dir_base = os.path.join(self.raw_dir, "pytorch")
        mkdir(output_dir_base)
        output_train_base = os.path.join(output_dir_base, "train")
        mkdir(output_train_base)
        output_query_base = os.path.join(output_dir_base, "query")
        mkdir(output_query_base)
        output_gallery_base = os.path.join(output_dir_base, "gallery")
        mkdir(output_gallery_base)

        cameras = [sorted(glob(os.path.join(self.raw_dir, 'cam_a', '*.bmp'))),
                   sorted(glob(os.path.join(self.raw_dir, 'cam_b', '*.bmp')))]
        assert len(cameras[0]) == len(cameras[1])
        num_test = len(cameras[0]) // self.test_ratio
        identities = []
        for idx, (cam1, cam2) in enumerate(zip(*cameras)):
            images = []
            # <id>_<view>.bmp
            id1 = int(os.path.basename(cam1).split('_')[0])
            id2 = int(os.path.basename(cam2).split('_')[0])
            if id1 != id2:
                print("id not match")
                continue
            # 0000_c1.jpg
            cam1_save_name = '{:04d}_c{:01d}.jpg'.format(id1, 1)
            cam2_save_name = '{:04d}_c{:01d}.jpg'.format(id2, 2)
            # 先存20%的query和gallery作为test，剩下的作为train
            if idx > num_test:
                train_save_dir = os.path.join(output_train_base, "{:04d}".format(id1))
                mkdir(train_save_dir)
                shutil.copyfile(cam1, os.path.join(train_save_dir, cam1_save_name))
                shutil.copyfile(cam2, os.path.join(train_save_dir, cam2_save_name))
            else:
                query_save_dir = os.path.join(output_query_base, "{:04d}".format(id1))
                mkdir(query_save_dir)
                gallery_save_dir = os.path.join(output_gallery_base, "{:04d}".format(id1))
                mkdir(gallery_save_dir)
                shutil.copyfile(cam1, os.path.join(query_save_dir, cam1_save_name))
                shutil.copyfile(cam2, os.path.join(gallery_save_dir, cam2_save_name))
            images.append([cam1_save_name])
            images.append([cam2_save_name])
            identities.append(images)

        # Save meta information into a json file
        meta = {'name': 'VIPeR', 'shot': 'single', 'num_cameras': 2,
                'identities': identities}
        write_json(meta, os.path.join(self.raw_dir, 'meta.json'))


class CUHK01:
    def __init__(self, root, test_ratio=5):
        self.raw_dir = os.path.join(root, 'CUHK01')
        self.test_ratio = test_ratio

    def prepare(self):
        output_dir_base = os.path.join(self.raw_dir, "pytorch")
        mkdir(output_dir_base)
        output_train_base = os.path.join(output_dir_base, "train")
        mkdir(output_train_base)
        output_query_base = os.path.join(output_dir_base, "query")
        mkdir(output_query_base)
        output_gallery_base = os.path.join(output_dir_base, "gallery")
        mkdir(output_gallery_base)

        raw_images = sorted(glob(os.path.join(self.raw_dir, 'campus', '*.png')))
        os.listdir()
        num_test = len(raw_images) // self.test_ratio
        identities = []
        last_id = 'init'
        for image in raw_images:
            images = []
            # <id>_<view>.png
            # <:04d><:03d>.png
            id = os.path.basename(image)[:4]
            view = os.path.basename(image)[4:7]
            # 0000_c1.jpg
            save_name = '{:04d}_c{:03d}.jpg'.format(int(id), int(view))
            # 先存20%的query和gallery作为test，剩下的作为train
            if idx > num_test:
                train_save_dir = os.path.join(output_train_base, "{:04d}".format(id1))
                mkdir(train_save_dir)
                shutil.copyfile(cam1, os.path.join(train_save_dir, cam1_save_name))
                shutil.copyfile(cam2, os.path.join(train_save_dir, cam2_save_name))
            else:
                query_save_dir = os.path.join(output_query_base, "{:04d}".format(id1))
                mkdir(query_save_dir)
                gallery_save_dir = os.path.join(output_gallery_base, "{:04d}".format(id1))
                mkdir(gallery_save_dir)
                shutil.copyfile(cam1, os.path.join(query_save_dir, cam1_save_name))
                shutil.copyfile(cam2, os.path.join(gallery_save_dir, cam2_save_name))
            images.append([cam1_save_name])
            identities.append(images)

        # Save meta information into a json file
        meta = {'name': 'VIPeR', 'shot': 'single', 'num_cameras': 2,
                'identities': identities}
        write_json(meta, os.path.join(self.raw_dir, 'meta.json'))


data = CUHK01("./data")
data.prepare()
