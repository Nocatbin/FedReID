from argparse import ArgumentParser
from glob import glob
from data_preprocess.utils import mkdir, write_json
import os
import shutil


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
        num_test = len(cameras[0]) - (len(cameras[0]) // self.test_ratio)
        identities = []
        last_id = -999
        id = 0
        for idx, (cam1, cam2) in enumerate(zip(*cameras)):
            images = []
            # <id>_<view>.bmp
            id1 = int(os.path.basename(cam1).split('_')[0])
            id2 = int(os.path.basename(cam2).split('_')[0])
            if id1 != id2:
                print("id not match")
                continue
            if last_id != id1:
                id += 1
                last_id = id1
            # 0000_c1.jpg
            cam1_save_name = '{:04d}_{:03d}_{:03d}.jpg'.format(id, 1, 0)
            cam2_save_name = '{:04d}_{:03d}_{:03d}.jpg'.format(id, 2, 0)
            if idx < num_test:
                train_save_dir = os.path.join(output_train_base, "{:04d}".format(id))
                mkdir(train_save_dir)
                shutil.copyfile(cam1, os.path.join(train_save_dir, cam1_save_name))
                shutil.copyfile(cam2, os.path.join(train_save_dir, cam2_save_name))
            else:
                query_save_dir = os.path.join(output_query_base, "{:04d}".format(id))
                mkdir(query_save_dir)
                gallery_save_dir = os.path.join(output_gallery_base, "{:04d}".format(id))
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
        num_test = len(raw_images) - (len(raw_images) // self.test_ratio)
        identities = []
        last_id = -999
        idx = 0
        id = -999
        for image in raw_images:
            images = []
            # <id>_<view>.png
            # <:04d><:03d>.png
            id_tmp = int(os.path.basename(image)[:4])
            view = int(os.path.basename(image)[4:7])
            # 0000_c1.jpg
            if id_tmp != last_id:
                id = id_tmp
            save_name = '{:04d}_{:03d}_{:03d}.jpg'.format(id, 1, view)
            if idx < num_test:
                train_save_dir = os.path.join(output_train_base, "{:04d}".format(id))
                mkdir(train_save_dir)
                shutil.copyfile(image, os.path.join(train_save_dir, save_name))
            else:
                if id != last_id:
                    # 每个id 第一张图作为query
                    query_save_dir = os.path.join(output_query_base, "{:04d}".format(id))
                    mkdir(query_save_dir)
                    shutil.copyfile(image, os.path.join(query_save_dir, save_name))
                else:
                    # 剩下的都作为gallery
                    gallery_save_dir = os.path.join(output_gallery_base, "{:04d}".format(id))
                    mkdir(gallery_save_dir)
                    shutil.copyfile(image, os.path.join(gallery_save_dir, save_name))
                last_id = id_tmp
            images.append(image)
            identities.append(images)
            idx += 1


def get_id_camera_frame(src_name, dataset):
    save_name = src_name
    if dataset == 'MSMT17':
        # ['0000', 'c14', '0030.jpg']
        group = src_name.split('_')
        id = int(group[0])
        camera = int(group[1][1:])
        frame = int(group[2][:-4])
    elif dataset == "Market":
        # 0001_c1s1_001051_00.jpg
        group = src_name.split('_')
        id = int(group[0])
        camera = int(group[1][1:2])
        frame = int(group[2])
    elif dataset == "DukeMTMC-reID":
        # 0001_c2_f0046182.jpg
        group = src_name.split('_')
        id = int(group[0])
        camera = int(group[1][1:])
        frame = int(group[2][1:-4])
    return id, camera, frame


def get_save_name(src_name, dataset):
    save_name = src_name
    if dataset == 'MSMT17':
        # ['0000', 'c14', '0030.jpg']
        group = src_name.split('_')
        save_name = '{:04d}_{:03d}_{:d}.jpg'.format(int(group[0]), int(group[1][1:]), int(group[2][:-4]))
    elif dataset == "Market":
        # 0001_c1s1_001051_00.jpg
        group = src_name.split('_')
        save_name = '{:04d}_{:03d}_{:d}.jpg'.format(int(group[0]), int(group[1][1:2]), int(group[2]))
    elif dataset == "DukeMTMC-reID":
        # 0001_c2_f0046182.jpg
        group = src_name.split('_')
        save_name = '{:04d}_{:03d}_{:d}.jpg'.format(int(group[0]), int(group[1][1:]), int(group[2][1:-4]))
    return save_name


def prepare_all_big_datasets():
    # datasets=['Market','MSMT17','DukeMTMC-reID','cuhk03-np-detected']
    datasets = ['DukeMTMC-reID']
    for i in datasets:
        data_dir = os.path.abspath('./data/' + i)
        save_path = os.path.join(data_dir, 'pytorch')
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        # -----------------------------------------
        # query
        query_path = os.path.join(data_dir, 'query')
        query_save_path = os.path.join(data_dir, 'pytorch/query')
        if not os.path.isdir(query_save_path):
            os.mkdir(query_save_path)

        for root, dirs, files in os.walk(query_path, topdown=True):
            for name in files:
                if not name[-3:] == 'jpg' and not name[-3:] == 'png':
                    continue
                id = name.split('_')
                src_path = query_path + '/' + name
                dst_path = query_save_path + '/' + id[0]
                if not os.path.isdir(dst_path):
                    os.mkdir(dst_path)
                name = get_save_name(name, i)
                shutil.copyfile(src_path, dst_path + '/' + name)

        # -----------------------------------------
        # multi-query gt_bbox as query
        query_path = os.path.join(data_dir, 'gt_bbox')
        # for dukemtmc-reid, we do not need multi-query, does not have gt_bbox
        if os.path.isdir(query_path):
            query_save_path = os.path.join(data_dir, 'pytorch/multi-query')
            if not os.path.isdir(query_save_path):
                os.mkdir(query_save_path)

            for root, dirs, files in os.walk(query_path, topdown=True):
                for name in files:
                    if not name[-3:] == 'jpg' and not name[-3:] == 'png':
                        continue
                    id = name.split('_')
                    src_path = query_path + '/' + name
                    dst_path = query_save_path + '/' + id[0]
                    if not os.path.isdir(dst_path):
                        os.mkdir(dst_path)
                    name = get_save_name(name, i)
                    shutil.copyfile(src_path, dst_path + '/' + name)

        # -----------------------------------------
        # gallery / training set
        gallery_path = os.path.join(data_dir, 'bounding_box_test')
        gallery_save_path = os.path.join(data_dir, 'pytorch/gallery')
        if not os.path.isdir(gallery_save_path):
            os.mkdir(gallery_save_path)

        for root, dirs, files in os.walk(gallery_path, topdown=True):
            for name in files:
                if not name[-3:] == 'jpg' and not name[-3:] == 'png':
                    continue
                id = name.split('_')
                src_path = gallery_path + '/' + name
                dst_path = gallery_save_path + '/' + id[0]
                if not os.path.isdir(dst_path):
                    os.mkdir(dst_path)
                name = get_save_name(name, i)
                shutil.copyfile(src_path, dst_path + '/' + name)

        # ---------------------------------------
        # train_all
        train_path = data_dir + '/bounding_box_train'
        train_save_path = data_dir + '/pytorch/train_all'
        if not os.path.isdir(train_save_path):
            os.mkdir(train_save_path)

        for root, dirs, files in os.walk(train_path, topdown=True):
            for name in files:
                if not name[-3:] == 'jpg' and not name[-3:] == 'png':
                    continue
                id = name.split('_')
                src_path = train_path + '/' + name
                dst_path = train_save_path + '/' + id[0]
                if not os.path.isdir(dst_path):
                    os.mkdir(dst_path)
                name = get_save_name(name, i)
                shutil.copyfile(src_path, dst_path + '/' + name)

        # ---------------------------------------
        # train_val
        last_id = -999
        id = 0
        train_path = data_dir + '/bounding_box_train'
        train_save_path = data_dir + '/pytorch/train'
        val_save_path = data_dir + '/pytorch/val'
        if not os.path.isdir(train_save_path):
            os.mkdir(train_save_path)
            os.mkdir(val_save_path)

        for root, dirs, files in os.walk(train_path, topdown=True):
            for name in files:
                if not name[-3:] == 'jpg' and not name[-3:] == 'png':
                    continue
                id_temp, camera, frame = get_id_camera_frame(name, i)
                if last_id != id_temp:
                    id += 1
                    last_id = id_temp
                src_path = train_path + '/' + name
                dst_path = train_save_path + '/' + '{:04d}'.format(id)
                if not os.path.isdir(dst_path):
                    os.mkdir(dst_path)
                    # first image of each person(id) is used as val image
                    dst_path = val_save_path + '/' + '{:04d}'.format(id)
                    os.mkdir(dst_path)
                save_name = '{:04d}_{:03d}_{:d}.jpg'.format(id, camera, frame)
                if os.path.isfile(dst_path + '/' + save_name):
                    save_name = '{:04d}_{:03d}_{:d}.jpg'.format(id, camera, frame + 1)
                shutil.copyfile(src_path, dst_path + '/' + save_name)


if __name__ == '__main__':
    parser = ArgumentParser(
        description="Create lists of image file and label")
    parser.add_argument(
        '--dataset_dir', type=str, default=' ',
        help="Directory of a formatted dataset")
    parser.add_argument(
        '--output_dir', type=str, default=' ',
        help="Output directory for the lists")
    parser.add_argument(
        '--val-ratio', type=float, default=0.2,
        help="Ratio between validation and trainval data. Default 0.2.")
    args = parser.parse_args()
    print(args.dataset_dir)

    # prepare_all_big_datasets()
    # '3dpes', 'cuhk01', 'cuhk02', 'ilids', 'prid', 'viper'
    for dataset in ['cuhk01']:
        if dataset == 'viper' or dataset == 'VIPeR':
            data = VIPeR("./data")
            data.prepare()
        elif dataset == 'cuhk01' or dataset == 'CUHK01':
            data = CUHK01("./data")
            data.prepare()
