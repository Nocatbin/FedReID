import os
import argparse
from shutil import copyfile


def prepare_all_big_datasets():
    # datasets=['Market','MSMT17','DukeMTMC-reID','cuhk03-np-detected']
    datasets = ['Market']
    for i in datasets:
        data_dir = os.path.abspath('C:/Users/15005/Desktop/FedReID/data_preprocess/data/' + i)

        # You only need to change this line to your dataset download path
        download_path = data_dir

        if not os.path.isdir(download_path):
            print('please change the download_path')

        save_path = os.path.join(download_path, 'pytorch')
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        # -----------------------------------------
        # query
        query_path = os.path.join(download_path, 'query')
        query_save_path = os.path.join(download_path, 'pytorch/query')
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
                copyfile(src_path, dst_path + '/' + name)

        # -----------------------------------------
        # multi-query gt_bbox as query
        query_path = os.path.join(download_path, 'gt_bbox')
        # for dukemtmc-reid, we do not need multi-query, does not have gt_bbox
        if os.path.isdir(query_path):
            query_save_path = os.path.join(download_path, 'pytorch/multi-query')
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
                    copyfile(src_path, dst_path + '/' + name)

        # -----------------------------------------
        # gallery / training set
        gallery_path = os.path.join(download_path, 'bounding_box_test')
        gallery_save_path = os.path.join(download_path, 'pytorch/gallery')
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
                copyfile(src_path, dst_path + '/' + name)

        # ---------------------------------------
        # train_all
        train_path = download_path + '/bounding_box_train'
        train_save_path = download_path + '/pytorch/train_all'
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
                copyfile(src_path, dst_path + '/' + name)

        # ---------------------------------------
        # train_val
        train_path = download_path + '/bounding_box_train'
        train_save_path = download_path + '/pytorch/train'
        val_save_path = download_path + '/pytorch/val'
        if not os.path.isdir(train_save_path):
            os.mkdir(train_save_path)
            os.mkdir(val_save_path)

        for root, dirs, files in os.walk(train_path, topdown=True):
            for name in files:
                if not name[-3:] == 'jpg' and not name[-3:] == 'png':
                    continue
                id = name.split('_')
                src_path = train_path + '/' + name
                dst_path = train_save_path + '/' + id[0]
                if not os.path.isdir(dst_path):
                    os.mkdir(dst_path)
                    # first image of each person(id) is used as val image
                    dst_path = val_save_path + '/' + id[0]
                    os.mkdir(dst_path)
                copyfile(src_path, dst_path + '/' + name)
