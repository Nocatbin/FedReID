from glob import glob
# from scipy.misc import imsave, imread
from data_preprocess.utils import mkdir
import os


class VIPeR:
    def __init__(self, root):
        self.raw_dir = os.path.join(root, 'VIPeR')

    def prepare(self):
        output_dir = os.path.join(self.raw_dir, "pytorch")
        mkdir(output_dir)

        cameras = [sorted(glob(os.path.join(self.raw_dir, 'cam_a', '*.bmp'))),
                   sorted(glob(os.path.join(self.raw_dir, 'cam_b', '*.bmp')))]
        assert len(cameras[0]) == len(cameras[1])
        identities = []
        for pid, (cam1, cam2) in enumerate(zip(*cameras)):
            images = []
            # view-0
            fname = '{:08d}_{:02d}_{:04d}.jpg'.format(pid, 0, 0)
            imsave(os.path.join(images_dir, fname), imread(cam1))
            images.append([fname])
            # view-1
            fname = '{:08d}_{:02d}_{:04d}.jpg'.format(pid, 1, 0)
            imsave(os.path.join(images_dir, fname), imread(cam2))
            images.append([fname])
            identities.append(images)

        # Save meta information into a json file
        meta = {'name': 'VIPeR', 'shot': 'single', 'num_cameras': 2,
                'identities': identities}
        write_json(meta, os.path.join(self.root, 'meta.json'))

        # Randomly create ten training and test split
        num = len(identities)
        splits = []
        for _ in range(10):
            pids = np.random.permutation(num).tolist()
            trainval_pids = sorted(pids[:num // 2])
            test_pids = sorted(pids[num // 2:])
            split = {'trainval': trainval_pids,
                     'query': test_pids,
                     'gallery': test_pids}
            splits.append(split)
        write_json(splits, os.path.join(self.root, 'splits.json'))


data = VIPeR("./data_preprocess/data")
data.prepare()
