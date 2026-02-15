"""
Borrowed from 3DGS-DR
https://github.com/gapszju/3DGS-DR/blob/main/nero2blender.py
"""

import os
import numpy as np
import math
import json
import glob
import argparse
import pickle
import shutil
from tqdm.auto import tqdm
from skimage.io import imread, imsave

def read_pickle(pkl_path):
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default="data/nero/GlossySynthetic", help="path to the GlossyBlender dataset")
    parser.add_argument('--scene', type=str, default="all", help="scene name")
    parser.add_argument('--no_split', action="store_true", help="scene name")
    opt = parser.parse_args()

    scenes = [opt.scene] if opt.scene != "all" else [s for s in sorted(os.listdir(opt.path))
                                                     if not s.endswith("blender") and os.path.isdir(os.path.join(opt.path, s))]
    for scene in scenes:
        root = os.path.join(opt.path, scene)
        output_path = os.path.join(opt.path, scene+"_blender")
        if not os.path.exists(output_path):
            os.makedirs(output_path, exist_ok=True)

        print(f'[INFO] Data from {root}')

        img_num = len(glob.glob(f'{root}/*.pkl'))
        print(img_num)
        img_ids = [str(k) for k in range(img_num)]
        cams = [read_pickle(f'{root}/{k}-camera.pkl') for k in range(img_num)]  # pose(3,4)  K(3,3)
        img_files = [f'{root}/{k}.png' for k in range(img_num)]
        depth_files = [f'{root}/{k}-depth.png' for k in range(img_num)]
        points_file = os.path.join(root, "eval_pts.ply")

        if not opt.no_split:
            #test_ids, train_ids = read_pickle(os.path.join(opt.path, 'synthetic_split_128.pkl'))
            test_ids = [i for i in range(128) if i%8 == 0]
            train_ids = [i for i in range(128) if i%8 != 0]
        else:
            train_ids = []
            test_ids = [i for i in range(img_num)]


        # process 2 splits
        for split in ['train', 'test']:
            print(f'[INFO] Process transforms split = {split}')

            ids = test_ids if split == "test" else train_ids
            if not ids:
                continue

            split_imgs = [img_files[int(i)] for i in ids]
            split_cams = [cams[int(i)] for i in ids]

            # import pdb; pdb.set_trace()
            frames = []
            for image, cam in zip(split_imgs, split_cams):
                w2c = np.array(cam[0].tolist()+[[0,0,0,1]])
                c2w = np.linalg.inv(w2c)
                c2w[:3, 1:3] *= -1  # opencv -> blender/opengl
                frames.append({
                    'file_path': os.path.join("rgb", os.path.basename(image)).replace(".png",""),
                    'transform_matrix': c2w.tolist(),
                })

            fl_x = float(split_cams[0][1][0,0])
            fl_y = float(split_cams[0][1][1,1])

            transforms = {
                'w': 800,
                'h': 800,
                'fl_x': fl_x,
                'fl_y': fl_y,
                'cx': 400,
                'cy': 400,
                'camera_angle_x': 2*np.arctan(400/fl_x),
                # 'aabb_scale': 2,
                'frames': frames,
            }

            # write json
            json_out_path = os.path.join(output_path, f'transforms_{split}.json')
            print(f'[INFO] write to {json_out_path}')
            with open(json_out_path, 'w') as f:
                json.dump(transforms, f, indent=2)

        # write imgs
        img_out_path = os.path.join(output_path, "rgb")
        if not os.path.exists(img_out_path):
            os.makedirs(img_out_path, exist_ok=True)
        print(f'[INFO] Process rgbs')
        print(f'[INFO] write to {img_out_path}')

        for img_id in tqdm(img_ids, total=len(img_ids), desc=f"Converting {scene}"):
            if os.path.exists(f'{root}/{img_id}-depth.png'):
                depth = imread(f'{root}/{img_id}-depth.png')
                depth = depth.astype(np.float32) / 65535 * 15
            elif os.path.exists(f'{root}/{img_id}-depth0001.exr'):
                depth = imread(f'{root}/{img_id}-depth0001.exr')[..., 0]
                depth *= 15  # to be consistent with png depths
            else:
                raise ValueError("Depth not found")
            mask = depth < 14.5
            mask = (mask[...,None] * 255).astype(np.uint8)

            image = imread(f'{root}/{img_id}.png')[..., :3]
            image = np.concatenate([image, mask], axis=-1)

            save_p = f'{img_out_path}/{img_id}.png'
            # print(save_p)
            imsave(save_p, image)

        # copy ply
        if os.path.exists(points_file):
            points_out_path = os.path.join(output_path, "points.ply")
            shutil.copy2(points_file, points_out_path)

        print("[INFO] Finished.")