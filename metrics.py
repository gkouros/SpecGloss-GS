#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr


from pathlib import Path
import os
from PIL import Image
import torch
import cv2
import torchvision.transforms.functional as tf
# from utils.loss_utils import ssim
from fused_ssim import fused_ssim as ssim
from lpipsPyTorch import LPIPS
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser
import wandb
from scene.NVDIFFREC.util import load_image
import datetime


def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names


def append_to_table(table_name, new_data):
    # Try to retrieve the latest version of your table artifact
    try:
        artifact = wandb.run.use_artifact(f"run-{wandb.run.id}-{table_name}:latest")
        artifact_dir = artifact.download(root='/tmp')
        with open(f"{artifact_dir}/Metrics.table.json", "r") as f:
            old_data = json.load(f)['data']
    except:
        old_data = []
    # new_data["label"] += datetime.datetime.now().strftime("_%Y%m%d-%H:%M:%S")
    old_data = [row for row in old_data if row[3] != new_data["label"]]
    table = wandb.Table(
        data=old_data + [list(new_data.values())],
        columns=["psnr", "ssim", "lpips", "label"])
    wandb.log({"Metrics": table})  # Log the updated table to W&B


def evaluate(model_paths, args):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}

    for scene_dir in model_paths:
        # try:
        if True:
            if args.wandb:
                exp_name, dataset_name, scene_name = scene_dir.split('/')[-3:]
                wandb.init(
                    project="refgshader",  # set the wandb project where this run will be logged
                    config=vars(args),  # track hyperparameters and run metadata
                    group=dataset_name,
                    name=f"{scene_name}.{exp_name}",
                    job_type=exp_name.split('.')[0],
                    id=args.run_id,
                    resume="allow",
                )

            print("Scene:", scene_dir)
            full_dict[scene_dir] = {}
            per_view_dict[scene_dir] = {}
            full_dict_polytopeonly[scene_dir] = {}
            per_view_dict_polytopeonly[scene_dir] = {}

            test_dir = Path(scene_dir) / args.test_dir

            lpips = LPIPS(net_type='vgg').to(torch.device("cuda")).eval()

            for method in sorted(os.listdir(test_dir)):
                print("Method:", method)
                iteration = int(method.split("_")[-1])  # e.g. ours_30000

                full_dict[scene_dir][method] = {}
                per_view_dict[scene_dir][method] = {}
                full_dict_polytopeonly[scene_dir][method] = {}
                per_view_dict_polytopeonly[scene_dir][method] = {}

                method_dir = test_dir / method
                gt_dir = method_dir/ "gt"
                renders_dir = method_dir / "renders"
                try:
                    renders, gts, image_names = readImages(renders_dir, gt_dir)
                except FileNotFoundError as err:
                    print(f'Directory {renders_dir} does not contain anything. Skipping method {method}.')
                    continue

                ssims = []
                psnrs = []
                lpipss = []

                for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                    with torch.no_grad():
                        ssims.append(ssim(renders[idx], gts[idx], train=False))
                    psnrs.append(psnr(renders[idx], gts[idx]))
                    if not args.debug:
                        lpipss.append(lpips(renders[idx], gts[idx]))
                    else:
                        lpipss.append(torch.tensor(0, dtype=torch.float32))

                psnrs = torch.tensor(psnrs)
                ssims = torch.tensor(ssims)
                lpipss = torch.tensor(lpipss)

                prefix = f"{args.test_dir}/" if args.test_dir != "test" else ""

                print("  {}SSIM : {:>12.7f}".format(prefix, ssims.mean().item(), ".5"))
                print("  {}PSNR : {:>12.7f}".format(prefix, psnrs.mean().item(), ".5"))
                print("  {}LPIPS: {:>12.7f}".format(prefix, lpipss.mean().item(), ".5"))
                print("")

                full_dict[scene_dir][method].update({
                    f"{prefix}SSIM": ssims.mean().item(),
                    f"{prefix}PSNR": psnrs.mean().item(),
                    f"{prefix}LPIPS": lpipss.mean().item()
                })
                per_view_dict[scene_dir][method].update({
                    f"{prefix}SSIM": {name: ssim for ssim, name in zip(ssims.tolist(), image_names)},
                    f"{prefix}PSNR": {name: psnr for psnr, name in zip(psnrs.tolist(), image_names)},
                    f"{prefix}LPIPS": {name: lp for lp, name in zip(lpipss.tolist(), image_names)},
                })

                if wandb.run:

                    # log metrics individually
                    wandb.log(full_dict[scene_dir][method])

                    # append metrics to metrics table to log
                    metrics_to_append = dict(zip(
                        ["psnr", "ssim", "lpips", "label"],
                        [psnrs.mean().item(), ssims.mean().item(), lpipss.mean().item(), os.path.join(args.test_dir, method)]
                    ))
                    # log metrics to a table
                    append_to_table("Metrics", metrics_to_append)

                    # log only relighted images here
                    if args.test_dir != "test":
                        wandb.log({f"{args.test_dir}/images": [
                            wandb.Image(gts[0].squeeze().permute(1, 2, 0).cpu().numpy(), caption="GT"),
                            wandb.Image(renders[0].squeeze().permute(1, 2, 0).cpu().numpy(), caption="Render"),
                        ]}, step=iteration)

                with open(os.path.join(scene_dir, args.test_dir, method, "results.json"), 'w') as fp:
                    json.dump(full_dict[scene_dir][method], fp, indent=True)
                with open(os.path.join(scene_dir, args.test_dir, method, "per_view.json"), 'w') as fp:
                    json.dump(per_view_dict[scene_dir][method], fp, indent=True)

            # if wandb.run:
            #     envmaps_to_log = []
            #     for envmap_name in ["gt_envmap", "estimated_envmap", "gt_envmap_tonemapped", "relight_envmap", "relight_envmap_tonemapped"]:
            #         if os.path.exists(os.path.join(scene_dir, args.test_dir, method, 'envmaps', f"{envmap_name}.png")):
            #             envmap = load_image(os.path.join(scene_dir, args.test_dir, method, 'envmaps', f"{envmap_name}.png"))
            #             envmap = cv2.resize(envmap, (1024, 512))
            #             envmaps_to_log.append(wandb.Image(envmap, caption=envmap_name))
            #     wandb.log({f"{args.test_dir}/envmap": envmaps_to_log})
        # except Exception as err:
        #     print(f"Unable to compute metrics for model {scene_dir} with error: {err}")


if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    parser.add_argument('--debug', help="Disable lpips computation for faster evaluation", action='store_true')
    parser.add_argument('--wandb', help="Log metrics to wandb", action='store_true')
    parser.add_argument('--test_dir', default="test", help="The directory that contains the exported images. Default is 'test'.")
    parser.add_argument('--run_id', default="", type=str)
    args = parser.parse_args()

    if args.wandb:
        for id_name in ["wandb_run_id" , "wandb_id"]:
            run_id_path = os.path.join(args.model_paths[0], f"{id_name}.txt")
            if os.path.exists(run_id_path):
                with open(run_id_path, 'r') as f:
                    args.run_id = f.read().strip()
                print(f"File {id_name}.txt found. Logging metrics in session with id {args.run_id}.")
                break
        else:
            print(f"No {id_name}.txt file found. Logging metrics in session with id {args.run_id}.")

    evaluate(args.model_paths, args)
