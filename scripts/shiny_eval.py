# training scripts for the shiny blender datasets
# this script is adopted from GOF
# https://github.com/autonomousvision/gaussian-opacity-fields/blob/main/scripts/run_nerf_synthetic.py
import os
import GPUtil
from concurrent.futures import ThreadPoolExecutor
import time
import itertools
import argparse
import uuid
import sys
import re
import socket

nodename = socket.gethostname()

# no caching to avoid raceconditions with jobs running on different architectures
os.environ["CUDA_CACHE_DISABLE"] = "1"
if "esat" in nodename:
    os.environ["TORCH_CUDA_ARCH_LIST"] = "6.0 7.0 7.5 8.0 8.6+PTX"
    os.environ["CUDA_HOME"] = os.environ["CONDA_PREFIX"]
    os.environ["PATH"] = f"{os.environ['CUDA_HOME']}/bin:{os.environ['CONDA_PREFIX']}/bin:{os.environ['PATH']}"
    os.environ["CPATH"] = f"{os.environ['CONDA_PREFIX']}/targets/x86_64-linux/include"
    os.environ["LD_LIBRARY_PATH"] = f"{os.environ['CONDA_PREFIX']}/lib64/:{os.environ['CONDA_PREFIX']}/bin/:{os.environ['CONDA_PREFIX']}/lib/:{os.environ['CONDA_PREFIX']}/lib/stub"

parser = argparse.ArgumentParser(description="Training script parameters")
parser.add_argument('out', help='Input experiment name')
parser.add_argument('--dry_run', help='Just print commands without executing them', action='store_true')
parser.add_argument('--scene', help='Select which scene to learn', default='all')
parser.add_argument('--skip_train', help='Skip training step', action='store_true')
parser.add_argument('--skip_render', help='Skip rendering step', action='store_true')
parser.add_argument('--skip_metrics', help='Skip rendering step', action='store_true')
parser.add_argument('--skip_relight', help='Skip relighting step', action='store_true')
parser.add_argument('--extra_relight', help='Skip relighting step', action='store_true')
parser.add_argument('--skip_relight_metrics', help='Skip relighting metrics step', action='store_true')
parser.add_argument("extra_args", nargs=argparse.REMAINDER, help="Arguments to forward to the script.")

args = parser.parse_args()

extra_args = ' ' + ' '.join(args.extra_args)
exp_name = args.out + extra_args.replace(" --wandb","").replace(" --gui","").replace(" --",".").replace(" ",".").replace("=","")
print("Experiment: ", exp_name, end="\n\n")
print('args: ', args, end="\n\n")
print('extra_args: ', args.extra_args, end="\n\n")

# get number of iterations if specified
match = re.search(r'--iterations=(\d+)', ''.join(extra_args))
iters = int(match.group(1)) if match else 30000

dataset = "ref_shiny"
scenes = ["ball", "car", "coffee", "helmet", "teapot", "toaster"]
if args.scene != 'all':
    if args.scene in scenes:
        scenes = [args.scene]
    else:
        raise RuntimeError(f'Scene {args.scene} does not exist in shiny-blender dataset')

output_dir = f"logs/{exp_name}/{dataset}"
dataset_dir = f"data/{dataset}"

gt_envmaps_dict = {
    "ball": "forest.exr",
    "car": "forest.exr",
    "coffee": "rj1.jpg",
    "helmet": "abandoned_factory_canteen_01_4k.exr",
    "teapot": "sunset_jhbcentral_4k.exr",
    "toaster": "interior.exr",
}
relight_envmap = "christmas_photo_studio_04_4k.exr"

envmaps_path = "data/envmaps/high_res_envmaps_1k"
envmaps = ["bridge.hdr", "city.hdr", "courtyard.hdr", "fireplace.hdr"]
        #    "forest.hdr", "interior.hdr", "museum.hdr", "night.hdr",
        #    "snow.hdr", "square.hdr", "studio.hdr", "sunrise.hdr", "sunset.hdr", "tunnel.hdr"]


factors = [1]
excluded_gpus = set([])
jobs = list(itertools.product(scenes, factors))


def run_script(command):
    print(command)
    if args.dry_run:
        return
    exit_code = os.system(command)
    if exit_code > 0:
        print("Error %d while running %s'" % (exit_code, command), file=sys.stderr)
        sys.exit(exit_code)

def train_scene(gpu, scene, factor):
    run_id = uuid.uuid4().hex  # Generates a random hexadecimal id for wandb

    if not args.skip_train:
        cmd = f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python train.py -s {dataset_dir}/{scene} -m {output_dir}/{scene} --run_id {run_id} --eval --white_background {extra_args}"
        run_script(cmd)

    if not args.skip_render:
        cmd = f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python render.py -m {output_dir}/{scene} --skip_train --skip_mesh"
        run_script(cmd)

    if not args.skip_metrics:
        cmd = f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python metrics.py -m {output_dir}/{scene} --run_id {run_id} {'--wandb' * ('wandb' in extra_args)}"
        run_script(cmd)

    if not args.skip_relight:
        cmd = f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python render.py -m {output_dir}/{scene} --skip_train --skip_test --skip_mesh"
        cmd += f" --relight_gt_path {dataset_dir}_relighting/{scene}_christmas"
        cmd += f" --relight_envmap_path {dataset_dir}_relighting/envmaps/{relight_envmap}"
        cmd += f" --gt_envmap_path {dataset_dir}_relighting/envmaps/{gt_envmaps_dict[scene]}"
        run_script(cmd)

        if not args.skip_relight_metrics:
            cmd = f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python metrics.py -m {output_dir}/{scene} --run_id {run_id} {'--wandb' * ('wandb' in extra_args)}"
            cmd += f" --test_dir=relight/{relight_envmap}"
            run_script(cmd)

    if args.extra_relight:
        for envmap_name in envmaps:
            cmd = f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python render.py -m {output_dir}/{scene} --skip_train --skip_test --skip_mesh"
            cmd += f" --relight_gt_path {dataset_dir}/{scene}"
            cmd += f" --relight_envmap_path {envmaps_path}/{envmap_name}"
            run_script(cmd)

    return True


def worker(gpu, scene, factor):
    print(f"Starting job on GPU {gpu} with scene {scene}\n")
    train_scene(gpu, scene, factor)
    print(f"Finished job on GPU {gpu} with scene {scene}\n")
    # This worker function starts a job and returns when it's done.


def dispatch_jobs(jobs, executor):
    future_to_job = {}
    reserved_gpus = set()  # GPUs that are slated for work but may not be active yet

    while jobs or future_to_job:
        # Get the list of available GPUs, not including those that are reserved.
        all_available_gpus = set(GPUtil.getAvailable(order="first", limit=10, maxMemory=0.5, maxLoad=0.5))
        available_gpus = list(all_available_gpus - reserved_gpus - excluded_gpus)

        # Launch new jobs on available GPUs
        while available_gpus and jobs:
            gpu = available_gpus.pop(0)
            job = jobs.pop(0)
            future = executor.submit(worker, gpu, *job)  # Unpacking job as arguments to worker
            future_to_job[future] = (gpu, job)

            reserved_gpus.add(gpu)  # Reserve this GPU until the job starts processing

        # Check for completed jobs and remove them from the list of running jobs.
        # Also, release the GPUs they were using.
        done_futures = [future for future in future_to_job if future.done()]
        for future in done_futures:
            job = future_to_job.pop(future)  # Remove the job associated with the completed future
            gpu = job[0]  # The GPU is the first element in each job tuple
            reserved_gpus.discard(gpu)  # Release this GPU
            print(f"Job {job} has finished., releasing GPU {gpu}")
        # (Optional) You might want to introduce a small delay here to prevent this loop from spinning very fast
        # when there are no GPUs available.
        time.sleep(5)

    print("All jobs have been processed.")


# Using ThreadPoolExecutor to manage the thread pool
with ThreadPoolExecutor(max_workers=8) as executor:
    dispatch_jobs(jobs, executor)
