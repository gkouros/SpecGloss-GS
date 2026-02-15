import sys
from scene import Scene, GaussianModel
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import render
from gaussian_renderer import network_gui
from utils.image_utils import render_net_image
from utils.general_utils import colormap, tensor_to_viz
from utils.image_utils import pca_transform
import torch
import viser
import numpy as np
from PIL import Image
from scene.cameras import Camera
from scene.NVDIFFREC.light import extract_env_map
from utils.camera_utils import get_closest_camera
from utils.general_utils import inverse_sigmoid


def view(dataset, pipe, args, server):

    gaussians = GaussianModel(dataset)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False, relight=args.relight_envmap_path, no_cameras=args.relight_envmap_path)
    if dataset.k_dim > 0 or dataset.sh_degree > 0:
        gaussians.activate_residual()

    if args.relight_envmap_path:
        gaussians.load_env_map(args.relight_envmap_path, tonemap=lambda x: np.roll(x, shift=x.shape[1]//4, axis=1))

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    if not args.relight_envmap_path:
        cameras = scene.getTestCameras()
        FoVx, FoVy = cameras[0].FoVx, cameras[0].FoVy
        image = cameras[0].original_image[:3]  # (3,H,W)
        gt_alpha_mask = cameras[0].gt_alpha_mask
        HWK = cameras[0].HWK
    else:
        FoVx, FoVy = 0.5, 0.5
        image=torch.zeros((3, 800, 800)).cuda()
        gt_alpha_mask = None
        HWK = None

    prev_camera = None
    while True:

        if server is None or server["client"] is None:
            continue

        render_type = network_gui.on_gui_change()
        with torch.no_grad():
            client = server["client"]
            RT_w2v = viser.transforms.SE3(wxyz_xyz=np.concatenate([client.camera.wxyz, client.camera.position], axis=-1)).inverse()
            R = RT_w2v.rotation().as_matrix().astype(np.float32)
            R = R.T if "glossy_synthetic" in  dataset.source_path else R  # fix visualization control for glossy_synthetic dataset
            T = RT_w2v.translation().astype(np.float32)
            camera = Camera(
                uid=None, colmap_id=None,
                R=R, T=T, HWK=HWK,
                FoVx=FoVx, FoVy=FoVy,
                image=image, gt_alpha_mask=gt_alpha_mask, image_name="",
            )

            render_pkg = render(camera, gaussians, pipe, background)
            rendering = render_pkg["render"]

            output = None
            if render_type == "Rendered":
                output = tensor_to_viz(rendering.clamp(0.0, 1.0))
            elif render_type in ["gt image", "stable normals", "stable delight"]:
                # find closest training view to the current camera
                if prev_camera is None or (camera.R != prev_camera.R).any() or (camera.T != prev_camera.T).any():
                    closest_camera = get_closest_camera(camera, scene.getTestCameras())
                else:
                    prev_camera = camera
                if render_type == "gt image":
                    image = closest_camera.original_image[:3]  # (3,H,W)
                elif render_type == "stable delight":
                    image = (closest_camera.delight_prior[:3] if closest_camera.delight_prior is not None else torch.zeros_like(rendering))
                elif render_type == "stable normals":
                    image = (((closest_camera.normal_prior[:3] + 1) / 2.0) if closest_camera.normal_prior is not None else torch.zeros_like(rendering))
                output = tensor_to_viz(image)
            elif render_type == "render normal":
                output = tensor_to_viz((render_pkg["rend_normal"] + 1) / 2.0)
            elif render_type == "surf normal":
                output = tensor_to_viz((render_pkg["surf_normal"] + 1) / 2.0)
            elif render_type == "surf depth":
                max_depth = 10.0
                rend_depth = torch.clamp(render_pkg["surf_depth"], 0.0, max_depth)
                rendered_image = colormap((rend_depth / max_depth).cpu().numpy()[0], cmap='turbo', bar=False)
                output = tensor_to_viz(torch.tensor(rendered_image))
            elif render_type == "distortion":
                output = tensor_to_viz((render_pkg["rend_dist"] * 1000000).repeat(3,1,1))
            elif render_type == "base color":
                output = tensor_to_viz(render_pkg["rend_base_color"])
            elif render_type == "roughness":
                output = tensor_to_viz(render_pkg["rend_roughness"].repeat(3,1,1))
            elif render_type == "visibility":
                output = tensor_to_viz(render_pkg["rend_visibility"].repeat(3,1,1))
            elif render_type == "metallic":
                output = tensor_to_viz(render_pkg["rend_metallic"].repeat(3,1,1))
            elif render_type == "specular":
                output = tensor_to_viz(render_pkg["rend_specular"])
            elif render_type == "diffuse color":
                output = tensor_to_viz(render_pkg["rend_diffuse_color"])
            elif render_type == "specular color":
                output = tensor_to_viz(render_pkg["rend_specular_color"])
            elif render_type == "residual color":
                output = tensor_to_viz(render_pkg["rend_residual_color"])
            elif render_type == "direct light":
                output = tensor_to_viz(render_pkg["rend_direct_light"])
            elif render_type == "indirect light":
                output = tensor_to_viz(render_pkg["rend_indirect_light"])
            elif render_type == "feature map":
                output = tensor_to_viz(gaussians.mipmap.visualization())
            elif render_type == "envmap":
                output = tensor_to_viz(extract_env_map(gaussians.brdf_mlp, srgb=True))
            elif render_type == "envmap2":
                output = tensor_to_viz(extract_env_map(gaussians.brdf_mlp, rotate=True, srgb=True))
            elif render_type == "alpha map":
                output = tensor_to_viz(render_pkg["rend_alpha"].repeat(3,1,1))
            elif render_type == "bounding volume map":
                mask_map = torch.zeros_like(rendering) if render_pkg["mask_map"] is None else render_pkg["mask_map"].repeat(3,1,1)
                output = tensor_to_viz(mask_map)
            else:
                print(f"Unsupported render type: {render_type}")

            output = np.asarray(Image.fromarray(output).resize((400,400)))
            client.scene.set_background_image(output, format="jpeg")

if __name__ == "__main__":

    # Set up command line argument parser
    parser = ArgumentParser(description="Exporting script parameters")
    lp = ModelParams(parser, sentinel=True)
    pp = PipelineParams(parser)
    parser.add_argument("-i", "--iteration", type=int, default=50000)
    parser.add_argument("-e", "--relight_envmap_path", default="", help="Envmap path to relight with")
    parser.add_argument("-v", "--view", default="Rendered", help="What to visualize first")
    args = get_combined_args(parser)

    # init gui
    server = network_gui.init(initial_value=args.view)

    print(args)
    # args = parser.parse_args(sys.argv[1:])
    print("View: " + args.model_path)
    view(lp.extract(args), pp.extract(args), args, server)
    print("\nViewing complete.")