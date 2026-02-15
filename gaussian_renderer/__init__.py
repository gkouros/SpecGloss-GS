import torch
import torch.nn.functional as F
import math
import numpy as np
from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
import torch.nn
from utils.sh_utils import eval_sh
from scene.NVDIFFREC.util import safe_normalize, reflect
from utils.point_utils import depth_to_normal
from utils.image_utils import DisplayMapper, linear_to_srgb
from utils.graphics_utils import sample_camera_rays


def reflect_rays(rays_d, normal):
    w_o = safe_normalize(-rays_d)  # (H,W,3)
    normal = normal.permute(1,2,0)  # (H,W,3)
    normal = safe_normalize(normal)  # (H,W,3)
    refl_rays = reflect(w_o, normal)  # (H,W,3)
    refl_rays = safe_normalize(refl_rays)  # (H,W,3)
    return refl_rays

def compute_visibility(ray_tracer, rays_d, rays_o, normal, surf_depth, render_alpha, visibility_threshold=0.95, eps=1e-1):
    """ estimate visibility """
    visibility = torch.ones_like(render_alpha)  # (H,W)
    if ray_tracer is not None:
        normal = normal / render_alpha.clamp_min(1e-6)  # remove opacity scaling
        refl_rays = reflect_rays(rays_d, normal)
        intersections = rays_o + surf_depth.permute(1,2,0) * rays_d  # (H,W,3)
        intersections = intersections + eps * refl_rays  # prevent self-hit
        mask = (render_alpha > visibility_threshold).squeeze(0)  # (H,W)
        _, _, depth = ray_tracer.trace(intersections[mask], refl_rays[mask])  # (H,W)?
        visibility[:, mask] = (depth >= 10).float()
    return visibility

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier=1.0, initial_stage=False):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # get gaussian attributes
    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity
    scales = pc.get_scaling
    rotations = pc.get_rotation

    # collect features for rasterization
    features = [pc.get_base_color, pc.get_roughness, pc.get_metallic]  # (N,5)
    if pc.spec_gloss_brdf or pc.use_specular:
        features.append(pc.get_specular)  # (N, 3)
    if pc.use_residual and pc.active_sh_degree > 0:
        rays_d_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_opacity.shape[0], 1))  # (N,3) view direction per point
        rays_d_pp = rays_d_pp / rays_d_pp.norm(dim=1, keepdim=True)
        if pc.env_scope_radius > 0:
            dirs = rays_d_pp
        else:
            normal_pp = pc.get_surfel_minimum_axis(rays_d_pp) if True else pc.get_surfel_minimum_axis()  # (N,3) normal per point
            refl_pp = reflect(-rays_d_pp, normal_pp)  # (N,3) reflection direction per point
            dirs = refl_pp
        shs_view = pc.get_features_sh.transpose(1,2).view(-1, 3, (pc.max_sh_degree+1)**2)  # (N,3,16) sh coefficients per point
        residual_color_pp = eval_sh(pc.active_sh_degree, shs_view, dirs)  # (N,3) eval SH coefficients to get residual color per point
        features.append(residual_color_pp.clamp(min=0.0))  # (N,3)
    if pc.env_scope_radius > 0:
        features.append(pc.get_mask)  # + (N,1)

    # aggregate all features
    features = torch.concatenate(features, axis=1)  # (N,6-20)

    bg_features = torch.scatter(torch.zeros(features.shape[-1]), 0, torch.tensor([3]), torch.tensor([pc.roughness_bg_color])).to(bg_color.device)  # (F,) bg should be 0 everywhere except for roughness e.g. [0,0,0,1,0,0,...,0]

    # setup rasterization settings
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=math.tan(viewpoint_camera.FoVx * 0.5),
        tanfovy=math.tan(viewpoint_camera.FoVy * 0.5),
        bg=bg_features,  # setting nonzero bg color for diffuse here causes boundary artifacts on envmap and stray gaussians
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=0,  # not used in our modified GaussianRasterizer
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False,  # pipe.debug
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # rasterize the material properties and features
    rendered_image, radii, allmap = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=features,
        colors_precomp=None,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=None,
    )

    # get the rasterized material properties
    base_color_map = rendered_image[0:3]  # base color map I_d (3,H,W)
    roughness_map = rendered_image[3:4]  # roughness map M (1,H,W)
    metallic_map = rendered_image[4:5]  # metallic map (1,H,W)
    idx = 5
    if pc.spec_gloss_brdf or pc.use_specular:
        specular_map = rendered_image[idx:idx+3]  # specular map (3,H,W)
        idx += 3
    else:
        specular_map = torch.zeros_like(base_color_map, requires_grad=False)

    # alpha map
    render_alpha = allmap[1:2]  # (1,H,W)

    # get normal map and transform from view space to world space
    viewspace_normal = allmap[2:5].permute(1,2,0)  # (H, W, 3) in view space
    render_normal = (viewspace_normal @ (viewpoint_camera.world_view_transform[:3,:3].T)).permute(2,0,1)  # (3,H,W)

    # get median depth map
    render_depth_median = allmap[5:6]
    render_depth_median = torch.nan_to_num(render_depth_median, 0, 0)

    # get expected depth map
    render_depth_expected = allmap[0:1]
    render_depth_expected = (render_depth_expected / render_alpha.clamp(min=1e-6))  # (1, H, W)
    render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)

    # get depth distortion map
    render_dist = allmap[6:7]

    # pseudo surface attributes: surf depth is either median or expected by setting depth_ratio to 1 or 0
    # for bounded scene, use median depth, i.e., depth_ratio = 1;
    # for unbounded scene, use expected depth, i.e., depth_ratio = 0, to reduce disk aliasing.
    surf_depth = render_depth_expected * (1 - pipe.depth_ratio) + (pipe.depth_ratio) * render_depth_median

    # assume the depth points form the 'surface' and generate pseudo surface normal for regularizations.
    surf_normal = depth_to_normal(viewpoint_camera, surf_depth).permute(2,0,1) * render_alpha.detach()  # (3,H,W)
    surf_normal_expected = depth_to_normal(viewpoint_camera, render_depth_expected).permute(2,0,1) * render_alpha.detach()  # (3,H,W)
    surf_normal_median = depth_to_normal(viewpoint_camera, render_depth_median).permute(2,0,1) * render_alpha.detach()  # (3,H,W)

    # get rays
    rays_d, rays_o = sample_camera_rays(viewpoint_camera, normalize=False)  # (H,W,3), (3,)

    # indirect color with spherical harmonics
    if pc.use_residual and pc.active_sh_degree > 0:
        residual_linear = rendered_image[idx:idx+3].permute(1,2,0)  # (H,W,3)
        idx += 3
    else:
        residual_linear = torch.zeros_like(rays_d, requires_grad=False)

    # mask for foreground vs background for real scenes
    mask_map = rendered_image[idx:idx+1].detach() if pc.env_scope_radius > 0 else None  # env scope mask (1,H,W)
    idx += int(pc.env_scope_radius > 0)
    if pc.env_scope_radius > 0:
        # visibility_map = torch.logical_xor(~mask_map.bool(), visibility_map.bool()).float()
        visibility_map = mask_map
    else:
        # compute visibility map
        visibility_map = compute_visibility(pc.ray_tracer, rays_d, rays_o, render_normal.detach(), surf_depth.detach(), render_alpha.detach(), visibility_threshold=pc.visibility_threshold)  # (1,H,W)

    # apply shading
    if initial_stage:
        diffuse_linear = base_color_map
        specular_linear = residual_linear
        direct_linear = indirect_linear = torch.zeros_like(render_normal)
    else:
        # unpremultiply materials to remove opacity scaling
        unpremultiply = lambda x: x / render_alpha.clamp_min(1e-6)  # DO NOT x CLAMP TO 0-1 and DO NOT DETACH!
        gb_normal = (unpremultiply(render_normal) if pc.unpremult_normals else render_normal).permute(1,2,0)

        # compute final rendered color
        _, brdf_pkg = pc.brdf_mlp.shade(
            viewdir=safe_normalize(rays_d)[None],
            gb_normal=safe_normalize(gb_normal)[None] if pc.normalize_normals else gb_normal[None],
            kd=base_color_map.permute(1,2,0)[None],
            kr=roughness_map.permute(1,2,0)[None],
            km=metallic_map.permute(1,2,0)[None],
            ks=specular_map.permute(1,2,0)[None],
            disney_brdf=pc.disney_brdf,
            spec_gloss_brdf=pc.spec_gloss_brdf,
        )  # ouputs in (1,H,W,3)

        diffuse_linear = brdf_pkg['diffuse'].squeeze().permute(2,0,1) * render_alpha  # (3,H,W)
        specular_linear = brdf_pkg['specular'].squeeze().permute(2,0,1) * render_alpha  # (3,H,W)
        F0 = brdf_pkg["F0"].squeeze().permute(2,0,1) * render_alpha  # (3,H,W)

        # Residual Branch
        residual_linear = residual_linear.permute(2,0,1) * render_alpha  # (3,H,W)

        # additional outputs for debugging
        with torch.no_grad():
            direct_linear = brdf_pkg['direct_light'].squeeze(0).permute(2,0,1) * render_alpha  # (3,H,W)
            indirect_linear = (1 - visibility_map) * residual_linear  # (3,H,W)

    # compute rendered color
    if pc.env_scope_radius > 0:
        diffuse_linear = mask_map * diffuse_linear + (1 - mask_map) * residual_linear  # (3,H,W)
        rendered_linear = mask_map * (diffuse_linear + specular_linear) + (1 - mask_map) * residual_linear  # (3,H,W)
    else:
        rendered_linear = diffuse_linear + visibility_map * specular_linear + (1 - visibility_map) * residual_linear  # (3,H,W)

    # apply srgb conversion and clamping to range 0-1 and apply background color
    def to_srgb(img, background=False):
        if pc.srgb:
            if pc.clamp_then_srgb:
                img = linear_to_srgb(img.clamp(0.0, 1.0))
            else:
                img = linear_to_srgb(img).clamp(0.0, 1.0)

        if background:
            img = img + (bg_color[:3, None, None] * (1 - render_alpha))  # DO NOT DETACH background mask
        return img

    rendered_rgb = to_srgb(rendered_linear, background=True)  # (3,H,W)

    # srgb conversion for color outputs
    with torch.no_grad():
        diffuse_color = to_srgb(diffuse_linear, background=False)  # (3,H,W)
        specular_color = to_srgb(specular_linear, background=False)  # (3,H,W)
        direct_light = to_srgb(direct_linear, background=True)  # (3,H,W)
        indirect_light = to_srgb(indirect_linear, background=True)  # (3,H,W)
        residual_color = to_srgb(residual_linear, background=True)  # (3,H,W)
        base_color = to_srgb(base_color_map, background=True)  # (3,H,W)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    rets = {
        "render": rendered_rgb,
        "rend_linear": rendered_linear,
        "rend_base_color": base_color,
        "rend_base_color_linear": base_color_map,
        "rend_diffuse_color": diffuse_color,
        "rend_diffuse_linear": diffuse_linear,
        "rend_specular_color": specular_color,
        "rend_specular_linear": specular_linear,
        "rend_residual_color": residual_color,
        "rend_residual_linear": residual_linear,
        "rend_roughness": roughness_map,
        "rend_metallic": metallic_map,
        "rend_specular": specular_map,
        "rend_F0": F0,
        "rend_direct_light": direct_light,
        "rend_indirect_light": indirect_light,
        "rend_visibility": visibility_map,
        "viewspace_points": means2D,
        "visibility_filter" : radii > 0,
        "radii": radii,
        "rend_alpha": render_alpha,
        "rend_dist": render_dist,
        "surf_depth": surf_depth,
        "rend_normal": render_normal,
        "surf_normal": surf_normal,
        "surf_normal_median": surf_normal_median,
        "surf_normal_expected": surf_normal_expected,
        "mask_map": mask_map,
    }

    return rets