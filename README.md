# SpecGloss-GS: Spec-Gloss Surfels and Normal–Diffuse Priors for Relightable Glossy Objects
Georgios Kouros, Minye Wu, Tinne Tuytelaars

| [Project page](https://gkouros.github.io/projects/SpecGloss-GS/) | [Full paper](https://arxiv.org/abs/2510.02069) |

**This repository contains the official implementation of the paper "Spec-Gloss Surfels and Normal–Diffuse Priors for Relightable Glossy Objects" that will appear at WACV 2026 (oral).**

## Abstract
Accurate reconstruction and relighting of glossy objects remains a longstanding challenge, as object shape, material properties, and illumination are inherently difficult to disentangle. Existing neural rendering approaches often rely on simplified BRDF models or parameterizations that couple diffuse and specular components, which restrict faithful material recovery and limit relighting fidelity. We propose a relightable framework that integrates a microfacet BRDF with the specular-glossiness parameterization into 2D Gaussian Splatting with deferred shading. This formulation enables more physically consistent material decomposition, while diffusion-based priors for surface normals and diffuse color guide early-stage optimization and mitigate ambiguity. A coarse-to-fine environment map optimization accelerates convergence, and negative-only environment map clipping preserves high-dynamic-range specular reflections. Extensive experiments on complex, glossy scenes demonstrate that our method achieves high-quality geometry and material reconstruction, delivering substantially more realistic and consistent relighting under novel illumination compared to existing Gaussian splatting methods.

## Pipeline
![Pipeline figure.](assets/pipeline.png)
Overview of our method's pipeline built on 2DGS. Gaussian splats rasterize to a G buffer of albedo, roughness, F0, indirect color, and surface normals. A differentiable prefiltered environment cubemap with mipmaps provides lighting in a physically based deferred renderer. The HDR environment map is learned in a coarse-to-fine manner. Supervision uses an sRGB photometric loss between shaded output and ground truth (GT), plus normal and diffuse priors that reduce ambiguity between geometry, materials, and lighting.

## Video
https://drive.google.com/file/d/1HA3WVqgphOXEiFGP90TJpIIvnOseteRS/preview

<!DOCTYPE html>
<html>
<body>
  <iframe src="https://drive.google.com/file/d/1HA3WVqgphOXEiFGP90TJpIIvnOseteRS/preview"></iframe>
</body>
</html>
