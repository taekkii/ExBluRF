
import numpy as np
import torch

HALF_PIX = 0.5
TINY_NUMBER = 1e-6      # float32 only has 7 decimal digits precision

def convert_to_ndc(origins, directions, ndc_coeffs, near: float = 1.0):
    """Convert a set of rays to NDC coordinates."""
    # Shift ray origins to near plane, not sure if needed
    t = (near - origins[Ellipsis, 2]) / directions[Ellipsis, 2]
    origins = origins + t[Ellipsis, None] * directions

    dx, dy, dz = directions.unbind(-1)
    ox, oy, oz = origins.unbind(-1)

    # Projection
    o0 = ndc_coeffs[0] * (ox / oz)
    o1 = ndc_coeffs[1] * (oy / oz)
    o2 = 1 - 2 * near / oz

    d0 = ndc_coeffs[0] * (dx / dz - ox / oz)
    d1 = ndc_coeffs[1] * (dy / dz - oy / oz)
    d2 = 2 * near / oz;

    origins = torch.stack([o0, o1, o2], -1)
    directions = torch.stack([d0, d1, d2], -1)
    return origins, directions

def setup_render_opts(opt, args):
    """
    Pass render arguments to the SparseGrid renderer options
    """
    opt.step_size = args.step_size
    opt.sigma_thresh = args.sigma_thresh
    opt.stop_thresh = args.stop_thresh
    opt.background_brightness = args.background_brightness
    opt.backend = args.renderer_backend
    opt.random_sigma_std = args.random_sigma_std
    opt.random_sigma_std_background = args.random_sigma_std_background
    opt.last_sample_opaque = args.last_sample_opaque
    opt.near_clip = args.near_clip
    opt.use_spheric_clip = args.use_spheric_clip
    opt.background_r = float(args.background_rgb.split(",")[0])/255.0
    opt.background_g = float(args.background_rgb.split(",")[1])/255.0
    opt.background_b = float(args.background_rgb.split(",")[2])/255.0
    

def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Continuous learning rate decay function. Adapted from JaxNeRF

    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.

    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper

def apply_regularizer(args, plenoxel_model, ndc_coeffs):
    if args.lambda_tv > 0.0:
        #  with Timing("tv_inpl"):
        plenoxel_model.inplace_tv_grad(plenoxel_model.density_data.grad,
                scaling=args.lambda_tv,
                sparse_frac=args.tv_sparsity,
                logalpha=args.tv_logalpha,
                ndc_coeffs=ndc_coeffs,
                contiguous=args.tv_contiguous)
    if args.lambda_tv_sh > 0.0:
        #  with Timing("tv_color_inpl"):
        plenoxel_model.inplace_tv_color_grad(plenoxel_model.sh_data.grad,
                scaling=args.lambda_tv_sh,
                sparse_frac=args.tv_sh_sparsity,
                ndc_coeffs=ndc_coeffs,
                contiguous=args.tv_contiguous)
        
    # FOR LLFF, Below regularizers are not used
    if args.lambda_tv_lumisphere > 0.0:
        plenoxel_model.inplace_tv_lumisphere_grad(plenoxel_model.sh_data.grad,
                scaling=args.lambda_tv_lumisphere,
                dir_factor=args.tv_lumisphere_dir_factor,
                sparse_frac=args.tv_lumisphere_sparsity,
                ndc_coeffs=ndc_coeffs)
    if args.lambda_l2_sh > 0.0:
        plenoxel_model.inplace_l2_color_grad(plenoxel_model.sh_data.grad,
                scaling=args.lambda_l2_sh)
    if plenoxel_model.use_background and (args.lambda_tv_background_sigma > 0.0 or args.lambda_tv_background_color > 0.0):
        plenoxel_model.inplace_tv_background_grad(plenoxel_model.background_data.grad,
                scaling=args.lambda_tv_background_color,
                scaling_density=args.lambda_tv_background_sigma,
                sparse_frac=args.tv_background_sparsity,
                contiguous=args.tv_contiguous)
    if args.lambda_tv_basis > 0.0:
        tv_basis = plenoxel_model.tv_basis()    
        loss_tv_basis = tv_basis * args.lambda_tv_basis
        loss_tv_basis.backward()

################################################
# [taekkii]                                    #
# BELOW ARE NDC HELPERS                         #
################################################

def ndc_rays(H, W, focal, near, rays_o, rays_d):
    """
    See Paper supplementary for details
    """
    # Shift ray origins to near plane
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d

    # Projection
    o0 = -1. / (W / (2. * focal)) * rays_o[..., 0] / rays_o[..., 2]
    o1 = -1. / (H / (2. * focal)) * rays_o[..., 1] / rays_o[..., 2]
    o2 = 1. + 2. * near / rays_o[..., 2]

    d0 = -1. / (W / (2. * focal)) * (rays_d[..., 0] / rays_d[..., 2] - rays_o[..., 0] / rays_o[..., 2])
    d1 = -1. / (H / (2. * focal)) * (rays_d[..., 1] / rays_d[..., 2] - rays_o[..., 1] / rays_o[..., 2])
    d2 = -2. * near / rays_o[..., 2]

    rays_o = torch.stack([o0, o1, o2], -1)
    rays_d = torch.stack([d0, d1, d2], -1)

    return rays_o, rays_d

def get_rays_np(H, W, K, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i + (HALF_PIX - K[0][2])) / K[0][0], -(j + (HALF_PIX - K[1][2])) / K[1][1], -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3, :3],
                    -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))
    return rays_o, rays_d

def get_reasonable_ndc_bounds(args, poses, H, W, K, virtual_far=150.0, radius_padding=0.2, focal_multiplier = 0.9):
    """
    [taekkii]
    Given poses, this function calculates reasonable ndc coefficients,
    Arguments:
        args: system args.
        poses: [view x 3 x 4] extrinsic
        H, W, K: height, width, intrinsic
        virtual_far: Roughly thought "Far plane". This doesn't have to be strictly exact.
        radius_padding: additional radius. Added to side of near plane.
        focal_padding: additional focal. Subtracted to cover more area.
    Returns:
        cx, cy: xy-center of grid space
        rx, ry: xy-radii of grid space
        ndc_focal: focal length with respect to ndc frustrum.
    """
    
    # get rays.
    rays = np.stack([get_rays_np(H, W, K, p) for p in poses[:, :3, :4]], 0)  # [N, ro+rd, H, W, 3]
    rays = np.transpose(rays, [0, 2, 3, 1, 4])
    rays = rays.reshape(-1, 2, 3)
    rays = torch.from_numpy(rays)
    
    rays_o = rays[:,0,:]
    rays_d = rays[:,1,:]
    
    t = ( virtual_far - rays_o[:,2] ) / rays_d[:,2]
    pts = rays_o + t.view(-1,1) * rays_d
    
    x_min = pts[:,0].min().item()
    x_max = pts[:,0].max().item()
    y_min = pts[:,1].min().item()
    y_max = pts[:,1].max().item()
    
    x_scr = max(abs(x_min),abs(x_max))
    y_scr = max(abs(y_min),abs(y_max))
    
    ndc_focal = min(virtual_far * (W/2) / x_scr , virtual_far*(H/2) / y_scr)
    ndc_focal *= focal_multiplier
    
    # transform to ndc rays origin. 
    ndc_rays_o, _ = ndc_rays(H, W, ndc_focal, 1.0, rays_o, rays_d)
            
    # get xy-radii (and xy-center) of grid space.
    # Bounds are determined by simply taking min-max of ndc-rays-origin.
    x_min, y_min = ndc_rays_o[:,:2].min(axis=0).values
    x_max, y_max = ndc_rays_o[:,:2].max(axis=0).values
    
    cx = (x_max.item() + x_min.item()) / 2.0
    cy = (y_max.item() + y_min.item()) / 2.0
    rx = (x_max.item() - x_min.item()) / 2.0 + radius_padding
    ry = (y_max.item() - y_min.item()) / 2.0 + radius_padding
      
    print("===== [Getting reasonable bounds] =====")
    print(f"center: ({cx:.3f},{cy:.3f},0) , radius: ({rx:.3f},{ry:.3f},1)")
    print(f"ndc_focal={ndc_focal:.4f}.")
    print(f"world camera focal={K[0,0]:.3f}")
    print("=======================================")
    return cx, cy, rx, ry, ndc_focal