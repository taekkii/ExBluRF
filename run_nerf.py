import os
import time
import math

import cv2
import imageio
from tensorboardX import SummaryWriter

from NeRF import *
from load_llff import load_llff_data, load_llff_data_learned, gen_render_path, load_subframe_poses
from run_nerf_helpers import *
from metrics import compute_img_metric

import tqdm
import json
import numpy as np

# np.random.seed(0)
DEBUG = False
##
import inference
import option
from plenoxel.utils import apply_regularizer

from utils.util_visualization import camera_traj_visualization
from collections import OrderedDict

import sys

def train():
    parser = option.config_parser()
    args = parser.parse_args()
    
    if len(args.torch_hub_dir) > 0:
        print(f"Change torch hub cache to {args.torch_hub_dir}")
        torch.hub.set_dir(args.torch_hub_dir)
    
    # Load data
    K = None
    if args.dataset_type == 'llff':
        if  args.eval_only and  args.dataset_name != 'synthetic':
            print("Load data from optimized camera poses")
            scene_name = os.path.basename(args.datadir)
            new_data_dir = os.path.join(args.basedir, args.expname, scene_name)

            
            images, poses, bds, render_poses, i_test = load_llff_data(args, new_data_dir, factor=None,
                                                                      recenter=False, bd_factor=None,
                                                                      path_epi=args.render_epi)

        else:
            images, poses, bds, render_poses, i_test = load_llff_data(args, args.datadir, args.factor,
                                                                    recenter=True, bd_factor=.75,
                                                                    spherify=args.spherify,
                                                                    path_epi=args.render_epi)
        hwf = poses[0, :3, -1]
        poses = poses[:, :3, :4]
        print('Loaded llff', images.shape, args.datadir)
    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

    if args.dataset_name in ['synthetic', 'exblur']:
        poses_subframes = load_subframe_poses(args, args.datadir, recenter=True)

    if not args.no_ndc and not args.eval_only:
        print('Camera pose, z_min: {}, z_max: {}'.format(np.min(poses[:,2,3]), np.max(poses[:,2,3])))
        print('Normalized Camera pose, z_min: {}, z_max: {}'.format(np.min(poses[:,2,3]), np.max(poses[:,2,3])))
        print("[WORLD CAMERA LOCATIONS]\n", poses[:,:,3], sep='')
        print("[MIN]", poses[:,:,3].min(axis=0), "[MAX]", poses[:,:,3].max(axis=0))

    print('LLFF holdout,', args.llffhold)
    i_test = np.arange(images.shape[0])[::args.llffhold]

    i_val = i_test
    i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                        (i not in i_test and i not in i_val)])

    print('Render path for video')
    if args.interp_video:
        render_poses = gen_render_path(poses[i_test], N_views=60)

    print('DEFINING BOUNDS')
    near = 0.
    far = 1.
    
    print('NEAR FAR', near, far)

    imagesf = images
    images = (images * 255).astype(np.uint8)
    images_idx = np.arange(0, len(images))

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    if K is None:
        K = np.array([
            [focal, 0, 0.5 * W],
            [0, focal, 0.5 * H],
            [0, 0, 1]
        ])

    if args.render_test:
        render_poses = np.array(poses)
    
    # Create log dir and copy the config file
    basedir = args.basedir
    tensorboardbase = args.tbdir
    expname = args.expname
    test_metric_file = os.path.join(basedir, expname, 'test_metrics.txt')
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    os.makedirs(os.path.join(tensorboardbase, expname), exist_ok=True)

    tensorboard = SummaryWriter(os.path.join(tensorboardbase, expname))

    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None and not args.render_only:
        if os.path.exists(test_metric_file) and not args.no_reload:
            with open(test_metric_file, 'a') as file:
                print("\n[Replayed Script from here] Highly likely to reload purpose.",file=file)
                print("Command used is:", file=file)
                print(*sys.argv, end="\n\n", file=file)
        else:
            with open(test_metric_file, 'w') as file:
                print("\n[New training]", file=file)
                print("Command used is:", file=file)
                print(*sys.argv, end="\n\n", file=file)
        
    # The DSK module
    if args.kernel_type == 'deformablesparsekernel':
        kernelnet = DSKnet(len(images), torch.tensor(poses[:, :3, :4]),
                           args.kernel_ptnum, args.kernel_hwindow,
                           random_hwindow=args.kernel_random_hwindow, in_embed=args.kernel_rand_embed,
                           random_mode=args.kernel_random_mode,
                           img_embed=args.kernel_img_embed,
                           spatial_embed=args.kernel_spatial_embed,
                           depth_embed=args.kernel_depth_embed,
                           num_hidden=args.kernel_num_hidden,
                           num_wide=args.kernel_num_wide,
                           short_cut=args.kernel_shortcut,
                           pattern_init_radius=args.kernel_pattern_init_radius,
                           isglobal=args.kernel_isglobal,
                           optim_trans=args.kernel_global_trans,
                           optim_spatialvariant_trans=args.kernel_spatialvariant_trans,
                           learnable_pose=args.learnable_pose,
                           recenter_traj=args.recenter_traj)
    elif args.kernel_type == 'camtrajectorykernel':
        kernelnet = CTKNet(num_imgs=len(images), num_pts=args.kernel_ptnum, poses=torch.tensor(poses[:, :3, :4]),
                            order_curve=args.curve_order,
                            learnable_pose=args.learnable_pose,
                            recenter_traj=args.recenter_traj)
    elif args.kernel_type == 'none':
        kernelnet = None
    else:
        raise RuntimeError(f"kernel_type {args.kernel_type} not recognized")
   
    # Create nerf model
    if args.architecture == 'plenoxel' and args.ndc_bound_config == "automatic" and not args.no_ndc:
        cx, cy, rx, ry, ndc_focal = plenoxel.utils.get_reasonable_ndc_bounds(args, poses, H, W, K, virtual_far = bds.max())
        nerf_init_kwargs = dict( bound_config=dict(cx=cx, cy=cy, rx=rx, ry=ry, ndc_focal=ndc_focal))
    else:
        nerf_init_kwargs = {}
    nerf_init_kwargs['wid'] = W
    nerf_init_kwargs['hei'] = H
    nerf = NeRFAll(args, kernelnet, nerf_init_kwargs=nerf_init_kwargs)
    optimizer, lr_funcs, lr_factors = prepare_optimizer_and_scheduler(args=args, nerf=nerf, kernelnet=kernelnet)

    if args.architecture == 'plenoxel':
        lr_sigma_func = lr_funcs['lr_sigma_func']
        lr_sh_func = lr_funcs['lr_sh_func']
        lr_basis_func = lr_funcs['lr_basis_func']
        lr_sigma_bg_func = lr_funcs['lr_sigma_bg_func']
        lr_color_bg_func = lr_funcs['lr_color_bg_func']
        lr_kernelnet_func = lr_funcs['lr_kernelnet_func']
        
        lr_sigma_factor = lr_factors['lr_sigma_factor']
        lr_sh_factor = lr_factors['lr_sh_factor']
        lr_basis_factor = lr_factors['lr_basis_factor']
    
    # Configure Plenoxel resolution settings
    if args.architecture == 'plenoxel':
        reso_list = json.loads(args.reso)
        reso_id = 0
        if args.dataset_type == 'llff':
            ndc_coeffs = (2*K[0,0]/W , 2*K[1,1]/H)
            resample_cameras = [
                    svox2.Camera('cuda',
                                K[0,0],
                                K[1,1],
                                K[0,2],
                                K[1,2],
                                width=W,
                                height=H,
                                ndc_coeffs=ndc_coeffs) for i in range(len(poses))]
        else:
            raise NotImplementedError

    # Load recent checkpoint
    global_step, time_total, max_memory_consumption, reso_id, last_upsamp_step, lr_factor = \
        load_recent_checkpoint(args=args, 
                               basedir=basedir, 
                               expname=expname, 
                               nerf=nerf, 
                               optimizer=optimizer,
                               W=W, H=H)
    lr_sigma_factor *= lr_factor
    lr_sh_factor *= lr_factor
    lr_basis_factor *= lr_factor
    
    # figuring out the train/test configuration
    render_kwargs_train = {
        'perturb': args.perturb,
        'N_importance': args.N_importance,
        'N_samples': args.N_samples,
        'use_viewdirs': args.use_viewdirs,
        'white_bkgd': args.white_bkgd,
        'raw_noise_std': args.raw_noise_std,
    }
    if args.architecture == 'plenoxel':
        render_kwargs_train['sparsity_loss'] = args.lambda_sparsity
        render_kwargs_train['beta_loss'] = args.lambda_beta
    # NDC only good for LLFF-style forward facing data
    if args.no_ndc:  # args.dataset_type != 'llff' or
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp
    render_kwargs_test = {k: render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    bds_dict = {
        'near': near,
        'far': far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # Move testing data to GPU
    render_poses = torch.tensor(render_poses[:, :3, :4]).cuda()
    nerf = nerf.cuda()
    # Evaluate only test images.
    if args.export_trajectory:
        inference.export_trajectory(args=args,
                                    nerf=nerf,
                                    poses_subframes=poses_subframes,
                                    i_train=i_train,
                                    basedir=basedir, expname=expname, global_step=global_step) 
    if args.visualize_voxel_grid:
        # Visualize boxel_grid
        inference.visualize_voxel_grid(args=args,
                                       nerf=nerf,
                                       poses=poses,
                                       render_kwargs_test=render_kwargs_test,
                                       H=H, W=W, K=K,
                                       i_test=i_test,
                                       basedir=basedir,
                                       expname=expname)
        return
    if args.eval_only:
        inference.eval_only(args=args,
                            nerf=nerf,
                            poses=poses,
                            render_kwargs_test=render_kwargs_test,
                            imagesf=imagesf,
                            i_test=i_test,
                            H=H, W=W, K=K ,basedir=basedir, expname=expname, start=global_step, global_step=global_step)
        
        return
    # Short circuit if only rendering out from trained model
    if args.render_only:
        inference.render_only(args=args, 
                              nerf=nerf,
                              images_train=images[i_train],
                              poses_train=poses[i_train],
                              poses=poses,
                              render_poses=render_poses,
                              render_kwargs_test=render_kwargs_test,
                              hwf=hwf, 
                              K=K,
                              basedir=basedir,
                              expname=expname,
                              start=global_step)
        return
    if args.export_colmap:
        inference.export_colmap(args=args, 
                                nerf=nerf,
                                kernelnet=kernelnet, 
                                render_kwargs_test=render_kwargs_test,
                                images=images,
                                i_train=i_train,
                                i_test=i_test,
                                H=H, W=W, K=K, hwf=hwf, 
                                basedir=basedir, expname=expname)
        return

    # ============================================
    # Prepare ray dataset if batching random rays
    # ============================================
    N_rand = args.N_rand
    train_datas = {}
    
    # if downsample, downsample the images
    if args.datadownsample > 0:
        images_train = np.stack([cv2.resize(img_, None, None,
                                            1 / args.datadownsample, 1 / args.datadownsample,
                                            cv2.INTER_AREA) for img_ in imagesf], axis=0)
    else:
        images_train = imagesf

    num_img, hei, wid, _ = images_train.shape
    print(f"train on image sequence of len = {num_img}, {wid}x{hei}")
    k_train = np.array([K[0, 0] * wid / W, 0, K[0, 2] * wid / W,
                        0, K[1, 1] * hei / H, K[1, 2] * hei / H,
                        0, 0, 1]).reshape(3, 3).astype(K.dtype)
    
    # For random ray batching
    print('get rays')
    rays = np.stack([get_rays_np(hei, wid, k_train, p) for p in poses[:, :3, :4]], 0)  # [N, ro+rd, H, W, 3]
    rays = np.transpose(rays, [0, 2, 3, 1, 4])

    train_datas['rays'] = rays[i_train].reshape(-1, 2, 3)
    xs, ys = np.meshgrid(np.arange(wid, dtype=np.float32), np.arange(hei, dtype=np.float32), indexing='xy')
    xs = np.tile((xs[None, ...] + HALF_PIX) * W / wid, [num_img, 1, 1])
    ys = np.tile((ys[None, ...] + HALF_PIX) * H / hei, [num_img, 1, 1])
    train_datas['rays_x'], train_datas['rays_y'] = xs[i_train].reshape(-1, 1), ys[i_train].reshape(-1, 1)
    train_datas['rgbsf'] = images_train[i_train].reshape(-1, 3)
    images_idx_tile = images_idx.reshape((num_img, 1, 1))
    images_idx_tile = np.tile(images_idx_tile, [1, hei, wid])
    train_datas['images_idx'] = images_idx_tile[i_train].reshape(-1, 1).astype(np.int64)
    
    print('shuffle rays')
    shuffle_idx = np.random.permutation(len(train_datas['rays']))
    train_datas = {k: v[shuffle_idx] for k, v in train_datas.items()}
    train_datas = {k: torch.tensor(v).cuda() for k, v in train_datas.items()}
    print('done')
    i_batch = 0

    # Move training data to GPU
    images = torch.tensor(images).cuda()
    imagesf = torch.tensor(imagesf).cuda()
    poses = torch.tensor(poses).cuda()

    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)
    
    def main_loop(args, start, N_iters, train_datas, render_kwargs_train):        
        nonlocal i_batch, time_total, max_memory_consumption, last_upsamp_step, global_step, reso_id
        nonlocal lr_sigma_factor, lr_sh_factor, lr_basis_factor
        nonlocal shuffle_idx
        # to print average at printing time.
        psnr_sum = 0.
        img_loss_sum = 0.
        align_loss_sum = 0.
        
        for i in tqdm.tqdm(range(start, N_iters),desc='[Training]', initial=start, total=N_iters-1):
            
            global_step += 1
            time0 = time.time()
            # Sample random ray batch
            iter_data = {k: v[i_batch:i_batch + N_rand] for k, v in train_datas.items()}
            batch_rays = iter_data.pop('rays').permute(0, 2, 1)

            #####  Core optimization loop  #####
            nerf.train()
            if global_step == args.kernel_start_iter:
                torch.cuda.empty_cache()
            
            target_rgb = iter_data['rgbsf'].squeeze(-2)
            
            rgb, rgb0, extra_loss = nerf(H, W, K, chunk=args.chunk,
                                        rays=batch_rays, rays_info=iter_data,
                                        retraw=True, force_naive=global_step < args.kernel_start_iter,
                                        freeze_kernel = i < args.kernel_start_iter,
                                        aggregate=args.kernel_aggregate_type,
                                        **render_kwargs_train)
            
            max_memory_consumption = max(max_memory_consumption, torch.cuda.memory_reserved())
            # Compute Losses
            # =====================
            img_loss = img2mse(rgb, target_rgb)
            loss = img_loss
            psnr = mse2psnr(img_loss)

            if rgb0 is not None:
                img_loss0 = img2mse(rgb0, target_rgb)
                loss = loss + img_loss0
            
            if not args.disable_align_loss and global_step >= args.kernel_start_iter and kernelnet is not None:
                extra_loss = {k: torch.mean(v) for k, v in extra_loss.items()}
                if len(extra_loss) > 0:
                    for k, v in extra_loss.items():
                        if f"kernel_{k}_weight" in vars(args).keys():
                            if vars(args)[f"{k}_start_iter"] <= i <= vars(args)[f"{k}_end_iter"]:
                                loss = loss + v * vars(args)[f"kernel_{k}_weight"]
                
            # Shuffle after every epoch.
            i_batch += N_rand
            
            if i_batch >= len(train_datas['rays']):
                print("shuffle data after an epoch!")
                shuffle_idx = np.random.permutation(len(train_datas['rays']))
                train_datas = {k: v[shuffle_idx] for k, v in train_datas.items()}
                
                i_batch = 0

            psnr_sum += psnr.item()
            img_loss_sum += img_loss.item()
            align_loss_sum += loss.item() - img_loss.item()
            loss.backward()
            
            if args.architecture == 'plenoxel':
                plenoxel_model = nerf.plenoxel
                if args.lr_fg_begin_step > 0 and i == args.lr_fg_begin_step:
                    plenoxel_model.density_data.data[:] = args.init_sigma
                lr_sigma = lr_sigma_func(i) * lr_sigma_factor
                lr_sh = lr_sh_func(i) * lr_sh_factor
                ### basically not used ###
                lr_basis = lr_basis_func(i - args.lr_basis_begin_step) * lr_basis_factor
                lr_sigma_bg = lr_sigma_bg_func(i - args.lr_basis_begin_step) * lr_basis_factor
                lr_color_bg = lr_color_bg_func(i - args.lr_basis_begin_step) * lr_basis_factor
                ###     until here     ###
                apply_regularizer(args, plenoxel_model=plenoxel_model, ndc_coeffs=ndc_coeffs)
                # run plenoxel optimizers
                # Manual SGD/rmsprop step
                if i >= args.lr_fg_begin_step:
                    plenoxel_model.optim_density_step(lr_sigma, beta=args.rms_beta, optim=args.sigma_optim)
                    plenoxel_model.optim_sh_step(lr_sh, beta=args.rms_beta, optim=args.sh_optim)
                
                if plenoxel_model.use_background:
                    plenoxel_model.optim_background_step(lr_sigma_bg, lr_color_bg, beta=args.rms_beta, optim=args.bg_optim)
                
                if i >= args.lr_basis_begin_step:
                    if plenoxel_model.basis_type == svox2.BASIS_TYPE_3D_TEXTURE:
                        plenoxel_model.optim_basis_step(lr_basis, beta=args.rms_beta, optim=args.basis_optim)
                
            # run optimizer (now same procedure for nerf/plenoxel)
            if optimizer is not None:
                if args.architecture == 'nerf':
                    decay_rate = 0.1
                    decay_steps = args.lrate_decay * 1000
                    new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
                elif args.architecture == 'plenoxel':
                    new_lrate = lr_kernelnet_func(i - args.kernel_start_iter)
                else:
                    raise NotImplementedError()
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lrate
                optimizer.step()
                optimizer.zero_grad()
            dt = time.time() - time0
            time_total += dt
            
            # Rest is logging
            if i % args.i_weights == 0:
                path = os.path.join(basedir, expname, 'recent.tar')
                path_plenoxel = os.path.join(basedir, expname, 'recent')
                sdict = {k:v for k,v in nerf.state_dict().items() if 'plenoxel' not in k}
                torch.save({
                    'global_step': global_step,
                    'network_state_dict': sdict,
                    'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None,
                    'reso_id': reso_id if args.architecture == 'plenoxel' else None,
                    'last_upsamp_step': last_upsamp_step,
                    'time': time_total,
                    'memory': max_memory_consumption,
                    'nerf_init_kwargs': nerf.nerf_init_kwargs,
                    'lr_factor': 0.0 if args.architecture == 'plenoxel' else lr_sigma_factor
                }, path)
                if args.architecture == 'plenoxel':                
                    nerf.plenoxel.save(path_plenoxel)
                print('Saved checkpoints at', path)

            if i % args.i_cam == 0 and args.kernel_type == 'camtrajectorykernel':
                if args.dataset_name in ['synthetic', 'exblur']:
                    inference.export_trajectory(args=args,
                                                nerf=nerf,
                                                poses_subframes=poses_subframes,
                                                i_train=i_train,
                                                basedir=basedir, expname=expname, global_step=global_step)

            if i % args.i_testset == 0 and kernelnet is not None:
                blurredimgdir = os.path.join(basedir, expname,
                                        f"blurred_{i:06d}")
                os.makedirs(blurredimgdir, exist_ok=True)
                dummy_num = ((len(i_train) - 1) // args.num_gpu + 1) * args.num_gpu - len(i_train)
                imgidx = torch.arange(len(poses), dtype=torch.long).to(poses.device).reshape(-1, 1)[i_train]
                imgidx_pad = torch.cat([imgidx, torch.zeros((dummy_num, 1), dtype=torch.long)], dim=0)

                with torch.no_grad():
                    nerf.eval()
                    rgbs = nerf(
                            hwf[0], hwf[1], K, args.chunk,
                            poses=None,
                            render_kwargs=render_kwargs_test,
                            render_factor=args.render_factor,
                            images_indices=imgidx_pad,
                        )
                for i_output, idx_rgb in enumerate(imgidx):
                    idx = idx_rgb.item()
                    rgb_blurred = rgbs[i_output]
                    rgb8 = to8b(rgb_blurred.cpu().numpy())
                    filename = os.path.join(blurredimgdir, f'{idx:03d}_blurred.png')
                    imageio.imwrite(filename, rgb8)

                    rgb8 = images[idx].cpu().numpy()
                    filename = os.path.join(blurredimgdir, f'{idx:03d}_input.png')
                    imageio.imwrite(filename, rgb8)

            if i % args.i_video == 0 and i > 0:
                # Turn on testing mode
                with torch.no_grad():
                    nerf.eval()
                    rgbs, disps = nerf(H, W, K, args.chunk, poses=render_poses, render_kwargs=render_kwargs_test)
                    rgbs = rgbs.reshape(-1, H, W, 3)
                    disps = disps.reshape(-1, H, W)

                print("done")
                moviebase = os.path.join(basedir, expname, '{}_spiral_{:06d}_'.format(expname, global_step))
                
                rgbs = rgbs.cpu().numpy()
                disps = disps.cpu().numpy()
                imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=30, quality=8)
                imageio.mimwrite(moviebase + 'disp.mp4', to8b(disps / disps.max()), fps=30, quality=8)

            if i % args.i_testset == 0 and i > 0:
                testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(global_step))
                os.makedirs(testsavedir, exist_ok=True)
                print('test poses shape', poses.shape)
                dummy_num = ((len(poses) - 1) // args.num_gpu + 1) * args.num_gpu - len(poses)
                dummy_poses = torch.eye(3, 4).unsqueeze(0).expand(dummy_num, 3, 4).type_as(render_poses)
                print(f"Append {dummy_num} # of poses to fill all the GPUs")
                with torch.no_grad():
                    nerf.eval()
                    rgbs, disps = nerf(H, W, K, args.chunk, poses=torch.cat([poses, dummy_poses], dim=0).cuda(),
                                render_kwargs=render_kwargs_test)
                    rgbs = rgbs[:len(rgbs) - dummy_num]
                    rgbs = rgbs.reshape(len(rgbs), H, W, 3)
                    disps = disps.reshape(-1, H, W)
                    rgbs_save = rgbs  # (rgbs - rgbs.min()) / (rgbs.max() - rgbs.min())
                    # saving
                    for rgb_idx, rgb in enumerate(rgbs_save):
                        rgb8 = to8b(rgb.cpu().numpy())
                        filename = os.path.join(testsavedir, f'{rgb_idx:03d}.png')
                        imageio.imwrite(filename, rgb8)
                    
                    for idx, disp in enumerate(disps):
                        disp /= disps.max()
                        disp8 = to8b(disp.cpu().numpy())
                        filename = os.path.join(testsavedir, f'{idx:03d}_disp.png')
                        imageio.imwrite(filename, disp8)
                        
                    # evaluation
                    rgbs = rgbs[i_test]
                    target_rgb_ldr = imagesf[i_test]
    
                    test_mse = compute_img_metric(rgbs, target_rgb_ldr, 'mse')
                    test_psnr = compute_img_metric(rgbs, target_rgb_ldr, 'psnr')
                    test_ssim = compute_img_metric(rgbs, target_rgb_ldr, 'ssim')
                    test_lpips = compute_img_metric(rgbs, target_rgb_ldr, 'lpips')
                    if isinstance(test_lpips, torch.Tensor):
                        test_lpips = test_lpips.item()
                        
                    tensorboard.add_scalar("Test MSE", test_mse, global_step)
                    tensorboard.add_scalar("Test PSNR", test_psnr, global_step)
                    tensorboard.add_scalar("Test SSIM", test_ssim, global_step)
                    tensorboard.add_scalar("Test LPIPS", test_lpips, global_step)
                    
                with open(test_metric_file, 'a') as outfile:
                    outfile.write(f"iter{i}/globalstep{global_step}: MSE:{test_mse:.8f} PSNR:{test_psnr:.8f}"
                                f" SSIM:{test_ssim:.8f} LPIPS:{test_lpips:.8f}"
                                f" Time:{int(time_total)//60//60}h {int(time_total)//60%60}m {int(time_total)%60}s"
                                f" Max_Memory:{max_memory_consumption//1024//1024}MiB\n")

                torch.cuda.empty_cache()

                print('Saved test set')

            if i % args.i_tensorboard == 0 :
                tensorboard.add_scalar("Loss", (img_loss_sum+align_loss_sum)/args.i_print, global_step)
                tensorboard.add_scalar("PSNR", psnr_sum/args.i_print, global_step)
                for k, v in extra_loss.items():
                    tensorboard.add_scalar(k, v.item(), global_step)
                tensorboard.add_scalar("MSELoss", img_loss_sum/args.i_print)
                tensorboard.add_scalar("AlignLoss", align_loss_sum/args.i_print)
                if args.architecture == 'plenoxel':
                    tensorboard.add_scalar("LR_SH", lr_sh)
                    tensorboard.add_scalar("LR_sigma", lr_sigma)
                if optimizer is not None:
                    tensorboard.add_scalar("LR_mlp", new_lrate)

            if i % args.i_print == 0:
                print(f"[TRAIN] Iter: {i} Loss: {(img_loss_sum+align_loss_sum)/args.i_print:.4f}",  
                    f"PSNR: {psnr_sum/args.i_print:.4f}")
                if args.architecture == 'plenoxel':
                    print(f"        lr_sigma: {lr_sigma:.4f} lr_sh: {lr_sh:.4f}")
                if optimizer is not None:
                    print(f"        lr_mlp: {new_lrate:.7f}")
                print(f"        MSE_Loss: {img_loss_sum/args.i_print:.4f} Align_Loss: {align_loss_sum/args.i_print:.4f}")
                print(f"        Total_time: {int(time_total)//60//60}h {int(time_total)//60%60}m {int(time_total)%60}s")
                print(f"        Max_memory_so_far: {max_memory_consumption//1024//1024}MiB")
                psnr_sum = 0.
                img_loss_sum = 0.
                align_loss_sum = 0.

            # adding upsample mechanism
            if args.architecture == 'plenoxel' and (i - last_upsamp_step) >= args.upsamp_every:
                last_upsamp_step = i
                plenoxel_model = nerf.plenoxel

                if reso_id < len(reso_list) - 1:
                    print('* Upsampling from', reso_list[reso_id], 'to', reso_list[reso_id + 1])
                    if args.tv_early_only > 0:
                        print('turning off TV regularization')
                        args.lambda_tv = 0.0
                        args.lambda_tv_sh = 0.0
                    elif args.tv_decay != 1.0:
                        args.lambda_tv *= args.tv_decay
                        args.lambda_tv_sh *= args.tv_decay
                    
                    lr_sigma_factor *= args.afterupsample_lr_factor
                    lr_sh_factor *= args.afterupsample_lr_factor
                    lr_basis_factor *= args.afterupsample_lr_factor
                    
                    reso_id += 1
                    use_sparsify = True
                    z_reso = reso_list[reso_id] if isinstance(reso_list[reso_id], int) else reso_list[reso_id][2]
                    plenoxel_model.resample(reso=reso_list[reso_id],
                                            sigma_thresh=args.density_thresh,
                                            weight_thresh=args.weight_thresh / z_reso if use_sparsify else 0.0,
                                            dilate=2, #use_sparsify,
                                            cameras=resample_cameras if args.thresh_type == 'weight' else None,
                                            max_elements=args.max_grid_elements)

                    if plenoxel_model.use_background and reso_id <= 1:
                        plenoxel_model.sparsify_background(args.background_density_thresh)

                    if args.upsample_density_add:
                        plenoxel_model.density_data.data[:] += args.upsample_density_add
            if args.dataset_name in ['exblur', 'synthetic'] and i==120000:
                break
    
    main_loop(args, global_step+1, args.N_iters+1, train_datas, render_kwargs_train)
    torch.cuda.empty_cache()
    
if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    np.set_printoptions(suppress=True, precision=4)
    train()