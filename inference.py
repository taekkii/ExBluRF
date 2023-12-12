
import os
import torch
import imageio
import numpy as np

from utils.util_pos import se3toQuat_single, Mat2Vec_single, inv3x4
from run_nerf_helpers import to8b
from run_nerf_helpers import colorize
from metrics import compute_img_metric
from utils.util_visualization import voxel_grid_visualization

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

def export_trajectory(args, nerf, poses_subframes, i_train, basedir, expname, global_step):
    dir_traj = os.path.join(basedir, expname, "trajectory_{}".format(global_step), expname)
    os.makedirs(dir_traj, exist_ok=True)

    num_imgs = poses_subframes.shape[0]
    num_subframes = poses_subframes.shape[1]
    
    poses_traj = nerf.kernelsnet.export_trajectory(num_points=num_subframes)
    poses_traj = poses_traj[..., :3, :4].detach().cpu()#.numpy()

    poses_traj_trans = poses_traj[..., 3]
    poses_traj_vec = Mat2Vec_single(poses_traj.reshape(-1, 3, 4))
    poses_traj_quat = se3toQuat_single(poses_traj_vec[:, 3:]).reshape(num_imgs, num_subframes, 4)
    
    poses_subframes = torch.from_numpy(poses_subframes)
    poses_subframes_trans = poses_subframes[..., 3]
    poses_subframes_vec = Mat2Vec_single(poses_subframes.reshape(-1, 3, 4))
    poses_subframes_quat = se3toQuat_single(poses_subframes_vec[:, 3:]).reshape(num_imgs, num_subframes, 4)
    
    pred_vis = poses_traj_vec.reshape(num_imgs, num_subframes,6).detach().cpu().numpy()
    gt_vis = poses_subframes_vec.reshape(num_imgs ,num_subframes,6).detach().cpu().numpy()
    #plt.plot(pred_vis[i, :, 0], pred_vis[i, :, 1], 'r-', gt_vis[i,:,0],gt_vis[i,:,1],'b-')

    for i_img in range(num_imgs):
        if not i_img in i_train:
            continue
        dir_colmap = os.path.join(dir_traj, 'colmap', "{}_{}_{}".format(expname, 'colmap', 'VIEW_'+str(i_img)))
        dir_forward = os.path.join(dir_traj, 'forward', "{}_{}_{}".format(expname, 'forward', 'VIEW_'+str(i_img)))
        dir_backward = os.path.join(dir_traj, 'backward', "{}_{}_{}".format(expname, 'backward', 'VIEW_'+str(i_img)))
        os.makedirs(dir_colmap, exist_ok=True)
        os.makedirs(dir_forward, exist_ok=True)
        os.makedirs(dir_backward, exist_ok=True)
        
        
        txt_name_gt_colmap = os.path.join(dir_colmap, "stamped_groundtruth.txt")
        txt_name_gt1 = os.path.join(dir_forward, "stamped_groundtruth.txt")
        txt_name_gt2 = os.path.join(dir_backward, "stamped_groundtruth.txt")

        txt_name_pred1 = os.path.join(dir_forward, "stamped_traj_estimate.txt")
        txt_name_pred2 = os.path.join(dir_backward, "stamped_traj_estimate.txt")
        txt_name_colmap = os.path.join(dir_colmap, "stamped_traj_estimate.txt")

        for i_sub in range(num_subframes):
            tr_pred = poses_traj_trans[i_img, i_sub].tolist()
            qt_pred = poses_traj_quat[i_img, i_sub].tolist()
            with open(txt_name_pred1, 'a') as f_pred:
                f_pred.write("{} {} {} {} {} {} {} {}\n".format(i_sub,\
                    tr_pred[0], tr_pred[1], tr_pred[2], \
                    qt_pred[0], qt_pred[1], qt_pred[2], qt_pred[3]))
            
            tr_gt = poses_subframes_trans[i_img, i_sub].tolist()
            qt_gt = poses_subframes_quat[i_img, i_sub].tolist()
            with open(txt_name_gt_colmap, 'a') as f_gt:
                f_gt.write("{} {} {} {} {} {} {} {}\n".format(i_sub,\
                    tr_gt[0], tr_gt[1], tr_gt[2], \
                    qt_gt[0], qt_gt[1], qt_gt[2], qt_gt[3]))
            with open(txt_name_gt1, 'a') as f_gt:
                f_gt.write("{} {} {} {} {} {} {} {}\n".format(i_sub,\
                    tr_gt[0], tr_gt[1], tr_gt[2], \
                    qt_gt[0], qt_gt[1], qt_gt[2], qt_gt[3]))
            with open(txt_name_gt2, 'a') as f_gt:
                f_gt.write("{} {} {} {} {} {} {} {}\n".format(i_sub,\
                    tr_gt[0], tr_gt[1], tr_gt[2], \
                    qt_gt[0], qt_gt[1], qt_gt[2], qt_gt[3]))


            tr_pred = poses_traj_trans[i_img, num_subframes//2].tolist()
            qt_pred = poses_traj_quat[i_img, num_subframes//2].tolist()
            with open(txt_name_colmap, 'a') as f_colmap:
                f_colmap.write("{} {} {} {} {} {} {} {}\n".format(i_sub,\
                    tr_pred[0], tr_pred[1], tr_pred[2], \
                    qt_pred[0], qt_pred[1], qt_pred[2], qt_pred[3]))

        for i_sub in reversed(range(num_subframes)):
            tr_pred = poses_traj_trans[i_img, i_sub].tolist()
            qt_pred = poses_traj_quat[i_img, i_sub].tolist()

            with open(txt_name_pred2, 'a') as f_pred:
                f_pred.write("{} {} {} {} {} {} {} {}\n".format(num_subframes-(i_sub+1),\
                    tr_pred[0], tr_pred[1], tr_pred[2], \
                    qt_pred[0], qt_pred[1], qt_pred[2], qt_pred[3]))




        ax = plt.figure().add_subplot(projection='3d')
        ax.plot(pred_vis[i_img, :, 0], pred_vis[i_img, :, 1], pred_vis[i_img, :, 2], 'r-') 
        ax.plot(gt_vis[i_img, :, 0], gt_vis[i_img, :, 1], gt_vis[i_img, :, 2], 'b-') 

        plt.savefig(os.path.join(dir_traj, '{0:03d}.png'.format(i_img)))
        plt.close()



def export_colmap(args, nerf, kernelnet, render_kwargs_test, images, i_train, i_test, H, W, K, hwf, basedir, expname):
    print('Export learned camera pose to colmap data format.')
        
    # Get poses from kernelnet.        
    poses_traj = kernelnet.compute_poses_trajectory()
    poses_mat = poses_traj[:, args.kernel_ptnum//2].detach().cpu()
    poses_mat_clone = poses_mat.clone()
    
    # Revert rotation matrix ordering
    # LLFF loader.
    poses_mat = torch.cat([-poses_mat[:, :, 1:2], poses_mat[:, :, 0:1], poses_mat[:, :, 2:]], dim=2)

    # load_colmap_data.
    poses_mat = torch.cat([poses_mat[:, :, 1:2], poses_mat[:, :, 0:1], -poses_mat[:, :, 2:3], poses_mat[:, :, 3:4]], dim=2)
    # Convert pose to world to camera.
    poses_mat = inv3x4(poses_mat[..., :3, :4])

    poses_vec = Mat2Vec_single(poses_mat)
    poses_quat = se3toQuat_single(poses_vec[:, 3:])

    poses_trans = poses_mat[:, :, 3]

    # Make dir for new data directory.
    scene_name = os.path.basename(args.datadir)
    new_data_dir = os.path.join(basedir, expname, scene_name)
    os.makedirs(new_data_dir, exist_ok=True)
    os.makedirs(os.path.join(new_data_dir, 'images_train'), exist_ok=True)
    os.makedirs(os.path.join(new_data_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(new_data_dir, 'sparse_learned'), exist_ok=True)
    
    # Export sharp training views.
    poses_train = poses_mat_clone[i_train]
    
    with torch.no_grad():
        nerf.eval()
        rgbs, _ = nerf(H, W, K, args.chunk, poses=poses_train.cuda(), 
                                            render_kwargs=render_kwargs_test)

    rgbs_save = to8b(rgbs.detach().cpu().numpy()).reshape(len(i_train), H, W, 3)
    images_save = images
    # Save images to new directory.
    for i_img, n_img in enumerate(i_train):
        img_out_dir = os.path.join(new_data_dir, 'images_train', f'{n_img:03d}.png')
        img_out_dir2 = os.path.join(new_data_dir, 'images', f'{n_img:03d}.png')
        imageio.imwrite(img_out_dir, rgbs_save[i_img])
        imageio.imwrite(img_out_dir2, rgbs_save[i_img])

    for i_img in i_test:
        img_out_dir = os.path.join(new_data_dir, 'images', f'{i_img:03d}.png')
        imageio.imwrite(img_out_dir, images_save[i_img])
    
    # Save camera intrinsic and extrinsic to colmap format.
    # cameras.txt.
    with open(os.path.join(new_data_dir, 'sparse_learned', 'cameras.txt'), 'w') as file:
        file.write("1 SIMPLE_PINHOLE {} {} {} {} {}".format(hwf[1], hwf[0], hwf[2], hwf[1]//2, hwf[0]//2))
    # point3D.txt.
    with open(os.path.join(new_data_dir, 'sparse_learned', 'points3D.txt'), 'w') as file:
        pass
    # images.txt.
    # sc: scale factor of load_llff_data
    
    with open(os.path.join(new_data_dir, 'sparse_learned', 'images.txt'), 'w') as file:
        for i_count, i_img in enumerate(i_train):
            # convert rot_mat to quaternion.
            quat_i = poses_quat[i_img]
            trans_i = poses_trans[i_img]
            # i_img+1
            file.write("{} {} {} {} {} {} {} {} {} {}".format(i_count+1, \
                quat_i[0], quat_i[1], quat_i[2], quat_i[3], trans_i[0], trans_i[1], trans_i[2], 1, f'{i_img:03d}.png\n'))
            file.write("\n")

    # Gen train.txt containing filenames.
    with open(os.path.join(new_data_dir, 'train.txt'), 'w') as file:
        i_all = np.concatenate([i_train, i_test])
        i_all.sort()
        for i_name, img_name in enumerate(i_all):
            if i_name < len(i_all) - 1:
                file.write(f'{img_name:03d}.png'+'\n')
            else:
                file.write(f'{img_name:03d}.png')
    
    os.system("sh export_colmap.sh {} {} {} {}".format(new_data_dir, hwf[2], hwf[1]//2, hwf[0]//2))
    os.system('cp {} {}'.format(os.path.join(args.datadir, 'test.txt'), os.path.join(new_data_dir, 'test.txt')))

def render_only(args, nerf, images_train, poses_train, poses, render_poses, render_kwargs_test, hwf, K, basedir, expname, start):
    print('RENDER ONLY')
    with torch.no_grad():
        testsavedir = os.path.join(basedir, expname,
                                    f"renderonly"
                                    f"_{'test' if args.render_test else 'path'}"
                                    f"_{start:06d}")
        os.makedirs(testsavedir, exist_ok=True)
        print('test poses shape', render_poses.shape)

        dummy_num = ((len(poses) - 1) // args.num_gpu + 1) * args.num_gpu - len(poses)
        dummy_poses = torch.eye(3, 4).unsqueeze(0).expand(dummy_num, 3, 4).type_as(render_poses)
        print(f"Append {dummy_num} # of poses to fill all the GPUs")
        nerf.eval()
        rgbshdr, disps = nerf(
            hwf[0], hwf[1], K, args.chunk,
            poses=torch.cat([render_poses, dummy_poses], dim=0),
            render_kwargs=render_kwargs_test,
            render_factor=args.render_factor,
        )
        
        rgbshdr = rgbshdr[:len(rgbshdr) - dummy_num]
        rgbs = rgbshdr
        rgbs = to8b(rgbs.cpu().numpy()).reshape(-1, hwf[0], hwf[1], 3)

        # Compute nearest train_poses.
        pos_train = poses_train[:, :, 3]
        pos_render = render_poses[:, :, 3]
        dist = torch.cdist(pos_render.cpu(), torch.from_numpy(pos_train))
        i_min_dist = torch.argmin(dist, dim=1)
        images_nearest = images_train[i_min_dist]

        """
        disps = (1. - disps)
        disps = disps[:len(disps) - dummy_num].cpu().numpy()
        disps = to8b(disps / disps.max())
        """
        if args.render_test:
            for rgb_idx, rgb8 in enumerate(rgbs):
                imageio.imwrite(os.path.join(testsavedir, f'{rgb_idx:03d}.png'), rgb8)
                #imageio.imwrite(os.path.join(testsavedir, f'{rgb_idx:03d}_disp.png'), disps[rgb_idx])
        else:
            prefix = 'epi_' if args.render_epi else ''
            imageio.mimwrite(os.path.join(testsavedir, f'{prefix}video.mp4'), rgbs, fps=30, quality=9)
            imageio.mimwrite(os.path.join(testsavedir, f'{prefix}video_nearest.mp4'), images_nearest, fps=30, quality=9)
            #imageio.mimwrite(os.path.join(testsavedir, f'{prefix}video_disp.mp4'), disps, fps=30, quality=9)

        if args.render_test and args.render_multipoints:
            for pti in range(args.kernel_ptnum):
                nerf.eval()
                poses_num = len(poses) + dummy_num
                imgidx = torch.arange(poses_num, dtype=torch.long).to(render_poses.device).reshape(poses_num, 1)
                rgbs, weights = nerf(
                    hwf[0], hwf[1], K, args.chunk,
                    poses=torch.cat([render_poses, dummy_poses], dim=0),
                    render_kwargs=render_kwargs_test,
                    render_factor=args.render_factor,
                    render_point=pti,
                    images_indices=imgidx
                )
                rgbs = rgbs[:len(rgbs) - dummy_num]
                weights = weights[:len(weights) - dummy_num]
                rgbs = to8b(rgbs.cpu().numpy())
                weights = to8b(weights.cpu().numpy())
                for rgb_idx, rgb8 in enumerate(rgbs):
                    imageio.imwrite(os.path.join(testsavedir, f'{rgb_idx:03d}_pt{pti}.png'), rgb8)
                    imageio.imwrite(os.path.join(testsavedir, f'w_{rgb_idx:03d}_pt{pti}.png'), weights[rgb_idx])
                    
def eval_only(args, nerf, poses, render_kwargs_test, imagesf, i_test, H, W, K, basedir, expname, start, global_step):
    print('EVAL ONLY')
    with torch.no_grad():
        testsavedir = os.path.join(basedir, expname,
                                    f"eval"
                                    f"_{start:06d}")
        os.makedirs(testsavedir, exist_ok=True)
        print('eval indices', i_test)

        poses_eval = poses[i_test]
        render_poses = torch.from_numpy(poses_eval).cuda()

        dummy_num = ((len(poses_eval) - 1) // args.num_gpu + 1) * args.num_gpu - len(poses_eval)
        dummy_poses = torch.eye(3, 4).unsqueeze(0).expand(dummy_num, 3, 4).type_as(render_poses)
        #print(f"Append {dummy_num} # of poses to fill all the GPUs")
        nerf.eval()
        
        rgbs, _ = nerf(H, W, K, args.chunk, poses=torch.cat([render_poses, dummy_poses], dim=0), 
                                            render_kwargs=render_kwargs_test)
        
        rgbs = rgbs[:len(rgbs)-dummy_num].view(-1, H, W, 3)
        target_rgb_ldr = torch.from_numpy(imagesf[i_test]).cuda()
        
        # saving
        for rgb_idx, rgb in enumerate(rgbs):
                rgb8 = to8b(rgb.cpu().numpy())
                filename = os.path.join(testsavedir, f'{rgb_idx:03d}_pred.png')
                imageio.imwrite(filename, rgb8)

        for rgb_idx, rgb in enumerate(target_rgb_ldr):
                rgb8 = to8b(rgb.cpu().numpy())
                filename = os.path.join(testsavedir, f'{rgb_idx:03d}_gt.png')
                imageio.imwrite(filename, rgb8)

        # draw error map.
        for rgb_idx, rgb in enumerate(rgbs):
            rgb_pred = rgb.cpu()
            rgb_gt = target_rgb_ldr[rgb_idx].cpu()
            rgb_pred = rgb_pred.view(rgb_gt.shape)
            errormap = colorize(torch.abs(rgb_pred-rgb_gt).mean(dim=-1), range=(0., 1.))
            filename = os.path.join(testsavedir, f'{rgb_idx:03d}_errormap.png')
            imageio.imwrite(filename, to8b(errormap.numpy()))

        test_mse = compute_img_metric(rgbs, target_rgb_ldr, 'mse')
        test_psnr = compute_img_metric(rgbs, target_rgb_ldr, 'psnr')
        test_ssim = compute_img_metric(rgbs, target_rgb_ldr, 'ssim')
        test_lpips = compute_img_metric(rgbs, target_rgb_ldr, 'lpips')

        eval_file = os.path.join(testsavedir, 'metric_{:06d}.txt'.format(start))
 
        with open(eval_file, 'a') as outfile:
            outfile.write(f"iter{start}/globalstep{global_step}: MSE:{test_mse:.8f} PSNR:{test_psnr:.8f}"\
                            f" SSIM:{test_ssim:.8f} LPIPS:{float(test_lpips.squeeze()):.8f}\n")

        print('Saved test set')

def visualize_voxel_grid(args, nerf, poses, H, W, K, i_test, render_kwargs_test, basedir, expname):
    print('Visualize Voxel Grid')

    with torch.no_grad():
        voxelsavedir = os.path.join(basedir, expname)
        render_poses = torch.from_numpy(poses[i_test]).cuda()
        dummy_num = ((len(render_poses) - 1) // args.num_gpu + 1) * args.num_gpu - len(render_poses)
        dummy_poses = torch.eye(3, 4).unsqueeze(0).expand(dummy_num, 3, 4).type_as(render_poses)

        nerf.eval()
        rgbs, disps = nerf(H, W, K, args.chunk, poses=torch.cat([render_poses, dummy_poses], dim=0).cuda(),
                           render_kwargs=render_kwargs_test)

        rgbs = rgbs[:len(rgbs) - dummy_num]
        rgbs = rgbs.reshape(len(rgbs), H, W, 3)
        disps = disps.reshape(-1, H, W)[:len(disps) - dummy_num]

        rgbs = rgbs.detach().cpu().numpy()
        disps = disps.detach().cpu().numpy()
        voxel_grid_visualization(args, rgbs, disps, K, poses, nerf.plenoxel, H, W, i_test, img_dir=voxelsavedir)