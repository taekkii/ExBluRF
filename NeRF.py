import torch
import torch.nn as nn
from pytorch3d.transforms.se3 import se3_exp_map, se3_log_map
from run_nerf_helpers import *
import os
import imageio
import time
import svox2
import json

import plenoxel.llff_dataset as svox_llff
import plenoxel.utils 

import scipy.special


LLFF_OFFSET = 250

def init_linear_weights(m):
    if isinstance(m, nn.Linear):
        if m.weight.shape[0] in [2, 3]:
            nn.init.xavier_normal_(m.weight, 0.1)
        else:
            nn.init.xavier_normal_(m.weight)
        # nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)


class DSKnet(nn.Module):
    def __init__(self, num_img, poses, num_pt, kernel_hwindow, *, random_hwindow=0.25,
                 in_embed=3, random_mode='input', img_embed=32, spatial_embed=0, depth_embed=0,
                 num_hidden=3, num_wide=64, short_cut=False, pattern_init_radius=0.1,
                 isglobal=False, optim_trans=False, optim_spatialvariant_trans=False, learnable_pose=False,
                 recenter_traj=False):
        """
        num_img: number of image, used for deciding the view embedding
        poses: the original poses, used for generating new rays, len(poses) == num_img
        num_pt: number of sparse point, we use 5 in the paper
        kernel_hwindow: the size of physically equivalent blur kernel, the sparse points are bounded inside the blur kernel. 
                        Can be a very big number
        
        random_hwindow: in training, we randomly perturb the sparse point to model a smooth manifold
        random_mode: 'input' or 'output', it controls whether the random perturb is added to the input of DSK or output of DSK
        // the above two parameters do not have big impact on the results

        in_embed: embedding for the canonical kernel location
        img_embed: the length of the view embedding
        spatial_embed: embedding for the pixel location of the blur kernel inside an image
        depth_embed: (deprecated) the embedding for the depth of current rays
        
        num_hidden, num_wide, short_cut: control the structure of the MLP
        pattern_init_radius: the little gain add to the deform location described in Sec. 4.4
        isglobal: control whether the canonical kernel should be shared by all the input views or not, does not have big impact on the results
        optim_trans: whether to optimize the ray origin described in Sec. 4.3
        optim_spatialvariant_trans: whether to optimize the ray origin for each view or each kernel point. 
        """
        super().__init__()
        self.num_pt = num_pt
        self.num_img = num_img
        self.short_cut = short_cut
        self.kernel_hwindow = kernel_hwindow
        self.random_hwindow = random_hwindow  # about 1 pix
        self.random_mode = random_mode
        self.isglobal = isglobal

        self.recenter_traj = recenter_traj

        pattern_num = 1 if isglobal else num_img
        assert self.random_mode in ['input', 'output'], f"DSKNet::random_mode {self.random_mode} unrecognized, " \
                                                        f"should be input/output"
        
        self.register_buffer("poses", poses)

        self.register_parameter("pattern_pos",
                                nn.Parameter(torch.randn(pattern_num, num_pt, 2)
                                             .type(torch.float32) * pattern_init_radius, True))
        self.optim_trans = optim_trans
        self.optim_sv_trans = optim_spatialvariant_trans

        if optim_trans:
            self.register_parameter("pattern_trans",
                                    nn.Parameter(torch.zeros(pattern_num, num_pt, 2)
                                                 .type(torch.float32), True))

        if in_embed > 0:
            self.in_embed_fn, self.in_embed_cnl = get_embedder(in_embed, input_dim=2)
        else:
            self.in_embed_fn, self.in_embed_cnl = None, 0

        self.img_embed_cnl = img_embed

        if spatial_embed > 0:
            self.spatial_embed_fn, self.spatial_embed_cnl = get_embedder(spatial_embed, input_dim=2)
        else:
            self.spatial_embed_fn, self.spatial_embed_cnl = None, 0

        if depth_embed > 0:
            self.require_depth = True
            self.depth_embed_fn, self.depth_embed_cnl = get_embedder(depth_embed, input_dim=1)
        else:
            self.require_depth = False
            self.depth_embed_fn, self.depth_embed_cnl = None, 0

        in_cnl = self.in_embed_cnl + self.img_embed_cnl + self.depth_embed_cnl + self.spatial_embed_cnl
        out_cnl = 1 + 2 + 2 if self.optim_sv_trans else 1 + 2  # u, v, w or u, v, w, dx, dy
        hiddens = [nn.Linear(num_wide, num_wide) if i % 2 == 0 else nn.ReLU()
                   for i in range((num_hidden - 1) * 2)]
        # hiddens = [nn.Linear(num_wide, num_wide), nn.ReLU()] * num_hidden
        self.linears = nn.Sequential(
            nn.Linear(in_cnl, num_wide), nn.ReLU(),
            *hiddens,
        )
        self.linears1 = nn.Sequential(
            nn.Linear((num_wide + in_cnl) if short_cut else num_wide, num_wide), nn.ReLU(),
            nn.Linear(num_wide, out_cnl)
        )
        self.linears.apply(init_linear_weights)
        self.linears1.apply(init_linear_weights)
        if img_embed > 0:
            self.register_parameter("img_embed",
                                    nn.Parameter(torch.zeros(num_img, img_embed).type(torch.float32), True))
        else:
            self.img_embed = None

    def forward(self, H, W, K, rays, rays_info):
        """
        inputs: all input has shape (ray_num, cnl)
        outputs: output shape (ray_num, ptnum, 3, 2)  last two dim: [ray_o, ray_d]
        """
        img_idx = rays_info['images_idx'].squeeze(-1)
        # Set up view embedding.
        img_embed = self.img_embed[img_idx] if self.img_embed is not None else \
            torch.tensor([]).reshape(len(img_idx), self.img_embed_cnl)

        # Set up cannonical (initial) pose pattern.
        pt_pos = self.pattern_pos.expand(len(img_idx), -1, -1) if self.isglobal \
            else self.pattern_pos[img_idx]
        pt_pos = torch.tanh(pt_pos) * self.kernel_hwindow

        if self.random_hwindow > 0 and self.random_mode == "input":
            random_pos = torch.randn_like(pt_pos) * self.random_hwindow
            pt_pos = pt_pos + random_pos

        input_pos = pt_pos  # the first point is the reference point
        if self.in_embed_fn is not None:
            pt_pos = pt_pos * (np.pi / self.kernel_hwindow)
            pt_pos = self.in_embed_fn(pt_pos)

        img_embed_expand = img_embed[:, None].expand(len(img_embed), self.num_pt, self.img_embed_cnl)
        x = torch.cat([pt_pos, img_embed_expand], dim=-1)

        rays_x, rays_y = rays_info['rays_x'], rays_info['rays_y']
        if self.spatial_embed_fn is not None:
            spatialx = rays_x / (W / 2 / np.pi) - np.pi
            spatialy = rays_y / (H / 2 / np.pi) - np.pi  # scale 2pi to match the freq in the embedder
            spatial = torch.cat([spatialx, spatialy], dim=-1)
            spatial = self.spatial_embed_fn(spatial)
            spatial = spatial[:, None].expand(len(img_idx), self.num_pt, self.spatial_embed_cnl)
            x = torch.cat([x, spatial], dim=-1)

        if self.depth_embed_fn is not None:
            depth = rays_info['ray_depth']
            depth = depth * np.pi  # TODO: please always check that the depth lies between [0, 1)
            depth = self.depth_embed_fn(depth)
            depth = depth[:, None].expand(len(img_idx), self.num_pt, self.depth_embed_cnl)
            x = torch.cat([x, depth], dim=-1)

        # forward
        x1 = self.linears(x)
        x1 = torch.cat([x, x1], dim=-1) if self.short_cut else x1
        x1 = self.linears1(x1)

        delta_trans = None
        if self.optim_sv_trans:
            delta_trans, delta_pos, weight = torch.split(x1, [2, 2, 1], dim=-1)
        else:
            delta_pos, weight = torch.split(x1, [2, 1], dim=-1)

        if self.optim_trans:
            delta_trans = self.pattern_trans.expand(len(img_idx), -1, -1) if self.isglobal \
                else self.pattern_trans[img_idx]

        if delta_trans is None:
            delta_trans = torch.zeros_like(delta_pos)

        delta_trans = delta_trans * 0.01
        new_rays_xy = delta_pos + input_pos
        weight = torch.softmax(weight[..., 0], dim=-1)
        if self.recenter_traj:
            # Normalize delta trans.
            delta_trans_center = delta_trans[:, 0]
            delta_trans = delta_trans - delta_trans_center.unsqueeze(-2)

        if self.random_hwindow > 0 and self.random_mode == 'output':
            raise NotImplementedError(f"{self.random_mode} for self.random_mode is not implemented")

        poses = self.poses[img_idx]

        # get rays from offsetted pt position
        rays_x = (rays_x - K[0, 2] + new_rays_xy[..., 0]) / K[0, 0]
        rays_y = -(rays_y - K[1, 2] + new_rays_xy[..., 1]) / K[1, 1]
        dirs = torch.stack([rays_x - delta_trans[..., 0],
                            rays_y - delta_trans[..., 1],
                            -torch.ones_like(rays_x)], -1)

        # Rotate ray directions from camera frame to the world frame
        rays_d = torch.sum(dirs[..., None, :] * poses[..., None, :3, :3],
                           -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]

        # Translate camera frame's origin to the world frame. It is the origin of all rays.
        translation = torch.stack([
            delta_trans[..., 0],
            delta_trans[..., 1],
            torch.zeros_like(rays_x),
            torch.ones_like(rays_x)
        ], dim=-1)
        rays_o = torch.sum(translation[..., None, :] * poses[:, None], dim=-1)
        # rays_o = poses[..., None, :3, -1].expand_as(rays_d)

        align = new_rays_xy[:, 0, :].abs().mean()
        align += (delta_trans[:, 0, :].abs().mean() * 10)
        return torch.stack([rays_o, rays_d], dim=-1), weight, align
    def compute_poses_trajectory(self):
        return self.poses[:, None].repeat(1,  self.num_pt, 1, 1)

class CTKNet(nn.Module):
    def __init__(self, num_imgs, num_pts, poses, order_curve=5, learnable_pose=False, recenter_traj=False):
        """
        Learns blur Kernel of multi-view images formluated (parameterized) by camera trajectory.

        Input:
        - num_imgs: the number of training view.
        - num_pts: the number of sampled camera poses on the trajectory.
        - poses: camera poses of training view from colmap.
        - order_curve: Order of bezier curve.

        Output:
        - ray_o: ray origin.
        - ray_d: ray direction.
        - weights: weigts for each samples on camera trajectory.
        - loss_align: (disabled)
        """
        super().__init__()
        self.num_imgs = num_imgs
        self.num_pts = num_pts
        self.order_curve = order_curve
        self.require_depth = False
        self.recenter_traj = recenter_traj
        # The number of sample on kernel must be odd.
        assert self.num_pts % 2 == 1
        
        self.register_buffer("poses_init", poses)
        # 6 DoF camera motion, parameterized by start and end points.
        poses_se3 = self.init_poses_se3(poses)

        self.register_parameter("control_points", 
                                nn.Parameter(torch.zeros((self.num_imgs, self.order_curve+1, 6)).float(), True))
        
        self.compute_bezier_coefficient_mat(num_pts, order_curve)


    def compute_bezier_coefficient_mat(self, num_points, order_curve):
        # Uniformly sample n points on t: [0, 1].
        t = torch.linspace(0, 1, num_points)
        
        binom_coeff = [scipy.special.binom(order_curve, k) for k in range(order_curve+1)]
        # Build coefficient matrix.
        bezier_coeff = []
        for i in range(order_curve+1):
            coeff_i = binom_coeff[i] * torch.pow(1-t, order_curve-i) * torch.pow(t, i)
            bezier_coeff.append(coeff_i)

        bezier_coeff = torch.stack(bezier_coeff, dim=-1)

        self.register_buffer("bezier_coeff", bezier_coeff)

    def init_poses_se3(self, poses):
        poses_4x4 = torch.zeros(self.num_imgs, 4, 4)
        poses_4x4[:, :3, :4] = poses
        poses_4x4[:, 3, 3] = 1.

        poses_4x4_T = poses_4x4.transpose(-1, -2)

        log_map = se3_log_map(poses_4x4_T)

        return log_map

    def compute_poses_trajectory(self):
        poses_traj = self.compute_pose_on_bezier_curve(self.bezier_coeff, self.control_points)

        if self.recenter_traj:
            poses_traj = self.recenter_trajectory(poses_traj, self.num_pts)
        else:
            poses_init = torch.zeros_like(poses_traj)
            poses_init[:, :, :3, :4] = self.poses_init[:, None]
            poses_init[:, :,3, 3] = 1.
        
            poses_traj = torch.matmul(poses_traj, poses_init)
        
        return poses_traj

    def recenter_trajectory(self, poses_traj, num_pts):

        poses_center = poses_traj[:, num_pts//2]
        poses_center_inv = self.inv4x4(poses_center)

        poses_init = torch.zeros_like(poses_center)
        poses_init[:, :3, :4] = self.poses_init
        poses_init[:, 3, 3] = 1.
        
        poses_traj_recenter =  torch.matmul(torch.matmul(poses_center_inv[:, None], poses_traj), poses_init[:, None])
    
        return poses_traj_recenter

    def compute_pose_on_bezier_curve(self, coeff, control_points):
        """
        coeff (n_img, n_pts, order_curve+1): bezier coefficient
        control_points (n_imgs, order_curve+1, 6): control_points on Lie-algebra (se(3)) space.
        """
        control_points_repeat = control_points.unsqueeze(1).repeat(1, self.num_pts, 1, 1)
        weighted_control_points = coeff.unsqueeze(-1) * control_points_repeat

        weighted_control_points_mat = self.se3_to_mat(weighted_control_points, self.num_pts)

        delta_poses_mat = torch.eye(4)[None, None].repeat(self.num_imgs, self.num_pts, 1, 1)

        for i in range(self.order_curve+1):
            delta_poses_mat = torch.matmul(weighted_control_points_mat[:, :, i], delta_poses_mat)

        return delta_poses_mat

    def inv4x4(self, mat):
        n_points = mat.shape[0]
        mat_inv = torch.eye(4)[None].repeat(n_points , 1, 1)
        
        rot_T = mat[:, :3, :3].transpose(-2, -1)
        trans = mat[..., :3, 3]

        trans_inv = - torch.matmul(rot_T, trans[..., None])

        mat_inv[:, :3, :3] = rot_T
        mat_inv[:, :3, 3] = trans_inv.squeeze(-1)

        return mat_inv

    def mat_to_se3(self, mat):
        mat_batch = mat.reshape(self.num_imgs*(self.order_curve+1), 4, 4)
        mat_batch_T = mat_batch.transpose(-1, -2)

        log_map_batch_T = se3_log_map(mat_batch_T)

        log_map = log_map_batch_T.reshape(self.num_imgs, self.order_curve+1, 6)

        return log_map

    def se3_to_mat(self, se3, num_pts):
        se3_batch = se3.reshape(self.num_imgs*num_pts*(self.order_curve+1), 6)
        #se3_batch = se3.reshape(self.num_imgs*num_pts, 6)
        mat_batch_T = se3_exp_map(se3_batch)
        
        #mat_T = mat_batch_T.reshape(self.num_imgs, num_pts, 4, 4)
        mat_T = mat_batch_T.reshape(self.num_imgs, num_pts, self.order_curve+1, 4, 4)
        mat = mat_T.transpose(-1, -2)
        
        return mat

    def forward(self, H, W, K, rays, rays_info, dataset_type=None):
        # Get image index.
        img_idx = rays_info['images_idx'].squeeze(-1)
        # Get pose trajectory of current image.
        poses_traj = self.compute_poses_trajectory()
        # Gather camera pose by image index.
        poses_rays = poses_traj[img_idx] 
        
        # Compute rays_o, rays_d
        rays_x, rays_y = rays_info['rays_x'], rays_info['rays_y']
        
        rays_x = (rays_x - K[0, 2]) / K[0, 0]
        rays_y = -(rays_y - K[1, 2]) / K[1, 1]
        dirs = torch.stack([rays_x, rays_y, -torch.ones_like(rays_x)], dim=-1)

        dirs = dirs.unsqueeze(1).repeat(1, self.num_pts, 1, 1)
        rays_d = torch.sum(dirs * poses_rays[:, :, :3, :3], dim=-1)
        # Ray origin is camera pose itself.
        rays_o = poses_rays[:, :, :3, 3]
        ## No weights (uniform).
        weights = torch.ones(poses_rays.shape[0], self.num_pts).float() / self.num_pts

        return torch.stack([rays_o, rays_d], dim=-1), weights, None

    def export_trajectory(self, num_points=100):
        t = torch.linspace(0, 1, num_points)

        binom_coeff = [scipy.special.binom(self.order_curve, k) for k in range(self.order_curve+1)]

        # Build coefficient matrix.
        bezier_coeff_dense = []
        for i in range(self.order_curve+1):
            coeff_i = binom_coeff[i] * torch.pow(1-t, self.order_curve-i) * torch.pow(t, i)
            bezier_coeff_dense.append(coeff_i)

        bezier_coeff_dense = torch.stack(bezier_coeff_dense, dim=-1)
        with torch.no_grad():
            control_points_se3 = self.control_points
            
            control_points_repeat = control_points_se3.unsqueeze(1).repeat(1, num_points, 1, 1)
            weighted_control_points = bezier_coeff_dense.unsqueeze(-1) * control_points_repeat

            curve_points_se3 = torch.sum(weighted_control_points, dim=2)
            
            weighted_control_points_mat = self.se3_to_mat(weighted_control_points, num_points)

            delta_poses_mat = torch.eye(4)[None, None].repeat(self.num_imgs, num_points, 1, 1)

            for i in range(self.order_curve+1):
                delta_poses_mat = torch.matmul(weighted_control_points_mat[:, :, i], delta_poses_mat)
            
            if self.recenter_traj:
                poses_traj = self.recenter_trajectory(delta_poses_mat, num_points)
            else:
                poses_init = torch.zeros_like(delta_poses_mat)
                poses_init[:, :, :3, :4] = self.poses_init[:, None]
                poses_init[:, :,3, 3] = 1.
            
                poses_traj = torch.matmul(delta_poses_mat, poses_init)

        return poses_traj


class NeRFAll(nn.Module):
    def __init__(self, args, kernelsnet=None,reso_id=0, nerf_init_kwargs={}):
        super().__init__()
        self.args = args
        self.embed_fn, self.input_ch = get_embedder(args.multires, args.i_embed)
        self.input_ch_views = 0
        self.kernelsnet = kernelsnet
        self.embeddirs_fn = None
        self.dataset_type = args.dataset_type
        self.nerf_init_kwargs = nerf_init_kwargs
        
        if args.use_viewdirs:
            self.embeddirs_fn, self.input_ch_views = get_embedder(args.multires_views, args.i_embed)

        self.output_ch = 5 if args.N_importance > 0 else 4

        skips = [4]
        # Plenoxel option
        if args.architecture == 'plenoxel':
            
            self.make_plenoxel(args, reso_id)
        else:
            self.mlp_coarse = NeRF(
                D=args.netdepth, W=args.netwidth,
                input_ch=self.input_ch, output_ch=self.output_ch, skips=skips,
                input_ch_views=self.input_ch_views, use_viewdirs=args.use_viewdirs)

            self.mlp_fine = None
            if args.N_importance > 0:
                self.mlp_fine = NeRF(
                    D=args.netdepth_fine, W=args.netwidth_fine,
                    input_ch=self.input_ch, output_ch=self.output_ch, skips=skips,
                    input_ch_views=self.input_ch_views, use_viewdirs=args.use_viewdirs)

            activate = {'relu': torch.relu, 'sigmoid': torch.sigmoid, 'exp': torch.exp, 'none': lambda x: x,
                        'sigmoid1': lambda x: 1.002 / (torch.exp(-x) + 1) - 0.001,
                        'softplus': lambda x: nn.Softplus()(x - 1)}
            self.rgb_activate = activate[args.rgb_activate]
            self.sigma_activate = activate[args.sigma_activate]
        self.tonemapping = ToneMapping(args.tone_mapping_type)
    
    def make_plenoxel(self, args, reso_id):
        assert torch.cuda.is_available()
        # if plenoxel already exists
        if self.is_plenoxel(): 
            print("Destroying old plenoxel...")
            del self.plenoxel
        # load resolution
        reso_list = json.loads(args.reso)
        wid = self.nerf_init_kwargs['wid']
        hei = self.nerf_init_kwargs['hei']

        if 'bound_config' in self.nerf_init_kwargs:
            radx = self.nerf_init_kwargs['bound_config']['rx']
            rady = self.nerf_init_kwargs['bound_config']['ry']
            cx = self.nerf_init_kwargs['bound_config']['cx']
            cy = self.nerf_init_kwargs['bound_config']['cy']
        else:
            cx, cy = 0.0, 0.0
            radx = 1 + 2 * LLFF_OFFSET / self.nerf_init_kwargs['wid']
            rady = 1 + 2 * LLFF_OFFSET / self.nerf_init_kwargs['hei']
        scene_center = [cx, cy, 0.0]
        scene_radius = [radx,rady,1.0]
        use_sphere_bound = False
        print("[PLenoxel] Center:", scene_center)
        print("[Plenoxel] Radius:",scene_radius)
        if 'bound_config' in self.nerf_init_kwargs:
            print("[Plenoxel] Focal:", self.nerf_init_kwargs['bound_config']['ndc_focal'])
        else:
            print("[Plenoxel] Focal: Default focal")
        self.plenoxel = svox2.SparseGrid(reso=reso_list[reso_id],
                                            center=scene_center,
                                            radius=scene_radius,
                                            use_sphere_bound=use_sphere_bound,
                                            basis_dim=9,
                                            use_z_order=True,
                                            #background_nlayers=args.background_nlayers,
                                            #background_reso=args.background_reso,
                                            device='cuda',
                                            basis_type=svox2.__dict__['BASIS_TYPE_SH'])
        #initialize sh coefficients/density
        self.plenoxel.sh_data.data[:] = 0.0
        self.plenoxel.density_data.data[:] = 0.0 if args.lr_fg_begin_step > 0 else args.init_sigma

        self.plenoxel.requires_grad_(True)
        
        plenoxel.utils.setup_render_opts(self.plenoxel.opt, args)

        print('Render options', self.plenoxel.opt)
    
    def mlpforward(self, inputs, viewdirs, mlp, netchunk=1024 * 64):
        """Prepares inputs and applies network 'fn'.
            """
        inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
        embedded = self.embed_fn(inputs_flat)

        if viewdirs is not None:
            input_dirs = viewdirs[:, None].expand(inputs.shape)
            input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
            embedded_dirs = self.embeddirs_fn(input_dirs_flat)
            embedded = torch.cat([embedded, embedded_dirs], -1)

        # batchify execution
        if netchunk is None:
            outputs_flat = mlp(embedded)
        else:
            outputs_flat = torch.cat([mlp(embedded[i:i + netchunk]) for i in range(0, embedded.shape[0], netchunk)], 0)

        outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
        return outputs

    def raw2outputs(self, raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
        """Transforms model's predictions to semantically meaningful values.
        Args:
            raw: [num_rays, num_samples along ray, 4]. Prediction from model.
            z_vals: [num_rays, num_samples along ray]. Integration time.
            rays_d: [num_rays, 3]. Direction of each ray.
        Returns:
            rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
            disp_map: [num_rays]. Disparity map. Inverse of depth map.
            acc_map: [num_rays]. Sum of weights along each ray.
            weights: [num_rays, num_samples]. Weights assigned to each sampled color.
            depth_map: [num_rays]. Estimated distance to object.
        """

        def raw2alpha(raw_, dists_, act_fn):
            alpha_ = - torch.exp(-act_fn(raw_) * dists_) + 1.
            return torch.cat([alpha_, torch.ones_like(alpha_[:, 0:1])], dim=-1)

        dists = z_vals[..., 1:] - z_vals[..., :-1]  # [N_rays, N_samples - 1]
        dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

        rgb = self.rgb_activate(raw[..., :3])
        noise = 0.
        if raw_noise_std > 0.:
            noise = torch.randn_like(raw[..., :-1, 3]) * raw_noise_std

            # Overwrite randomly sampled data if pytest
            if pytest:
                np.random.seed(0)
                noise = np.random.rand(*list(raw[..., 3].shape)) * raw_noise_std
                noise = torch.tensor(noise)

        density = self.sigma_activate(raw[..., :-1, 3] + noise)
        if not self.training and self.args.render_rmnearplane > 0:
            mask = z_vals[:, 1:]
            mask = mask > self.args.render_rmnearplane / 128
            mask = mask.type_as(density)
            density = mask * density

        alpha = - torch.exp(- density * dists) + 1.
        alpha = torch.cat([alpha, torch.ones_like(alpha[:, 0:1])], dim=-1)

        # alpha = raw2alpha(raw[..., :-1, 3] + noise, dists, act_fn=self.sigma_activate)  # [N_rays, N_samples]
        # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
        weights = alpha * \
                  torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), - alpha + (1. + 1e-10)], -1), -1)[:, :-1]

        rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]
        depth_map = torch.sum(weights * z_vals, -1)

        # disp_map = 1. / torch.clamp_min(depth_map, 1e-10)
        acc_map = torch.sum(weights, -1)

        if white_bkgd:
            rgb_map = rgb_map + (1. - acc_map[..., None])

        return rgb_map, density, acc_map, weights, depth_map
    
    def render_rays(self,
                    ray_batch,
                    N_samples,
                    retraw=False,
                    lindisp=False,
                    perturb=0.,
                    N_importance=0,
                    white_bkgd=False,
                    raw_noise_std=0.,
                    pytest=False):
        """Volumetric rendering.
        Args:
          ray_batch: array of shape [batch_size, ...]. All information necessary
            for sampling along a ray, including: ray origin, ray direction, min
            dist, max dist, and unit-magnitude viewing direction.
          N_samples: int. Number of different times to sample along each ray.
          retraw: bool. If True, include model's raw, unprocessed predictions.
          lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
          perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
            random points in time.
          N_importance: int. Number of additional times to sample along each ray.
            These samples are only passed to network_fine.
          white_bkgd: bool. If True, assume a white background.
          raw_noise_std: ...
          verbose: bool. If True, print more debugging info.
        """
        N_rays = ray_batch.shape[0]
        rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]  # [N_rays, 3] each
        viewdirs = ray_batch[:, -3:] if ray_batch.shape[-1] > 8 else None
        bounds = torch.reshape(ray_batch[..., 6:8], [-1, 1, 2])
        near, far = bounds[..., 0], bounds[..., 1]  # [-1,1]

        t_vals = torch.linspace(0., 1., steps=N_samples).type_as(rays_o)
        if not lindisp:
            z_vals = near * (1. - t_vals) + far * (t_vals)
        else:
            z_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * (t_vals))

        z_vals = z_vals.expand([N_rays, N_samples])

        if perturb > 0.:
            # get intervals between samples
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape).type_as(rays_o)

            # Pytest, overwrite u with numpy's fixed random numbers
            if pytest:
                np.random.seed(0)
                t_rand = np.random.rand(*list(z_vals.shape))
                t_rand = torch.tensor(t_rand)

            z_vals = lower + (upper - lower) * t_rand

        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [N_rays, N_samples, 3]

        #     raw = run_network(pts)
        raw = self.mlpforward(pts, viewdirs, self.mlp_coarse)
        rgb_map, density_map, acc_map, weights, depth_map = self.raw2outputs(raw, z_vals, rays_d, raw_noise_std,
                                                                             white_bkgd, pytest=pytest)

        if N_importance > 0:
            rgb_map_0, depth_map_0, acc_map_0, density_map0 = rgb_map, depth_map, acc_map, density_map

            z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            z_samples = sample_pdf(z_vals_mid, weights[..., 1:-1], N_importance, det=(perturb == 0.), pytest=pytest)
            z_samples = z_samples.detach()

            z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
            pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :,
                                                                None]  # [N_rays, N_samples + N_importance, 3]

            mlp = self.mlp_coarse if self.mlp_fine is None else self.mlp_fine
            raw = self.mlpforward(pts, viewdirs, mlp)

            rgb_map, density_map, acc_map, weights, depth_map = self.raw2outputs(raw, z_vals, rays_d, raw_noise_std,
                                                                                 white_bkgd, pytest=pytest)

        ret = {'rgb_map': rgb_map, 'depth_map': depth_map, 'acc_map': acc_map, 'density_map': density_map}
        if retraw:
            ret['raw'] = raw
        if N_importance > 0:
            ret['rgb0'] = rgb_map_0
            ret['depth0'] = depth_map_0
            ret['acc0'] = acc_map_0
            ret['density0'] = density_map0
            ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]

        for k in ret:
            if torch.isnan(ret[k]).any():
                print(f"! [Numerical Error] {k} contains nan.")
            if torch.isinf(ret[k]).any():
                print(f"! [Numerical Error] {k} contains inf.")
        return ret

    def forward(self, H, W, K, chunk=1024 * 32, rays=None, rays_info=None, poses=None, aggregate='mean', **kwargs):
        """
        render rays or render poses, rays and poses should atleast specify one
        calling model.train() to render rays, where rays, rays_info, should be specified
        calling model.eval() to render an image, where poses should be specified

        optional args:
        force_naive: when True, will only run the naive NeRF, even if the kernelsnet is specified

        """
        # training
        if self.training:
            assert rays is not None, "Please specify rays when in the training mode"

            self.force_baseline = kwargs.pop("force_naive", True)
            self.freeze_kernel = kwargs.pop("freeze_kernel", True)
            # kernel mode, run multiple rays to get result of one ray
            if self.kernelsnet is not None and not self.force_baseline:
                if self.kernelsnet.require_depth:
                    with torch.no_grad():
                        rgb, depth, acc, extras = self.render(H, W, K, chunk, rays, **kwargs)
                        rays_info["ray_depth"] = depth[:, None]

                #if not self.freeze_kernel:
                #    new_rays, weight, align_loss = self.kernelsnet(H, W, K, rays, rays_info)
                #else:
                #    with torch.no_grad():
                #        new_rays, weight, align_loss = self.kernelsnet(H, W, K, rays, rays_info)
                new_rays, weight, align_loss = self.kernelsnet(H, W, K, rays, rays_info)
                
                ray_num, pt_num = new_rays.shape[:2]

                rgb, depth, _ , extras = self.render(H, W, K, chunk, new_rays.reshape(-1, 3, 2), weight=weight, n_pts=pt_num,**kwargs)
                rgb_pts = rgb.reshape(ray_num, pt_num, 3)
                
                # Decide to learn weight to aggregate blur kernel outputs, or to simply calculate average
                if aggregate == 'weight':
                    rgb = torch.sum(rgb_pts * weight[..., None], dim=1)
                else:
                    rgb = rgb_pts.mean(dim=1)
                rgb = self.tonemapping(rgb)
                
                # This part is only for Vanilla NeRF (coarse-to-fine)
                if extras.get('rgb0',None) is not None:
                    rgb0_pts = extras['rgb0'].reshape(ray_num, pt_num, 3)
                    rgb0 = torch.sum(rgb0_pts * weight[..., None], dim=1)
                    rgb0 = self.tonemapping(rgb0)
                else: rgb0 = None

                other_loss = {}
                # compute align loss, some priors of the ray pattern
                # ========================
                if align_loss is not None:
                    other_loss["align"] = align_loss.reshape(1, 1)
                return rgb, rgb0, other_loss
            else:
                rgb, depth, acc, extras = self.render(H, W, K, chunk, rays, **kwargs)
                rgb0 = extras['rgb0']
                # rgb0: rgb output of coarse network.
                
                # tonemapping: gamma function
                if rgb0 is not None:
                    return self.tonemapping(rgb), self.tonemapping(rgb0), {} 
                else:
                    return self.tonemapping(rgb), None, {}
        # evaluation
        else:
            if poses is not None:
                #assert poses is not None, "Please specify poses when in the eval model"
                if "render_point" in kwargs.keys():
                    rgbs, depths, weights = self.render_subpath(H, W, K, chunk, poses, **kwargs)
                    depths = weights * 2
                else:
                    rgbs, depths = self.render_path(H, W, K, chunk, poses, **kwargs)
                return self.tonemapping(rgbs), depths
            else:
                rgbs = self.render_subpath_blurred(H, W, K, chunk, **kwargs)

                return rgbs
   
    def render(self, H, W, K, chunk, rays=None, c2w=None, ndc=True,
               near=0., far=1.,
               use_viewdirs=False, c2w_staticcam=None, weight=None, n_pts=1,
               **kwargs):  # the render function
        """Render rays
            Args:
              H: int. Height of image in pixels.
              W: int. Width of image in pixels.
              focal: float. Focal length of pinhole camera.
              chunk: int. Maximum number of rays to process simultaneously. Used to
                control maximum memory usage. Does not affect final results.
              rays: array of shape [2, batch_size, 3]. Ray origin and direction for
                each example in batch.
              c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
              ndc: bool. If True, represent ray origin, direction in NDC coordinates.
              near: float or array of shape [batch_size]. Nearest distance for a ray.
              far: float or array of shape [batch_size]. Farthest distance for a ray.
              use_viewdirs: bool. If True, use viewing direction of a point in space in model.
              c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for
               camera while using other c2w argument for viewing directions.
              n_pts: the number of points on blur kernel.
            Returns:
              rgb_map: [batch_size, 3]. Predicted RGB values for rays.
              disp_map: [batch_size]. Disparity map. Inverse of depth.
              acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
              extras: dict with everything returned by render_rays().
            """
        rays_o, rays_d = rays[..., 0], rays[..., 1]
        if use_viewdirs:
            # provide ray directions as input
            viewdirs = rays_d
            if c2w_staticcam is not None:
                # special case to visualize effect of viewdirs
                rays_o, rays_d = get_rays(H, W, K, c2w_staticcam)
                raise NotImplementedError
            viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
            viewdirs = torch.reshape(viewdirs, [-1, 3]).float()

        sh = rays_d.shape  # [..., 3]
        if ndc:
            # for forward facing scenes
            if 'bound_config' in self.nerf_init_kwargs:
                ndc_focal = self.nerf_init_kwargs['bound_config']['ndc_focal']
            else:
                ndc_focal = K[0,0]
            rays_o, rays_d = ndc_rays(H, W, ndc_focal, 1., rays_o, rays_d)

        # Create ray batch
        rays_o = torch.reshape(rays_o, [-1, 3]).float()
        rays_d = torch.reshape(rays_d, [-1, 3]).float()
        #if plenoxel is being used
        if self.is_plenoxel():
            rays_d = rays_d / rays_d.norm(dim=-1).view(-1,1)
            # make plenoxel-style ray class
            svox2_rays = svox2.Rays(rays_o.contiguous(), rays_d.contiguous())
            beta_loss = kwargs.get('beta_loss', 0.)
            sparsity_loss = kwargs.get('sparsity_loss', 0.)
            rgb_map = self.plenoxel.volume_render(svox2_rays, beta_loss=beta_loss, sparsity_loss=sparsity_loss)
            if self.training:
                depth_map = None
            else:
                depth_map = self.plenoxel.volume_render_depth(svox2_rays, None)
            return rgb_map, depth_map , None , {'rgb0':None}
        else:
            near, far = near * torch.ones_like(rays_d[..., :1]), far * torch.ones_like(rays_d[..., :1])
            rays = torch.cat([rays_o, rays_d, near, far], -1)
            if use_viewdirs:
                rays = torch.cat([rays, viewdirs], -1)

            # Batchfy and Render and reshape
            all_ret = {}
            for i in range(0, rays.shape[0], chunk):
                ret = self.render_rays(rays[i:i + chunk], **kwargs)
                for k in ret:
                    if k not in all_ret:
                        all_ret[k] = []
                    all_ret[k].append(ret[k])
            all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}

            for k in all_ret:
                k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
                all_ret[k] = torch.reshape(all_ret[k], k_sh)

            k_extract = ['rgb_map', 'depth_map', 'acc_map']
            ret_list = [all_ret[k] for k in k_extract]
            ret_dict = {k: all_ret[k] for k in all_ret if k not in k_extract}
        return ret_list + [ret_dict]

    def render_path(self, H, W, K, chunk, render_poses, render_kwargs, render_factor=0, ):
        """
        render image specified by the render_poses
        """
        if render_factor != 0:
            # Render downsampled for speed
            H = H // render_factor
            W = W // render_factor

        rgbs = []
        depths = []

        t = time.time()
        for i, c2w in enumerate(render_poses):
            print(i, time.time() - t)
            t = time.time()
            
            rays = get_rays(H, W, K, c2w)
            rays = torch.stack(rays, dim=-1)
            rgb, depth, acc, extras = self.render(H, W, K, chunk=chunk, rays=rays, c2w=c2w[:3, :4], **render_kwargs)

            rgbs.append(rgb)
            depths.append(depth)
            if i == 0:
                print(rgb.shape, depth.shape)

        rgbs = torch.stack(rgbs, 0)
        depths = torch.stack(depths, 0)

        return rgbs, depths

    def render_subpath(self, H, W, K, chunk, render_poses, render_point, images_indices, render_kwargs,
                       render_factor=0):
        if render_factor != 0:
            # Render downsampled for speed
            H = H // render_factor
            W = W // render_factor

        rgbs = []
        depths = []
        weights = []

        t = time.time()

        rayx, rayy = torch.meshgrid(torch.linspace(0, W - 1, W),
                                    torch.linspace(0, H - 1, H))
        rayx = rayx.t().reshape(-1, 1) + HALF_PIX
        rayy = rayy.t().reshape(-1, 1) + HALF_PIX

        for imgidx, c2w in zip(images_indices, render_poses):

            i = int(imgidx.item())
            print(i, time.time() - t)
            t = time.time()
            
            rays = get_rays(H, W, K, c2w)

            rays = torch.stack(rays, dim=-1).reshape(H * W, 3, 2)

            rays_info = {}

            if self.kernelsnet.require_depth:
                with torch.no_grad():
                    rgb, depth, acc, extras = self.render(H, W, K, chunk, rays, **render_kwargs)
                    rays_info["ray_depth"] = depth[..., None]

            i = i if i < self.kernelsnet.num_img else 1
            rays_info["images_idx"] = torch.ones_like(rays[:, 0:1, 0]).type(torch.long) * i
            rays_info["rays_x"] = rayx
            rays_info["rays_y"] = rayy

            new_rays, weight, _ = self.kernelsnet(H, W, K, rays, rays_info) #, self.dataset_type)

            new_rays = new_rays[:, render_point]
            weight = weight[:, render_point]
            
            rgb, depth, acc, extras = self.render(H, W, K, chunk=chunk, rays=new_rays.reshape(-1, 3, 2),
                                                c2w=c2w[:3, :4], **render_kwargs)

            rgbs.append(rgb.reshape(H, W, 3))
            #temporarily disables depth for plenoxel
            if not self.is_plenoxel():
                depths.append(depth.reshape(H, W))
                weights.append(weight.reshape(H, W))
                if i == 0:
                    print(rgb.shape, depth.shape)

        rgbs = torch.stack(rgbs, 0)
        
        #temporarily disables depth for plenoxel
        depths = torch.stack(depths, 0) if not self.is_plenoxel() else None
        weights = torch.stack(weights, 0) if not self.is_plenoxel() else None

        return rgbs, depths, weights
    
    def render_subpath_blurred(self, H, W, K, chunk, images_indices, render_kwargs, render_factor=0):
        
        # GUARD
        if self.kernelsnet is None:
            print("render_subpath_blurred: no kernelsnet detected. Ignoring...")
            return
        
        if render_factor != 0:
            # Render downsampled for speed
            H = H // render_factor
            W = W // render_factor

        t = time.time()

        rayx, rayy = torch.meshgrid(torch.linspace(0, W - 1, W),
                                    torch.linspace(0, H - 1, H))
        rayx = rayx.t().reshape(-1, 1) + HALF_PIX
        rayy = rayy.t().reshape(-1, 1) + HALF_PIX

        rgbs = []

        for imgidx in images_indices:
            print(imgidx.item(), time.time() - t)
            t = time.time()
            
            rays_info = {
                'images_idx': imgidx[None].repeat(rayx.shape[0], 1),
                'rays_x': rayx,
                'rays_y': rayy,
            }

            new_rays, weight, _ = self.kernelsnet(H, W, K, None, rays_info) #, self.dataset_type)

            ray_num, pt_num = new_rays.shape[:2]

            rgb, depth, _ , extras = self.render(H, W, K, chunk, new_rays.reshape(-1, 3, 2), weight=weight, n_pts=pt_num, **render_kwargs)

            rgb_pts = rgb.reshape(ray_num, pt_num, 3)

            # time2 = time.time()
            rgb = torch.sum(rgb_pts * weight[..., None], dim=1)
            rgb = self.tonemapping(rgb)

            rgbs.append(rgb.reshape(H, W, 3))

        rgbs = torch.stack(rgbs, 0)
        
        return rgbs
        
    def is_plenoxel(self):
        return hasattr(self,'plenoxel')
    
    