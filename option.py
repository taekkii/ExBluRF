

import configargparse
from typing import Optional
def config_parser(parser:Optional[configargparse.ArgumentParser]=None):
    if parser is None:
        parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')
    parser.add_argument("--expname", type=str,
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/', required=True,
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, required=True,
                        help='input data directory')
    parser.add_argument("--datadownsample", type=float, default=-1,
                        help='if downsample > 0, means downsample the image to scale=datadownsample')
    parser.add_argument("--tbdir", type=str, required=True,
                        help="tensorboard log directory")
    parser.add_argument("--num_gpu", type=int, default=1,
                        help=">1 will use DataParallel")
    parser.add_argument("--torch_hub_dir", type=str, default='',
                        help=">1 will use DataParallel")
    # training options
    parser.add_argument("--netdepth", type=int, default=8,
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256,
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8,
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256,
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=32 * 32 * 4,
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=5e-4,
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250,
                        help='exponential learning rate decay (in 1000 steps)')
    # generate N_rand # of rays, divide into chunk # of batch
    # then generate chunk * N_samples # of points, divide into netchunk # of batch
    parser.add_argument("--chunk", type=int, default=1024 * 32,
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024 * 64,
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None,
                        help='specific weights npy file to reload for coarse network')

    # rendering options
    parser.add_argument("--N_iters", type=int, default=50000,
                        help='number of iteration')
    parser.add_argument("--N_samples", type=int, default=64,
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true',
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0,
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10,
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4,
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0.,
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    parser.add_argument("--rgb_activate", type=str, default='sigmoid',
                        help='activate function for rgb output, choose among "none", "sigmoid"')
    parser.add_argument("--sigma_activate", type=str, default='relu',
                        help='activate function for sigma output, choose among "relu", "softplue"')

    
    parser.add_argument("--recenter_traj", action='store_true',
                        help='recenter traj center')
    # ===============================
    # Kernel optimizing
    # ===============================
    parser.add_argument("--kernel_type", type=str, default='kernel',
                        help='choose among <none>, <itsampling>, <sparsekernel>')
    parser.add_argument("--kernel_isglobal", action='store_true',
                        help='if specified, the canonical kernel position is global')
    parser.add_argument("--kernel_start_iter", type=int, default=0,
                        help='start training kernel after # iteration')
    parser.add_argument("--kernel_ptnum", type=int, default=5,
                        help='the number of sparse locations in the kernels '
                             'that involves computing the final color of ray')
    parser.add_argument("--kernel_random_hwindow", type=float, default=0.25,
                        help='randomly displace the predicted ray position')
    parser.add_argument("--kernel_img_embed", type=int, default=32,
                        help='the dim of image laten code')
    parser.add_argument("--kernel_rand_dim", type=int, default=2,
                        help='dimensions of input random number which uniformly sample from (0, 1)')
    parser.add_argument("--kernel_rand_embed", type=int, default=3,
                        help='embed frequency of input kernel coordinate')
    parser.add_argument("--kernel_rand_mode", type=str, default='float',
                        help='<float>, <<int#, such as<int5>>>, <fix>')
    parser.add_argument("--kernel_random_mode", type=str, default='input',
                        help='<input>, <output>')
    parser.add_argument("--kernel_spatial_embed", type=int, default=0,
                        help='the dim of spatial coordinate embedding')
    parser.add_argument("--kernel_depth_embed", type=int, default=0,
                        help='the dim of depth coordinate embedding')
    parser.add_argument("--kernel_hwindow", type=int, default=10,
                        help='the max window of the kernel (sparse location will lie inside the window')
    parser.add_argument("--kernel_pattern_init_radius", type=float, default=0.1,
                        help='the initialize radius of init pattern')
    parser.add_argument("--kernel_num_hidden", type=int, default=3,
                        help='the number of hidden layer')
    parser.add_argument("--kernel_num_wide", type=int, default=64,
                        help='the wide of hidden layer')
    parser.add_argument("--kernel_shortcut", action='store_true',
                        help='if yes, add a short cut to the network')
    parser.add_argument("--align_start_iter", type=int, default=0,
                        help='start iteration of the align loss')
    parser.add_argument("--align_end_iter", type=int, default=1e10,
                        help='end iteration of the align loss')
    parser.add_argument("--kernel_align_weight", type=float, default=0,
                        help='align term weight')
    parser.add_argument("--prior_start_iter", type=int, default=0,
                        help='start iteration of the prior loss')
    parser.add_argument("--prior_end_iter", type=int, default=1e10,
                        help='end iteration of the prior loss')
    parser.add_argument("--kernel_prior_weight", type=float, default=0,
                        help='weight of prior loss (regularization)')
    parser.add_argument("--sparsity_start_iter", type=int, default=0,
                        help='start iteration of the sparsity loss')
    parser.add_argument("--sparsity_end_iter", type=int, default=1e10,
                        help='end iteration of the sparsity loss')
    parser.add_argument("--kernel_sparsity_type", type=str, default='tv',
                        help='type of sparse gradient loss', choices=['tv', 'normalize', 'robust'])
    parser.add_argument("--kernel_sparsity_weight", type=float, default=0,
                        help='weight of sparsity loss')
    parser.add_argument("--kernel_spatialvariant_trans", action='store_true',
                        help='if true, optimize spatial variant 3D translation of each sampling point')
    parser.add_argument("--kernel_global_trans", action='store_true',
                        help='if true, optimize global 3D translation of each sampling point')
    parser.add_argument("--tone_mapping_type", type=str, default='none',
                        help='the tone mapping of linear to LDR color space, <none>, <gamma>, <learn>')

    ####### render option, will not effect training ########
    parser.add_argument("--visualize_voxel_grid", action='store_true',
                        help='do not optimize, visualize voxel_grid')
    parser.add_argument("--eval_only", action='store_true',
                        help='do not optimize, reload weights and evaluate test images')
    parser.add_argument("--render_only", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--export_colmap", action='store_true',
                        help='do not optimize, reload weights and export learned camera pose to colmap data format.')
    parser.add_argument("--export_trajectory", action='store_true',
                        help='do not optimize, reload weights and export learned camera pose trajectory.')
    parser.add_argument("--render_test", action='store_true',
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_multipoints", action='store_true',
                        help='render sub image that reconstruct the blur image')
    parser.add_argument("--render_rmnearplane", type=int, default=0,
                        help='when render, set the density of nearest plane to 0')
    parser.add_argument("--render_focuspoint_scale", type=float, default=1.,
                        help='scale the focal point when render')
    parser.add_argument("--render_radius_scale", type=float, default=1.,
                        help='scale the radius of the camera path')
    parser.add_argument("--render_factor", type=int, default=0,
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')
    parser.add_argument("--render_epi", action='store_true',
                        help='render the video with epi path')

    ## llff flags
    parser.add_argument("--factor", type=int, default=None,
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true',
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true',
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true',
                        help='set for spherical 360 scenes')
    parser.add_argument("--interp_video", action='store_true',
                        help='set for interpolating test views for video')
    parser.add_argument("--llffhold", type=int, default=8,
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    # ######### Unused params from the original ###########
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops')
    # dataset options
    parser.add_argument("--dataset_type", type=str, default='llff',
                        help='options: llff / synthetic / exblur')
    parser.add_argument("--dataset_name", type=str, default='exblur',
                        help='options: real_motion_blur / exblur / synthetic')
    parser.add_argument("--testskip", type=int, default=8,
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')
    ## deepvoxels flags
    parser.add_argument("--shape", type=str, default='greek',
                        help='options : armchair / cube / greek / vase')
    ## blender flags
    parser.add_argument("--white_bkgd", action='store_true',
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true',
                        help='load blender synthetic data at 400x400 instead of 800x800')


    ################# logging/saving options ##################
    parser.add_argument("--i_print", type=int, default=200,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_tensorboard", type=int, default=200,
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=20000,
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=20000,
                        help='frequency of testset saving')
    parser.add_argument("--i_video", type=int, default=20000,
                        help='frequency of render_poses video saving')


    ################## Architecture Option ########################
    parser.add_argument("--architecture",type=str,default='nerf',
                        help="choose nerf or plenoxel as an architecture")

    ## camera pose into learnable param
    parser.add_argument("--learnable_pose", action='store_true',
                        help='register 6DOF pose instead of 3x4 matrix buffer')

    # vanilla_nerf / kernel_optimzier
    parser.add_argument("--mlp_optimizer", type=str, default='adam',
                        help='choose adam/sgd/rmsprop as an optimizer for nerf/dsk/ctk')
    ################## Blur model Option ########################
    parser.add_argument("--curve_order", type=int, default=6,
                        help='the order of control points for bezier curve.')
    parser.add_argument("--i_cam", type=int, default=20000,
                        help='Visualize camera trajectory.')
    parser.add_argument("--lrate_traj", type=float, default=1e-4,
                        help='learning rate for camera trajectory.')
    ################### etc. #####################################
    parser.add_argument("--disable_align_loss", action='store_true',
                        help='disable align loss')
                        
    parser.add_argument("--kernel_aggregate_type", type=str, default="weight",
                        help="choose 'weight' or 'mean' for the mode to aggregate output from blur kernel")
    
    parser.add_argument("--afterupsample_lr_factor", type=float, default=1.0,
                        help="if plenoxel, lr will be mult'd by this value for every upsampling.")
    parser.add_argument("--hard_sample_ratio", type=float, default=0.,
                        help="ratio n_hard_sample / all_samples")
    parser.add_argument("--ndc_bound_config", type=str, default='automatic', help="'automatic' to set plenoxel boundary with ndc_rays. 'manual' to use plenoxel default.")
    ###############################################################
    # BELOW IS PLENOXEL OPTIONS                                   #
    ###############################################################
    
    group = parser.add_argument_group("Data loading")
    group.add_argument('--scene_scale',
                         type=float,
                         default=None,
                         help="Global scene scaling (or use dataset default)")
    group.add_argument('--scale',
                         type=float,
                         default=None,
                         help="Image scale, e.g. 0.5 for half resolution (or use dataset default)")
    group.add_argument('--seq_id',
                         type=int,
                         default=1000,
                         help="Sequence ID (for CO3D only)")
    group.add_argument('--epoch_size',
                         type=int,
                         default=12800,
                         help="Pseudo-epoch size in term of batches (to be consistent across datasets)")

    group = parser.add_argument_group("Render options")
    group.add_argument('--step_size',
                         type=float,
                         default=0.5,
                         help="Render step size (in voxel size units)")
    group.add_argument('--sigma_thresh',
                         type=float,
                         default=1e-8,
                         help="Skips voxels with sigma < this")
    group.add_argument('--stop_thresh',
                         type=float,
                         default=1e-7,
                         help="Ray march stopping threshold")
    group.add_argument('--background_brightness',
                         type=float,
                         default=1.0,
                         help="Brightness of the infinite background")
    group.add_argument("--background_rgb",default="59,134,232",help="Brightness of the infinite background in rgb")
    group.add_argument('--renderer_backend', '-B',
                         choices=['cuvol', 'svox1', 'nvol'],
                         default='cuvol',
                         help="Renderer backend")
    group.add_argument('--random_sigma_std',
                         type=float,
                         default=0.0,
                         help="Random Gaussian std to add to density values (only if enable_random)")
    group.add_argument('--random_sigma_std_background',
                         type=float,
                         default=0.0,
                         help="Random Gaussian std to add to density values for BG (only if enable_random)")
    group.add_argument('--near_clip',
                         type=float,
                         default=0.00,
                         help="Near clip distance (in world space distance units, only for FG)")
    group.add_argument('--use_spheric_clip',
                         action='store_true',
                         default=False,
                         help="Use spheric ray clipping instead of voxel grid AABB "
                              "(only for FG; changes near_clip to mean 1-near_intersection_radius; "
                              "far intersection is always at radius 1)")
    group.add_argument('--enable_random',
                         action='store_true',
                         default=False,
                         help="Random Gaussian std to add to density values")
    group.add_argument('--last_sample_opaque',
                         action='store_true',
                         default=False,
                         help="Last sample has +1e9 density (used for LLFF)")
    group.add_argument('--camera_scale_factor',
                         type=float,
                         default=0.9,
                         help="Scaling size of z-axis of the plenoxels volume")
    

    group = parser.add_argument_group("general")
    group.add_argument('--train_dir', '-t', type=str, default='ckpt',
                        help='checkpoint and logging directory')

    group.add_argument('--reso',
                            type=str,
                            default=
                            "[[256, 256, 256], [512, 512, 512]]",
                        help='List of grid resolution (will be evaled as json);'
                                'resamples to the next one every upsamp_every iters, then ' +
                                'stays at the last one; ' +
                                'should be a list where each item is a list of 3 ints or an int')
    group.add_argument('--upsamp_every', type=int, default=
                        3 * 12800,
                        help='upsample the grid every x iters')
    group.add_argument('--init_iters', type=int, default=
                        0,
                        help='do not upsample for first x iters')
    group.add_argument('--upsample_density_add', type=float, default=
                        0.0,
                        help='add the remaining density by this amount when upsampling')

    group.add_argument('--basis_type',
                        choices=['sh', '3d_texture', 'mlp'],
                        default='sh',
                        help='Basis function type')

    group.add_argument('--basis_reso', type=int, default=32,
                    help='basis grid resolution (only for learned texture)')
    group.add_argument('--sh_dim', type=int, default=9, help='SH/learned basis dimensions (at most 10)')

    group.add_argument('--mlp_posenc_size', type=int, default=4, help='Positional encoding size if using MLP basis; 0 to disable')
    group.add_argument('--mlp_width', type=int, default=32, help='MLP width if using MLP basis')

    group.add_argument('--background_nlayers', type=int, default=0,#32,
                    help='Number of background layers (0=disable BG model)')
    group.add_argument('--background_reso', type=int, default=512, help='Background resolution')

    group = parser.add_argument_group("optimization")

    # TODO: make the lr higher near the end
    group.add_argument('--sigma_optim', choices=['sgd', 'rmsprop'], default='rmsprop', help="Density optimizer")
    group.add_argument('--lr_sigma', type=float, default=3e1, help='SGD/rmsprop lr for sigma')
    group.add_argument('--lr_sigma_final', type=float, default=5e-2)
    group.add_argument('--lr_sigma_decay_steps', type=int, default=250000)
    group.add_argument('--lr_sigma_delay_steps', type=int, default=15000,
                    help="Reverse cosine steps (0 means disable)")
    group.add_argument('--lr_sigma_delay_mult', type=float, default=1e-2)#1e-4)#1e-4)


    group.add_argument('--sh_optim', choices=['sgd', 'rmsprop'], default='rmsprop', help="SH optimizer")
    group.add_argument('--lr_sh', type=float, default=
                        1e-2,
                    help='SGD/rmsprop lr for SH')
    group.add_argument('--lr_sh_final', type=float,
                        default=
                        5e-6
                        )
    group.add_argument('--lr_sh_decay_steps', type=int, default=250000)
    group.add_argument('--lr_sh_delay_steps', type=int, default=0, help="Reverse cosine steps (0 means disable)")
    group.add_argument('--lr_sh_delay_mult', type=float, default=1e-2)

    group.add_argument('--lr_fg_begin_step', type=int, default=0, help="Foreground begins training at given step number")

    # BG LRs
    group.add_argument('--bg_optim', choices=['sgd', 'rmsprop'], default='rmsprop', help="Background optimizer")
    group.add_argument('--lr_sigma_bg', type=float, default=3e0,
                        help='SGD/rmsprop lr for background')
    group.add_argument('--lr_sigma_bg_final', type=float, default=3e-3,
                        help='SGD/rmsprop lr for background')
    group.add_argument('--lr_sigma_bg_decay_steps', type=int, default=250000)
    group.add_argument('--lr_sigma_bg_delay_steps', type=int, default=0, help="Reverse cosine steps (0 means disable)")
    group.add_argument('--lr_sigma_bg_delay_mult', type=float, default=1e-2)

    group.add_argument('--lr_color_bg', type=float, default=1e-1,
                        help='SGD/rmsprop lr for background')
    group.add_argument('--lr_color_bg_final', type=float, default=5e-6,#1e-4,
                        help='SGD/rmsprop lr for background')
    group.add_argument('--lr_color_bg_decay_steps', type=int, default=250000)
    group.add_argument('--lr_color_bg_delay_steps', type=int, default=0, help="Reverse cosine steps (0 means disable)")
    group.add_argument('--lr_color_bg_delay_mult', type=float, default=1e-2)
    # END BG LRs

    group.add_argument('--basis_optim', choices=['sgd', 'rmsprop'], default='rmsprop', help="Learned basis optimizer")
    group.add_argument('--lr_basis', type=float, default=#2e6,
                        1e-6,
                    help='SGD/rmsprop lr for SH')
    group.add_argument('--lr_basis_final', type=float,
                        default=
                        1e-6
                        )
    group.add_argument('--lr_basis_decay_steps', type=int, default=250000)
    group.add_argument('--lr_basis_delay_steps', type=int, default=0,#15000,
                    help="Reverse cosine steps (0 means disable)")
    group.add_argument('--lr_basis_begin_step', type=int, default=0)#4 * 12800)
    group.add_argument('--lr_basis_delay_mult', type=float, default=1e-2)

    group.add_argument('--rms_beta', type=float, default=0.95, help="RMSProp exponential averaging factor")

    group.add_argument('--print_every', type=int, default=20, help='print every')
    group.add_argument('--save_every', type=int, default=5,
                    help='save every x epochs')
    group.add_argument('--eval_every', type=int, default=1,
                    help='evaluate every x epochs')

    group.add_argument('--init_sigma', type=float,
                    default=0.1,
                    help='initialization sigma')
    group.add_argument('--init_sigma_bg', type=float,
                    default=0.1,
                    help='initialization sigma (for BG)')

    # Extra logging
    group.add_argument('--log_mse_image', action='store_true', default=False)
    group.add_argument('--log_depth_map', action='store_true', default=False)
    group.add_argument('--log_depth_map_use_thresh', type=float, default=None,
            help="If specified, uses the Dex-neRF version of depth with given thresh; else returns expected term")


    group = parser.add_argument_group("misc experiments")
    group.add_argument('--thresh_type',
                        choices=["weight", "sigma"],
                        default="weight",
                    help='Upsample threshold type')
    group.add_argument('--weight_thresh', type=float,
                        default=0.0005 * 512,
                        #  default=0.025 * 512,
                    help='Upsample weight threshold; will be divided by resulting z-resolution')
    group.add_argument('--density_thresh', type=float,
                        default=5.0,
                    help='Upsample sigma threshold')
    group.add_argument('--background_density_thresh', type=float,
                        default=1.0+1e-9,
                    help='Background sigma threshold for sparsification')
    group.add_argument('--max_grid_elements', type=int,
                        default=44_000_000,
                    help='Max items to store after upsampling '
                            '(the number here is given for 22GB memory)')

    group.add_argument('--tune_mode', action='store_true', default=False,
                    help='hypertuning mode (do not save, for speed)')
    group.add_argument('--tune_nosave', action='store_true', default=False,
                    help='do not save any checkpoint even at the end')



    group = parser.add_argument_group("losses")
    # Foreground TV
    group.add_argument('--lambda_tv', type=float, default=1e-5)
    group.add_argument('--tv_sparsity', type=float, default=0.01)
    group.add_argument('--tv_logalpha', action='store_true', default=False,
                    help='Use log(1-exp(-delta * sigma)) as in neural volumes')

    group.add_argument('--lambda_tv_sh', type=float, default=1e-3)
    group.add_argument('--tv_sh_sparsity', type=float, default=0.01)

    group.add_argument('--lambda_tv_lumisphere', type=float, default=0.0)#1e-2)#1e-3)
    group.add_argument('--tv_lumisphere_sparsity', type=float, default=0.01)
    group.add_argument('--tv_lumisphere_dir_factor', type=float, default=0.0)

    group.add_argument('--tv_decay', type=float, default=1.0)

    group.add_argument('--lambda_l2_sh', type=float, default=0.0)#1e-4)
    group.add_argument('--tv_early_only', type=int, default=1, help="Turn off TV regularization after the first split/prune")

    group.add_argument('--tv_contiguous', type=int, default=1,
                            help="Apply TV only on contiguous link chunks, which is faster")
    # End Foreground TV

    group.add_argument('--lambda_sparsity', type=float, default=
                        0.0,
                        help="Weight for sparsity loss as in SNeRG/PlenOctrees " +
                            "(but applied on the ray)")
    group.add_argument('--lambda_beta', type=float, default=
                        0.0,
                        help="Weight for beta distribution sparsity loss as in neural volumes")


    # Background TV
    group.add_argument('--lambda_tv_background_sigma', type=float, default=1e-2)
    group.add_argument('--lambda_tv_background_color', type=float, default=1e-2)

    group.add_argument('--tv_background_sparsity', type=float, default=0.01)
    # End Background TV

    # Basis TV
    group.add_argument('--lambda_tv_basis', type=float, default=0.0,
                    help='Learned basis total variation loss')
    # End Basis TV

    group.add_argument('--weight_decay_sigma', type=float, default=1.0)
    group.add_argument('--weight_decay_sh', type=float, default=1.0)

    group.add_argument('--lr_decay', action='store_true', default=True)

    group.add_argument('--n_train', type=int, default=None, help='Number of training images. Defaults to use all avaiable.')

    group.add_argument('--nosphereinit', action='store_true', default=False,
                        help='do not start with sphere bounds (please do not use for 360)')
    
    
    return parser


