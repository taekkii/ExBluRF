import os
import argparse
import numpy as np
import colmap_read_model as read_model

def load_colmap_data(realdir):

    camerasfile = os.path.join(realdir, 'sparse/0/cameras.bin')
    camdata = read_model.read_cameras_binary(camerasfile)

    # cam = camdata[camdata.keys()[0]]
    list_of_keys = list(camdata.keys())
    cam = camdata[list_of_keys[0]]
    print( 'Cameras', len(cam))

    h, w, f = cam.height, cam.width, cam.params[0]
    # w, h, f = factor * w, factor * h, factor * f
    hwf = np.array([h,w,f]).reshape([3,1])

    imagesfile = os.path.join(realdir, 'sparse/0/images.bin')
    imdata = read_model.read_images_binary(imagesfile)

    w2c_mats = []
    bottom = np.array([0,0,0,1.]).reshape([1,4])

    names = [imdata[k].name for k in imdata]

    print( 'Images #', len(names))
    
    with open(os.path.join(realdir, 'train.txt'), 'r') as f:
        lines = f.readlines()

        for line in lines:
            if line.strip('\n') not in names:
                print('COLMAP fails for image: {}'.format(line.strip('\n')))
        if len(lines) != len(names):
            exit()

    perm = []
    with open(os.path.join(realdir, 'train.txt'), 'r') as f:
        lines = f.readlines()
        for line in lines:
            for i in range(len(names)):
                if names[i] == line.strip():
                    perm.append(i)
    if os.path.exists(os.path.join(realdir, 'test.txt')):
        with open(os.path.join(realdir, 'test.txt'), 'r') as f:
            lines = f.readlines()
            for line in lines:
                for i in range(len(names)):
                    if names[i] == line.strip():
                        perm.append(i)
    if len(names) != len(perm):
        print('COLMAP fails for some images!')
        exit()
    for k in imdata:
        im = imdata[k]
        R = im.qvec2rotmat()
        t = im.tvec.reshape([3,1])
        m = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
        w2c_mats.append(m)

    w2c_mats = np.stack(w2c_mats, 0)
    c2w_mats = np.linalg.inv(w2c_mats)

    poses = c2w_mats[:, :3, :4].transpose([1,2,0])
    poses = np.concatenate([poses, np.tile(hwf[..., np.newaxis], [1,1,poses.shape[-1]])], 1)

    points3dfile = os.path.join(realdir, 'sparse/0/points3D.bin')
    pts3d = read_model.read_points3d_binary(points3dfile)

    # must switch to [-u, r, -t] from [r, -u, t], NOT [r, u, -t]
    poses = np.concatenate([poses[:, 1:2, :], poses[:, 0:1, :], -poses[:, 2:3, :], poses[:, 3:4, :], poses[:, 4:5, :]], 1)

    return poses, pts3d, perm


def save_poses(basedir, poses, pts3d, perm):
    pts_arr = []
    vis_arr = []
    for k in pts3d:
        pts_arr.append(pts3d[k].xyz)
        cams = [0] * poses.shape[-1]
        for ind in pts3d[k].image_ids:
            if len(cams) < ind - 1:
                print('ERROR: the correct camera poses for current points cannot be accessed')
                return
            cams[ind-1] = 1
        vis_arr.append(cams)

    pts_arr = np.array(pts_arr)
    vis_arr = np.array(vis_arr)
    print( 'Points', pts_arr.shape, 'Visibility', vis_arr.shape )

    zvals = np.sum(-(pts_arr[:, np.newaxis, :].transpose([2,0,1]) - poses[:3, 3:4, :]) * poses[:3, 2:3, :], 0)
    valid_z = zvals[vis_arr==1]
    print( 'Depth stats', valid_z.min(), valid_z.max(), valid_z.mean() )

    save_arr = []
    for i in perm:
        vis = vis_arr[:, i]
        zs = zvals[:, i]
        zs = zs[vis==1]
        close_depth, inf_depth = np.percentile(zs, .1), np.percentile(zs, 99.9)
        # print( i, close_depth, inf_depth )

        save_arr.append(np.concatenate([poses[..., i].ravel(), np.array([close_depth, inf_depth])], 0))
    save_arr = np.array(save_arr)

    np.save(os.path.join(basedir, 'poses_bounds.npy'), save_arr)


def gen_poses(basedir, factors=None):

    print( 'Post-colmap')

    poses, pts3d, perm = load_colmap_data(basedir)
    
    save_poses(basedir, poses, pts3d, perm)

    if factors is not None:
        print( 'Factors:', factors)
        minify(basedir, factors)

    print( 'Done with imgs2poses' )

    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str,)
    parser.add_argument('--factor', type=str,)
    args = parser.parse_args()

    gen_poses(args.datadir)


#print('hello')

if __name__=='__main__':
    main()

