import torch

def vectoMat_single(vec):
    '''
    input:  vector  (-1, 6)
    output: matrix  (-1, 3, 4)
    '''

    # vec: (x, y, z, pi, chi, psi)
    cos_pi = torch.cos(vec[:, 3])
    sin_pi = torch.sin(vec[:, 3])

    cos_chi = torch.cos(vec[:, 4])
    sin_chi = torch.sin(vec[:, 4])

    cos_psi = torch.cos(vec[:, 5])
    sin_psi = torch.sin(vec[:, 5])

    mat = torch.zeros(vec.shape[0], 3, 4).float()

    # 3D Rotation.
    mat[:, 0, 0] = cos_pi * cos_chi
    mat[:, 0, 1] = cos_pi * sin_chi * sin_psi - sin_pi * cos_psi
    mat[:, 0, 2] = cos_pi * sin_chi * cos_psi + sin_pi * sin_psi

    mat[:, 1, 0] = sin_pi * cos_chi
    mat[:, 1, 1] = sin_pi * sin_chi * sin_psi + cos_pi * cos_psi
    mat[:, 1, 2] = sin_pi * sin_chi * cos_psi - cos_pi * sin_psi

    mat[:, 2, 0] = -sin_chi
    mat[:, 2, 1] = cos_chi * sin_psi
    mat[:, 2, 2] = cos_chi * cos_psi

    # Translation.
    mat[:, 0, 3] = vec[:, 0]
    mat[:, 1, 3] = vec[:, 1]
    mat[:, 2, 3] = vec[:, 2]

    return mat

def Mat2Vec_single(mat):
    '''
    input:  matrix  (-1, 3, 4)
    output: vector  (-1, 6)
    '''

    input_shape = mat.shape

    # vec: (x, y, z, pi, chi, psi)
    vec = torch.zeros(input_shape[0], 6).float()

    # (x,y,z)
    vec[:, 0] = mat[:, 0, 3]  # x
    vec[:, 1] = mat[:, 1, 3]  # y
    vec[:, 2] = mat[:, 2, 3]  # z

    # (pi, chi, psi)
    vec[:, 3] = torch.atan2(mat[:, 1, 0], mat[:, 0, 0])  # pi
    vec[:, 4] = torch.atan2( -mat[:, 2, 0], torch.sqrt( mat[:, 0, 0] ** 2 + mat[:, 1, 0] ** 2 ))  # chi
    vec[:, 5] = torch.atan2(mat[:, 2, 1], mat[:, 2, 2])  # psi
    
    # when |chi| == 90
    for i in range(input_shape[0]):
        if mat[i, 2, 0] == -1:
            vec[i, 3] = torch.atan2(mat[i, 1, 2], mat[i, 0, 2])
            vec[i, 5] = 0
        elif mat[i, 2, 0] == 1:
            vec[i, 3] = torch.atan2(-mat[i, 1, 2], -mat[i, 0, 2])
            vec[i, 5] = 0

    return vec 

def se3toQuat_single(vec):
    '''
    input:  vector  (-1, 3)
    output: vector  (-1, 4)
    '''

    # vec: (pi, chi, psi)
    cos_pi = torch.cos(vec[:, 0]/2)
    sin_pi = torch.sin(vec[:, 0]/2)

    cos_chi = torch.cos(vec[:, 1]/2)
    sin_chi = torch.sin(vec[:, 1]/2)

    cos_psi = torch.cos(vec[:, 2]/2)
    sin_psi = torch.sin(vec[:, 2]/2)

    n_vec = vec.shape[0]
    quat = torch.zeros(n_vec, 4).float()

    # quarternion.
    quat[:, 0] = cos_psi*cos_chi*cos_pi + sin_psi*sin_chi*sin_pi
    quat[:, 1] = sin_psi*cos_chi*cos_pi - cos_psi*sin_chi*sin_pi
    quat[:, 2] = cos_psi*sin_chi*cos_pi + sin_psi*cos_chi*sin_pi
    quat[:, 3] = cos_psi*cos_chi*sin_pi - sin_psi*sin_chi*cos_pi
    
    return quat

def inv3x4(mat):
    '''
    input: mat (-1, 4, 4)
    output: mat (-1, 4, 4)
    '''
    num = mat.shape[0]
    bottom = torch.tensor([0., 0., 0., 1.], device='cpu').reshape(1, 1, 4).repeat(num, 1, 1)

    mat1 = torch.cat([mat, bottom], dim=1)

    mat_inv = torch.inverse(mat1)

    return mat_inv[:, :3, :4]

def alignSE3Single(es, gt):
    
    # The ATE is borrowed and modified from repo rpg_trajectory_evaluation.

    '''
    Calculate SE3 transformation R and t so that:
        gt = R * est + t
    Using only the first poses of est and gt
    '''
    t_es = es[0, 0:3, 3:4]
    R_es = es[0, 0:3, 0:3]

    t_gt = gt[0, 0:3, 3:4]
    R_gt = gt[0, 0:3, 0:3]

    R = torch.mm(R_gt, torch.t(R_es))
    t = t_gt - torch.mm(R, t_es)

    return R, t