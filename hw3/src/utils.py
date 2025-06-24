import numpy as np


def solve_homography(u, v):
    N = u.shape[0]
    H = None

    if v.shape[0] is not N:
        print('u and v should have the same size')
        return None
    if N < 4:
        print('At least 4 points should be given')
    # TODO: 1.forming A
    x = u[:, 0]
    y = u[:, 1]
    x_p = v[:, 0]
    y_p = v[:, 1]

    zeros = np.zeros_like(x)
    ones = np.ones_like(x)
    A1 = np.stack([x, y, ones, zeros, zeros, zeros, -x * x_p, -y * x_p, -x_p], axis=1)
    A2 = np.stack([zeros, zeros, zeros, x, y, ones, -x * y_p, -y * y_p, -y_p], axis=1)
    A = np.vstack([A1, A2])  # shape: (2N, 9)
    # TODO: 2.solve H with A
    U, S, Vt = np.linalg.svd(A)
    h = Vt[-1, :]
    h = h / h[-1] 
    H = h.reshape((3,3))
    return H


def warping(src, dst, H, ymin, ymax, xmin, xmax, direction='b'):
    h_src, w_src, ch = src.shape
    h_dst, w_dst, ch = dst.shape
    H_inv = np.linalg.inv(H)

    # TODO: 1.meshgrid the (x,y) coordinate pairs
    x = np.arange(xmin, xmax)
    y = np.arange(ymin, ymax)
    xv, yv = np.meshgrid(x, y)
    # TODO: 2.reshape the destination pixels as N x 3 homogeneous coordinate
    num_pixels = xv.size
    ones = np.ones((num_pixels,))
    
    if direction == 'b':
        dst_coords = np.stack([xv.ravel(), yv.ravel(), ones], axis=1)
        # TODO: 3.apply H_inv to the destination pixels and retrieve (u,v) pixels, then reshape to (ymax-ymin),(xmax-xmin)
        src_coords = (H_inv @ dst_coords.T).T  # shape: (N, 3)
        src_coords = src_coords / src_coords[:, 2:3]  # Normalize by the third (homogeneous) coordinate
        ux = src_coords[:, 0].reshape((ymax - ymin), (xmax - xmin))
        uy = src_coords[:, 1].reshape((ymax - ymin), (xmax - xmin))
        # TODO: 4.calculate the mask of the transformed coordinate (should not exceed the boundaries of source image)
        valid_mask = (
            (ux >= 0) & (ux < w_src) &
            (uy >= 0) & (uy < h_src)
        )
        # TODO: 5.sample the source image with the masked and reshaped transformed coordinates
        valid_x = ux[valid_mask].astype(int)
        valid_y = uy[valid_mask].astype(int)
        
        # Get the corresponding destination coordinates
        dst_x_valid = xv[valid_mask]
        dst_y_valid = yv[valid_mask]
        # TODO: 6. assign to destination image with proper masking
        dst[dst_y_valid, dst_x_valid] = src[valid_y, valid_x]

    elif direction == 'f':
        src_coords = np.stack([xv.ravel(), yv.ravel(), ones], axis=1)# Shape: (N, 3)
        # TODO: 3.apply H to the source pixels and retrieve (u,v) pixels, then reshape to (ymax-ymin),(xmax-xmin)
        dst_coords = (H @ src_coords.T).T  # shape: (N, 3)
        dst_coords = dst_coords / dst_coords[:, 2:3]
        vx = dst_coords[:, 0].reshape((ymax-ymin),(xmax-xmin))
        vy = dst_coords[:, 1].reshape((ymax-ymin),(xmax-xmin))
        # TODO: 4.calculate the mask of the transformed coordinate (should not exceed the boundaries of destination image)
        valid_mask = (
            (vx >= 0) & (vx < w_dst) &
            (vy >= 0) & (vy < h_dst)
        )
        # TODO: 5.filter the valid coordinates using previous obtained mask
        x_valid = vx[valid_mask].astype(int)
        y_valid = vy[valid_mask].astype(int)
        src_x_valid = xv[valid_mask]
        src_y_valid = yv[valid_mask]    
        # TODO: 6. assign to destination image using advanced array indicing
        dst[y_valid, x_valid] = src[src_y_valid, src_x_valid]
    return dst 
