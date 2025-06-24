import numpy as np
import cv2
import random
from tqdm import tqdm
from utils import solve_homography, warping

random.seed(999)

def panorama(imgs):
    """
    Image stitching with estimated homograpy between consecutive
    :param imgs: list of images to be stitched
    :return: stitched panorama
    """
    h_max = max([x.shape[0] for x in imgs])
    w_max = sum([x.shape[1] for x in imgs])

    # create the final stitched canvas
    dst = np.zeros((h_max, w_max, imgs[0].shape[2]), dtype=np.uint8)
    dst[:imgs[0].shape[0], :imgs[0].shape[1]] = imgs[0]
    last_best_H = np.eye(3)
    out = None
    width_acc=0
    # for all images to be stitched:
    for idx in tqdm(range(len(imgs)-1)):
        im1 = imgs[idx]
        im2 = imgs[idx + 1]

        # TODO: 1.feature detection & matching
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(im1, None)
        kp2, des2 = orb.detectAndCompute(im2, None)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 2)

        # TODO: 2. apply RANSAC to choose best H
        best_H = None
        max_inliers = 0
        iterations = 2000
        threshold = 4.0
        for _ in range(iterations):
            # Randomly select 4 points
            idxs = random.sample(range(len(src_pts)), 4)
            src_sample = src_pts[idxs]
            dst_sample = dst_pts[idxs]
            
            # Compute homography
            H = solve_homography(src_sample, dst_sample)
            
            if H is None:
                continue
                
            # Transform all points
            ones = np.ones((src_pts.shape[0], 1))
            src_pts_homo = np.hstack((src_pts, ones))
            transformed_pts = H @ src_pts_homo.T
            transformed_pts = transformed_pts[:2] / transformed_pts[2]
            transformed_pts = transformed_pts.T
            
            # Compute distances
            distances = np.linalg.norm(dst_pts - transformed_pts, axis=1)
            
            # Count inliers
            inliers = np.sum(distances < threshold)
            
            if inliers > max_inliers:
                max_inliers = inliers
                best_H = H
        
        # TODO: 3. chain the homographies
        last_best_H = last_best_H @ np.linalg.inv(best_H)
        # TODO: 4. apply warping
        width_acc = width_acc + im1.shape[1]
        # Warp the current image
        new_canvas = warping(im2, dst, last_best_H, 0, h_max, width_acc, width_acc+im2.shape[1], 'b')
        dst = new_canvas
        
        out = new_canvas

    return out 

if __name__ == "__main__":
    # ================== Part 4: Panorama ========================
    # TODO: change the number of frames to be stitched
    FRAME_NUM = 3
    imgs = [cv2.imread('../resource/frame{:d}.jpg'.format(x)) for x in range(1, FRAME_NUM + 1)]
    output4 = panorama(imgs)
    cv2.imwrite('output4.png', output4)