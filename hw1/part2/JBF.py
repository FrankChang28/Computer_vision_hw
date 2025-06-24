import numpy as np
import cv2

class Joint_bilateral_filter(object):
    def __init__(self, sigma_s, sigma_r):
        self.sigma_r = sigma_r
        self.sigma_s = sigma_s
        self.wndw_size = 6 * sigma_s + 1
        self.pad_w = 3 * sigma_s  

        x = np.arange(-self.pad_w, self.pad_w + 1)
        y = np.arange(-self.pad_w, self.pad_w + 1)
        xx, yy = np.meshgrid(x, y)
        self.spatial_kernel = np.exp(-(xx ** 2 + yy ** 2) / (2 * self.sigma_s ** 2))
        self.spatial_kernel = self.spatial_kernel[:, :, np.newaxis]

    def joint_bilateral_filter(self, img, guidance):
        BORDER_TYPE = cv2.BORDER_REFLECT
        padded_img = cv2.copyMakeBorder(img, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE).astype(np.float64)
        padded_guidance = cv2.copyMakeBorder(guidance, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE).astype(np.float64)
        
        height, width, channels = img.shape
        filtered_image = np.zeros_like(img, dtype=np.float64)
        padded_guidance = padded_guidance / 255.0

        if padded_guidance.ndim == 2:
            padded_guidance = padded_guidance[:, :, np.newaxis]

        for i in range(height):
            for j in range(width):
                window_img = padded_img[i:i + self.wndw_size, j:j + self.wndw_size]
                window_guidance = padded_guidance[i:i + self.wndw_size, j:j + self.wndw_size]

                intensity_diff = window_guidance - padded_guidance[i + self.pad_w, j + self.pad_w]
                intensity_weight = np.exp(-np.sum(intensity_diff ** 2, axis=2) / (2 * self.sigma_r ** 2))
                intensity_weight = intensity_weight[:, :, np.newaxis]

                weight = self.spatial_kernel * intensity_weight

                weighted_sum = np.sum(weight * window_img, axis=(0, 1)) 
                sum_weights = np.sum(weight, axis=(0, 1))

                filtered_image[i, j] = weighted_sum // (sum_weights)
        #filtered_image_uint8 = np.clip(filtered_image, 0, 255).astype(np.uint8)
        #filtered_image_bgr = cv2.cvtColor(filtered_image_uint8, cv2.COLOR_RGB2BGR)
        #cv2.imwrite("filtered_image.png", filtered_image_bgr)
        return np.clip(filtered_image, 0, 255).astype(np.uint8)
