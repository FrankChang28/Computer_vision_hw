import numpy as np
import cv2

class Difference_of_Gaussian(object):
    def __init__(self, threshold):
        self.threshold = threshold
        self.sigma = 2**(1/4)
        self.num_octaves = 2
        self.num_DoG_images_per_octave = 4
        self.num_guassian_images_per_octave = self.num_DoG_images_per_octave + 1

    def get_keypoints(self, image):
        ### TODO ####
        # Step 1: Filter images with different sigma values (5 images per octave, 2 octave in total)
        # - Function: cv2.GaussianBlur (kernel = (0, 0), sigma = self.sigma**___)
        gaussian_images = []

        img_temp=image
        for i in range(self.num_octaves):
            gaussian_images.append(img_temp)
            for j in range(1,self.num_guassian_images_per_octave):
                gaussian_image=cv2.GaussianBlur(img_temp, (0, 0), self.sigma**(j))
                gaussian_images.append(gaussian_image)
                #cv2.imwrite(f'gaussian_octave{i}_image{j}.jpg', gaussian_image)
            img_temp = cv2.resize(gaussian_image, (gaussian_image.shape[1] // 2, gaussian_image.shape[0] // 2), interpolation=cv2.INTER_NEAREST)             
            

        # Step 2: Subtract 2 neighbor images to get DoG images (4 images per octave, 2 octave in total)
        # - Function: cv2.subtract(second_image, first_image)
        dog_images = []
        for i in range(self.num_octaves):
            for j in range(self.num_DoG_images_per_octave):
                dog_image = cv2.subtract(gaussian_images[j+1+self.num_guassian_images_per_octave*i], gaussian_images[j+self.num_guassian_images_per_octave*i])
                dog_images.append(dog_image)

                normalized_image = cv2.normalize(dog_image, None, 0, 255, cv2.NORM_MINMAX)
                normalized_image = normalized_image.astype(np.uint8)
                cv2.imwrite(f'dog_octave{i}_image{j}.jpg', normalized_image)
        # Step 3: Thresholding the value and Find local extremum (local maximun and local minimum)
        #         Keep local extremum as a keypoint
        keypoints=[]
        for i in range(self.num_octaves):
            for j in range(1, self.num_DoG_images_per_octave-1):
                current_dog=dog_images[j + self.num_DoG_images_per_octave*i]
                prev_dog=dog_images[j + self.num_DoG_images_per_octave*i - 1]
                next_dog=dog_images[j + self.num_DoG_images_per_octave*i + 1]
                cols, rows = current_dog.shape

                for x in range(1, cols - 1):
                    for y in range(1, rows - 1):
                        pixel_value = current_dog[x, y]
                        if abs(pixel_value) > self.threshold:
                            neighbors = [
                                current_dog[x-1:x+2, y-1:y+2, ].flatten(),  # 當前層
                                prev_dog[x-1:x+2, y-1:y+2].flatten(),  # 前一層
                                next_dog[x-1:x+2, y-1:y+2].flatten()   # 下一層
                            ]
                            neighbors = np.concatenate(neighbors)
                            if pixel_value == np.max(neighbors) or pixel_value == np.min(neighbors):
                                keypoints.append((x * (2 ** i), y * (2 ** i)))
        # Step 4: Delete duplicate keypoints
        # - Function: np.unique
        keypoints = np.array(keypoints)
        keypoints = np.unique(keypoints, axis=0)
        # sort 2d-point by y, then by x
        keypoints = keypoints[np.lexsort((keypoints[:,1],keypoints[:,0]))] 
        return keypoints
