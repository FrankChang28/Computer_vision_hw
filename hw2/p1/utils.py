# ============================================================================
# File: util.py
# Date: 2025-03-11
# Author: TA
# Description: Utility functions to process BoW features and KNN classifier.
# ============================================================================

import numpy as np
from PIL import Image
from tqdm import tqdm
from cyvlfeat.sift.dsift import dsift
from cyvlfeat.kmeans import kmeans
from scipy.spatial.distance import cdist

CAT = ['Kitchen', 'Store', 'Bedroom', 'LivingRoom', 'Office',
       'Industrial', 'Suburb', 'InsideCity', 'TallBuilding', 'Street',
       'Highway', 'OpenCountry', 'Coast', 'Mountain', 'Forest']

CAT2ID = {v: k for k, v in enumerate(CAT)}

########################################
###### FEATURE UTILS              ######
###### use TINY_IMAGE as features ######
########################################

###### Step 1-a
def get_tiny_images(img_paths: str):
    '''
    Build tiny image features.
    - Args: : 
        - img_paths (N): list of string of image paths
    - Returns: :
        - tiny_img_feats (N, d): ndarray of resized and then vectorized
                                 tiny images
    NOTE:
        1. N is the total number of images
        2. if the images are resized to 16x16, d would be 256
    '''
    
    #################################################################
    # TODO:                                                         #
    # To build a tiny image feature, you can follow below steps:    #
    #    1. simply resize the original image to a very small        #
    #       square resolution, e.g. 16x16. You can either resize    #
    #       the images to square while ignoring their aspect ratio  #
    #       or you can first crop the center square portion out of  #
    #       each image.                                             #
    #    2. flatten and normalize the resized image.                #
    #################################################################

    tiny_img_feats = []
    for image_path in img_paths:
            image = Image.open(image_path).convert("L")
            image = image.resize((16,16), Image.LANCZOS)
            image = np.array(image).flatten()
            image = image - np.mean(image)
            norm = np.linalg.norm(image)
            if norm > 0:
                image = image / norm  # Unit length
                
            tiny_img_feats.append(image)

    #################################################################
    #                        END OF YOUR CODE                       #
    #################################################################

    return np.array(tiny_img_feats)

#########################################
###### FEATURE UTILS               ######
###### use BAG_OF_SIFT as features ######
#########################################

###### Step 1-b-1
def build_vocabulary(
        img_paths: list, 
        vocab_size: int = 500
    ):
    '''
    Args:
        img_paths (N): list of string of image paths (training)
        vocab_size: number of clusters desired
    Returns:
        vocab (vocab_size, sift_d): ndarray of clusters centers of k-means
    NOTE:
        1. sift_d is 128
        2. vocab_size is up to you, larger value will works better
           (to a point) but be slower to compute,
           you can set vocab_size in p1.py
    '''
    descriptors=[]
    
    #sample_size = min(200, len(img_paths))  # Use at most 200 images
    feature_size=500
    img_per_class=100
    sample_per_class=20
    for class_idx in range(15):
        class_path = img_paths[class_idx * img_per_class : class_idx * img_per_class + img_per_class]
        sampled_paths = np.random.choice(class_path, size=sample_per_class, replace=False)

        for image_path in sampled_paths:
            image = Image.open(image_path).convert("L")
            image = np.array(image)
            _, desc = dsift(image, step=[4,4], fast=True)
            
            if len(desc) > feature_size:
                desc = grid_sampling(_, desc, grid_size=(3,3), max_features=feature_size, image_shape=image.shape)
            descriptors.append(desc)

    descriptors = np.vstack(descriptors).astype(np.float32)
    vocab = kmeans(descriptors, num_centers=vocab_size)
    
    return vocab

def grid_sampling(keypoints, descriptors, grid_size=(3,3), max_features=None, image_shape=None):
    h, w = image_shape[0], image_shape[1]
    selected = []
    grid_h, grid_w = h // grid_size[0], w // grid_size[1]
    
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            x_min, x_max = j * grid_w, (j+1) * grid_w
            y_min, y_max = i * grid_h, (i+1) * grid_h

            mask = (
                (keypoints[:,0] >= x_min) & (keypoints[:,0] < x_max) & 
                (keypoints[:,1] >= y_min) & (keypoints[:,1] < y_max)
            )
            grid_desc = descriptors[mask]

            if len(grid_desc) > 0:
                k = max_features // (grid_size[0]*grid_size[1]) if max_features else len(grid_desc)
                selected.extend(grid_desc[np.random.choice(len(grid_desc), min(k, len(grid_desc)), replace=False)])
    
    return np.array(selected)
###### Step 1-b-2
def get_bags_of_sifts(
        img_paths: list,
        vocab: np.array
    ):
    '''
    Args:
        img_paths (N): list of string of image paths
        vocab (vocab_size, sift_d) : ndarray of clusters centers of k-means
    Returns:
        img_feats (N, d): ndarray of feature of images, each row represent
                          a feature of an image, which is a normalized histogram
                          of vocabularies (cluster centers) on this image
    NOTE :
        1. d is vocab_size here
    '''

    ############################################################################
    # TODO:                                                                    #
    # To get bag of SIFT words (centroids) of each image, you can follow below #
    # steps:                                                                   #
    #   1. for each loaded image, get its 128-dim SIFT features (descriptors)  #
    #      in the same way you did in build_vocabulary()                       #
    #   2. calculate the distances between these features and cluster centers  #
    #   3. assign each local feature to its nearest cluster center             #
    #   4. build a histogram indicating how many times each cluster presents   #
    #   5. normalize the histogram by number of features, since each image     #
    #      may be different                                                    #
    # These histograms are now the bag-of-sift feature of images               #
    #                                                                          #
    # NOTE:                                                                    #
    # Some useful functions                                                    #
    #   Function : dsift(img, step=[x, x], fast=True)                          #
    #   Function : cdist(feats, vocab)                                         #
    #                                                                          #
    # NOTE:                                                                    #
    #   1. we recommend first completing function 'build_vocabulary()'         #
    ############################################################################

    img_feats = []
    vocab_size = vocab.shape[0]
    for image_path in img_paths:
        image = Image.open(image_path).convert("L")
        image = np.array(image)
        _, desc = dsift(image, step=[4,4], fast=True)
        distances = cdist(desc, vocab, metric="euclidean")
        nearest_indices = np.argmin(distances, axis=1)

        hist, _ = np.histogram(nearest_indices, bins = np.arange(vocab_size+1), density=False)
        hist = hist.astype(np.float32)/hist.sum()

        img_feats.append(hist)

    ############################################################################
    #                                END OF YOUR CODE                          #
    ############################################################################
    
    return img_feats

################################################
###### CLASSIFIER UTILS                   ######
###### use NEAREST_NEIGHBOR as classifier ######
################################################

###### Step 2
def nearest_neighbor_classify(
        train_img_feats: np.array,
        train_labels: list,
        test_img_feats: list
    ):
    '''
    Args:
        train_img_feats (N, d): ndarray of feature of training images
        train_labels (N): list of string of ground truth category for each 
                          training image
        test_img_feats (M, d): ndarray of feature of testing images
    Returns:
        test_predicts (M): list of string of predict category for each 
                           testing image
    NOTE:
        1. d is the dimension of the feature representation, depending on using
           'tiny_image' or 'bag_of_sift'
        2. N is the total number of training images
        3. M is the total number of testing images
    '''

    ###########################################################################
    # TODO:                                                                   #
    # KNN predict the category for every testing image by finding the         #
    # training image with most similar (nearest) features, you can follow     #
    # below steps:                                                            #
    #   1. calculate the distance between training and testing features       #
    #   2. for each testing feature, select its k-nearest training features   #
    #   3. get these k training features' label id and vote for the final id  #
    # Remember to convert final id's type back to string, you can use CAT     #
    # and CAT2ID for conversion                                               #
    #                                                                         #
    # NOTE:                                                                   #
    # Some useful functions                                                   #
    #   Function : cdist(feats, feats)                                        #
    #                                                                         #
    # NOTE:                                                                   #
    #   1. instead of 1 nearest neighbor, you can vote based on k nearest     #
    #      neighbors which may increase the performance                       #
    #   2. hint: use 'minkowski' metric for cdist() and use a smaller 'p' may #
    #      work better, or you can also try different metrics for cdist()     #
    ###########################################################################

    test_predicts = []
    
    train_label_ids = np.array([CAT2ID[label] for label in train_labels])
    distances = cdist(test_img_feats, train_img_feats, metric='minkowski', p=1.1)
    
    k = 5 
    nearest_indices = np.argpartition(distances, k, axis=1)[:, :k]
    nearest_labels = train_label_ids[nearest_indices]

    for labels in nearest_labels:
        counts = np.bincount(labels)
        predicted_id = np.argmax(counts)
        test_predicts.append(CAT[predicted_id])
    ###########################################################################
    #                               END OF YOUR CODE                          #
    ###########################################################################

    return test_predicts
