
import cv2
import numpy as np
import os

def comparison_by_method(img_warped, folder_path, method):
    if method == 'SIFT':
        return sift_comparison(img_warped, folder_path)
    elif method == 'SURF':
        return surf_comparison(img_warped, folder_path)
    elif method == 'ORB':
        return orb_comparison(img_warped, folder_path)
    elif method == 'AKAZE':
        return akaze_comparison(img_warped, folder_path)
    else:
        raise ValueError(f"Invalid feature matching method: {method}")

def sift_comparison(img_query, folder_path):
    # Initialize SIFT
    sift = cv2.SIFT_create()

    # Find keypoints and descriptors for the query image
    kp_query, des_query = sift.detectAndCompute(img_query, None)

    # Initialize BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    best_match = None
    best_match_score = float('inf')

    for file in os.listdir(folder_path):
        if file.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder_path, file)
            img_train = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            # Find keypoints and descriptors for the train image
            kp_train, des_train = sift.detectAndCompute(img_train, None)

            # Match keypoints
            matches = bf.match(des_query, des_train)
            matches = list(matches)
            matches.sort(key=lambda x: x.distance)

            # Calculate the total distance of the top matches
            top_matches = matches[:min(10, len(matches))]
            match_score = sum([match.distance for match in top_matches])

            # Update the best match
            if match_score < best_match_score:
                best_match = file
                best_match_score = match_score

    return os.path.splitext(best_match)[0]


def orb_comparison(img_warped, folder_path):
    # Initialize the ORB detector
    orb = cv2.ORB_create()
    
    # Compute the keypoints and descriptors of the query image
    kp1, des1 = orb.detectAndCompute(img_warped, None)
    
    # Initialize the BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    max_matches = 0
    matching_image = None
    
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img2 = cv2.imread(os.path.join(folder_path, filename), 0)
            kp2, des2 = orb.detectAndCompute(img2, None)
            
            matches = bf.match(des1, des2)
            
            # Convert the matches object to a list before sorting
            matches_list = list(matches)
            matches_list.sort(key=lambda x: x.distance)
            
            if len(matches_list) > max_matches:
                max_matches = len(matches_list)
                matching_image = filename
                
    return matching_image
    





def surf_comparison(img_query, folder_path):
    # Initialize SURF
    surf = cv2.xfeatures2d.SURF_create()

    # Find keypoints and descriptors for the query image
    kp_query, des_query = surf.detectAndCompute(img_query, None)

    # Initialize BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    # The rest of the function is the same as the sift_comparison function
    # ...
    best_match = None
    best_match_score = float('inf')

    for file in os.listdir(folder_path):
        if file.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder_path, file)
            img_train = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            # Find keypoints and descriptors for the train image
            kp_train, des_train = surf.detectAndCompute(img_train, None)

            # Match keypoints
            matches = bf.match(des_query, des_train)
            matches.sort(key=lambda x: x.distance)

            # Calculate the total distance of the top matches
            top_matches = matches[:min(10, len(matches))]
            match_score = sum([match.distance for match in top_matches])

            # Update the best match
            if match_score < best_match_score:
                best_match = file
                best_match_score = match_score

    return os.path.splitext(best_match)[0]

def akaze_comparison(img_query, folder_path):
    # Initialize AKAZE
    akaze = cv2.AKAZE_create()

    # Find keypoints and descriptors for the query image
    kp_query, des_query = akaze.detectAndCompute(img_query, None)

    # Initialize BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # The rest of the function is the same as the sift_comparison function
    # ...
    best_match = None
    best_match_score = float('inf')

    for file in os.listdir(folder_path):
        if file.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder_path, file)
            img_train = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            # Find keypoints and descriptors for the train image
            kp_train, des_train = akaze.detectAndCompute(img_train, None)

            # Match keypoints
            matches = bf.match(des_query, des_train)
            matches.sort(key=lambda x: x.distance)

            # Calculate the total distance of the top matches
            top_matches = matches[:min(10, len(matches))]
            match_score = sum([match.distance for match in top_matches])

            # Update the best match
            if match_score < best_match_score:
                best_match = file
                best_match_score = match_score

    return os.path.splitext(best_match)[0]