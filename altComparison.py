
import cv2
import numpy as np
import os
import embeddingFunc
# from createEmbed import image_paths


#Global Variables for image embeddings
loaded_image_embeddings=embeddingFunc.image_embeddings
loaded_image_paths =embeddingFunc.image_paths



def comparison_by_method(img_warped, folder_path, method, conn=None):
    if method == 'SIFT':
        return sift_comparison(img_warped, folder_path)
    elif method == 'BRISK':
        return brisk_comparison(img_warped, folder_path)
    elif method == 'ORB':
        return orb_comparison(img_warped, folder_path)
    elif method == 'AKAZE':
        return akaze_comparison(img_warped, folder_path)
    elif method == 'EMBEDDING':
        return comparison_by_embedding(img_warped)
    elif method == 'AKAZEDB':
        return  akaze_comparison_db(img_warped, conn)
    else:
        raise ValueError(f"Invalid method: {method}")

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

    return os.path.splitext(best_match)[0],None,None


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
                
    return matching_image,None,None
    





def brisk_comparison(img_query, folder_path):
    # Initialize BRISK
    brisk = cv2.BRISK_create()

    # Find keypoints and descriptors for the query image
    kp_query, des_query = brisk.detectAndCompute(img_query, None)

    # Initialize BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    best_match = None
    best_match_score = float('inf')

    for file in os.listdir(folder_path):
        if file.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder_path, file)
            img_train = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            # Find keypoints and descriptors for the train image
            kp_train, des_train = brisk.detectAndCompute(img_train, None)

            # Match keypoints
            matches = bf.match(des_query, des_train)
            
            # Convert the matches object to a list before sorting
            matches_list = list(matches)
            matches_list.sort(key=lambda x: x.distance)

            # Calculate the total distance of the top matches
            top_matches = matches_list[:min(10, len(matches_list))]
            match_score = sum([match.distance for match in top_matches])

            # Update the best match
            if match_score < best_match_score:
                best_match = file
                best_match_score = match_score

    return os.path.splitext(best_match)[0],None,None

def akaze_comparison(img_query, folder_path):
    # Initialize AKAZE
    akaze = cv2.AKAZE_create(threshold=0.002)

    # Find keypoints and descriptors for the query image
    kp_query, des_query = akaze.detectAndCompute(img_query, None)

    # Initialize BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

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
            
            # Convert the matches object to a list before sorting
            matches_list = list(matches)
            matches_list.sort(key=lambda x: x.distance)

            # Calculate the total distance of the top matches
            top_matches = matches_list[:min(10, len(matches_list))]
            match_score = sum([match.distance for match in top_matches])

            # Update the best match
            if match_score < best_match_score:
                best_match = file
                best_match_score = match_score

    return os.path.splitext(best_match)[0], kp_query, des_query



    
def comparison_by_embedding(img_warped):
    img_warped_preprocessed = embeddingFunc.preprocess_image(img_warped)
    img_warped_embedding = embeddingFunc.embedding_function(img_warped_preprocessed)
    
    min_distance = float("inf")
    min_distance_index = -1
    
    for index, embedding in enumerate(loaded_image_embeddings):
        distance = np.linalg.norm(img_warped_embedding - embedding)
        if distance < min_distance:
            min_distance = distance
            min_distance_index = index
    matched_image_path = loaded_image_paths[min_distance_index]
    matched_image_filename = os.path.splitext(os.path.basename(matched_image_path))[0]
    return matched_image_filename, None, None

def akaze_comparison_db(img_query, conn):
    # Initialize AKAZE
    akaze = cv2.AKAZE_create(threshold=0.002)

    # Find keypoints and descriptors for the query image
    kp_query, des_query = akaze.detectAndCompute(img_query, None)

    # Initialize BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    best_match = None
    best_match_score = float('inf')

    # Fetch stored AKAZE features from the database
    cursor = conn.cursor()
    cursor.execute("SELECT id, akaze FROM cards")
    rows = cursor.fetchall()

    for row in rows:
        image_id, akaze_features_str = row

        if akaze_features_str is not None:
            # Convert the string representation of AKAZE features back to a NumPy array
            akaze_features = np.fromstring(akaze_features_str, sep=',').astype(np.uint8).reshape(-1, 61)

            # Match keypoints
            matches = bf.match(des_query, akaze_features)

            # Convert the matches object to a list before sorting
            matches_list = list(matches)
            matches_list.sort(key=lambda x: x.distance)

            # Calculate the total distance of the top matches
            top_matches = matches_list[:min(10, len(matches_list))]
            match_score = sum([match.distance for match in top_matches])

            # Update the best match
            if match_score < best_match_score:
                best_match = image_id
                best_match_score = match_score

    return best_match, kp_query, des_query
