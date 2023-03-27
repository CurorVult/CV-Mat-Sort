
import cv2
import numpy as np
import os
import embeddingFunc
import createEmbed

#Global Variables for image embeddings
loaded_image_embeddings = np.load('image_embeddings.npy')
with open('image_paths.txt', 'r') as f:
    loaded_image_paths = [line.strip() for line in f.readlines()]



def comparison_by_method(img_warped, folder_path, method):
    if method == 'SIFT':
        return sift_comparison(img_warped, folder_path)
    elif method == 'BRISK':
        return brisk_comparison(img_warped, folder_path)
    elif method == 'ORB':
        return orb_comparison(img_warped, folder_path)
    elif method == 'AKAZE':
        return akaze_comparison(img_warped, folder_path)
    elif method == 'IDENTIFY':
        return comparison_by_identification(img_warped, folder_path)
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

    return os.path.splitext(best_match)[0]

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


def identify_image(query_image_path, image_embeddings, threshold=0.5):
    query_image = embeddingFunc.load_preprocess_image(query_image_path)
    query_embedding = embeddingFunc.embedding_function(np.expand_dims(query_image, axis=0))

    distances = np.linalg.norm(image_embeddings - query_embedding, axis=1)
    closest_index = np.argmin(distances)

    if distances[closest_index] < threshold:
        return loaded_image_paths[closest_index], distances[closest_index]
    else:
        return None, None

def comparison_by_identification(img_warped, folder_path):
    # Save the warped image as a temporary file
    temp_image_path = os.path.join(folder_path, 'temp_image.jpg')
    cv2.imwrite(temp_image_path, img_warped)

    # Call the identify_image function with the temporary image path
    identified_image, distance = identify_image(temp_image_path, loaded_image_embeddings)

    # Remove the temporary image file
    os.remove(temp_image_path)

    
    if identified_image:
        # Extract the image name (ID) from the identified image path
        image_name = os.path.splitext(os.path.basename(identified_image))[0]
        return image_name, None, None
    else:
        return None, None, None