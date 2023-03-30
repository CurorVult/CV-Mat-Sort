import cv2
import os
import numpy as np
import pymysql


# Connect to MySQL database
def connect_to_database():
    config = {
        'host': 'localhost',
        'port': 3306,
        'user': 'root',
        'password': 'Mvemjsunp9',
        'database': 'cardscandb'
    }

    try:
        conn = pymysql.connect(**config)
        if conn.open:
            print("Connected to MySQL database")
            return conn
    except pymysql.Error as e:
        print(f"Error connecting to MySQL database: {e}")

# Connect to the database
conn = connect_to_database()

# Store AKAZE features in the database
def store_akaze_features(conn, image_id, features):
    cursor = conn.cursor()
    try:
        # Convert AKAZE features to a string representation
        features_str = ",".join(map(str, features.flatten()))
        
        # Update the row in the cards table
        query = f"UPDATE cards SET akaze = '{features_str}' WHERE id = '{image_id}'"
        cursor.execute(query)
        conn.commit()
    except pymysql.Error as e:
        print(f"The error '{e}' occurred")

# Extract AKAZE features from an image
def extract_akaze_features(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    akaze = cv2.AKAZE_create()
    _, descriptors = akaze.detectAndCompute(image, None)
    
    return descriptors

# Main function
def main():
    folder_path = "\\Users\\Sean\\Documents\\GitHub\\CV-Mat-Sort\\Phyrexia_ All Will Be One_images"


    # Process images
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_id, _ = os.path.splitext(filename)
            image_path = os.path.join(folder_path, filename)
            
            print(f"Processing image {filename}")
            akaze_features = extract_akaze_features(image_path)
            
            if akaze_features is not None:
                store_akaze_features(conn, image_id, akaze_features)
            else:
                print(f"Could not extract AKAZE features for image {filename}")

    conn.close()

if __name__ == "__main__":
    main()