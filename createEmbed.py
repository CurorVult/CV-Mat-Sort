import numpy as np
import os
import embeddingFunc

def create_embeddings(directory_path):
    # Load images
    image_paths = [os.path.join(directory_path, file) for file in os.listdir(directory_path) if file.endswith(('.png', '.jpg', '.jpeg'))]

    # Calculate and save embeddings
    image_embeddings = [embeddingFunc.embedding_function(np.expand_dims(embeddingFunc.load_preprocess_image(image_path), axis=0)) for image_path in image_paths]
    image_embeddings = np.squeeze(np.array(image_embeddings))
    np.save('image_embeddings.npy', image_embeddings)

    # Save image paths
    with open('image_paths.txt', 'w') as f:
        for image_path in image_paths:
            f.write(image_path + '\n')


create_embeddings('C:\\Users\\Sean\Documents\\GitHub\\CV-Mat-Sort\\Phyrexia_ All Will Be One_images')