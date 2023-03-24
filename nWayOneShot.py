import numpy as np

def n_way_one_shot_evaluation(siamese_model, support_set, test_image, N):
    support_embeddings = siamese_model.predict(support_set)
    test_embedding = siamese_model.predict(np.expand_dims(test_image, axis=0))

    similarities = np.dot(support_embeddings, test_embedding.T)
    predicted_class = np.argmax(similarities)

    return predicted_class
