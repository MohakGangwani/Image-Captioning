import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def initialize():
    
    # Load the InceptionV3 model pre-trained on ImageNet
    global cnn_model
    cnn_model = InceptionV3(weights='imagenet')
    cnn_model = Model(inputs=cnn_model.input, outputs=cnn_model.layers[-2].output)
    
    # Reading Captions
    file = open("captions.txt", "r")
    lines = file.read().split("\n")
    file.close()
    global captions
    captions = pd.DataFrame(data = [line.split(',', 1) for line in lines], columns=['img', 'caption'])
    captions = captions.dropna()
    new_header = captions.iloc[0]
    captions = captions[1:]
    captions.columns = new_header
    captions = captions.sample(frac=1)
    
    # Create a tokenizer to convert text to sequences
    global tkoenizer
    tokenizer = Tokenizer(num_words=5000)  # Only consider the top 5000 words
    tokenizer.fit_on_texts(captions)  # captions is a list of all image captions

    # Convert captions to sequences of integers
    captions_list = list(captions['caption'])
    sequences = tokenizer.texts_to_sequences(captions_list)
    global max_length
    max_length = max(len(seq) for seq in sequences)  # Max length of any caption

    # Pad sequences to ensure consistent length
    global padded_sequences
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')


# Function to preprocess an image
def preprocess_img(img_path):
    img = image.load_img(img_path, target_size=(299, 299))
    img.show()
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

# Function to extract features from a preprocessed image
def extract_img_features(preprocessed_img):
    return cnn_model.predict(preprocess_img)


def main():
    initialize()    
    

if __name__ == '__main__':
    main()