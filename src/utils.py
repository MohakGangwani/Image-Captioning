import os
import pickle
import random
import numpy as np
from tqdm.notebook import tqdm
from nltk.translate.bleu_score import corpus_bleu
import mlflow
from mlflow.models.signature import infer_signature
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import EarlyStopping
import optuna
from pathlib import Path
import yaml


global code_params
with open('params.yaml', 'r') as f:
    code_params = yaml.load(f, Loader=yaml.SafeLoader)



def load_cnn():
    # load vgg16 model
    cnn_model = InceptionV3()
    # restructure the model
    cnn_model = Model(inputs=cnn_model.inputs, outputs=cnn_model.layers[-2].output)
    # summarize
    print(cnn_model.summary())
    return cnn_model

# Function to preprocess an image
def predict_img_features(cnn_model, img_path):
    img = load_img(img_path, target_size=(299, 299))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return cnn_model.predict(img)


def get_img_features(cnn_model, imgs_path, features_path):
    
    features_file = Path(features_path)
    if features_file.is_file():
        with open(features_path, 'rb') as f:
            features = pickle.load(f)
        return features
    
    features = {}
    for img_name in tqdm(os.listdir(imgs_path)):
        if img_name.endswith('.jpg'):
            features[img_name] = predict_img_features(cnn_model, f"{imgs_path}{img_name}")
    os.makedirs(os.path.dirname(features_path), exist_ok=True)
    pickle.dump(features, open(features_path, 'wb'))
    return features


def get_tokenizer(captions, tokenizer_path):
    
    tokenizer_file = Path(tokenizer_path)
    if tokenizer_file.is_file():
        with open(tokenizer_path, 'rb') as f:
            tokenizer = pickle.load(f)
        return tokenizer
    else:
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(captions)
        os.makedirs(os.path.dirname(tokenizer_path), exist_ok=True)
        with open(tokenizer_path, "wb") as f:
            pickle.dump(tokenizer, f)
        return tokenizer


def img_cap_mapping(caps_path):
    with open(caps_path, 'r') as f:
        next(f)
        captions_doc = f.read()
    
    # create mapping of image to captions
    mapping = {}
    
    # process lines
    for line in tqdm(captions_doc.split('\n')):
        # split the line by comma(,)
        tokens = line.split(',')
        if len(line) < 2:
            continue
        image_id, caption = tokens[0], tokens[1:]
        # remove extension from image ID
        image_id = image_id.split('.')[0]
        # convert caption list to string
        caption = " ".join(caption)
        # create list if needed
        if image_id not in mapping:
            mapping[image_id] = []
        # store the caption
        mapping[image_id].append(caption)
    
    return mapping


def clean_caps(cap_mapping, prefix, suffix):
    for img in cap_mapping.keys():
        preprocessed_caps = []
        for cap in cap_mapping[img]:
            cap = cap.lower()
            cap = cap.replace('[^A-Za-z]', '')
            cap = cap.replace(r'\s+', '')
            cap = prefix + " " + " ".join([word for word in cap.split() if len(word)>1]) + " " + suffix
            preprocessed_caps.append(cap)
        cap_mapping[img] = preprocessed_caps
    return cap_mapping


def preprocess_caps(mapping):
    all_captions = []
    for key in mapping:
        for caption in mapping[key]:
            all_captions.append(caption)


# create data generator to get data in batch (avoids session crash)
def data_generator(data_keys, mapping, features, tokenizer, max_length, vocab_size, batch_size):
    # loop over images
    X1, X2, y = list(), list(), list()
    n = 0
    while 1:
        for key in data_keys:
            n += 1
            captions = mapping[key]
            # process each caption
            for caption in captions:
                # encode the sequence
                seq = tokenizer.texts_to_sequences([caption])[0]
                # split the sequence into X, y pairs
                for i in range(1, len(seq)):
                    # split into input and output pairs
                    in_seq, out_seq = seq[:i], seq[i]
                    # pad input sequence
                    in_seq = pad_sequences([in_seq], maxlen=max_length, padding='post')[0]
                    # encode output sequence
                    out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                    # store the sequences
                    X1.append(features[key+'.jpg'][0])
                    X2.append(in_seq)
                    y.append(out_seq)
            if n == batch_size:
                X1, X2, y = np.array(X1), np.array(X2), np.array(y)
                yield {"image": X1, "text": X2}, y
                X1, X2, y = list(), list(), list()
                n = 0


def generate_caption(image_feature, model, tokenizer, max_length):
    """
    Generate a caption for an image using the trained model.
    """
    in_text = code_params['caption_preprocessing']['prefix']
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length, padding='post')
        y_pred = model.predict([image_feature, sequence], verbose=0)
        predicted_id = np.argmax(y_pred)
        if predicted_id not in tokenizer.index_word:
            break
        predicted_word = tokenizer.index_word[predicted_id]
        if predicted_word == code_params['caption_preprocessing']['suffix']:
            break
        in_text += ' ' + predicted_word
    return in_text


def evaluate_bleu(model, tokenizer, val_images, val_captions, max_length):
    references = []
    candidates = []
    for img_feature, ground_truths in zip(val_images, val_captions):
        generated_caption = generate_caption(img_feature, model, tokenizer, max_length)
        references.append([caption.split() for caption in ground_truths])
        candidates.append(generated_caption.split())
    return corpus_bleu(references, candidates, weights=(0.5, 0.5, 0.0, 0.0))
