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


def build_and_train_model(params, mapping, features, tokenizer, max_length):
    """
    Build and train the image captioning model with given hyperparameters.
    
    params: dict of hyperparameters to tune
    train_data, val_data: Training and validation datasets
    tokenizer: Tokenizer for captions
    max_length: Maximum sequence length for captions
    """
    image_ids = list(mapping.keys())
    split = int(len(image_ids) * code_params['train_ratio'])
    train = image_ids[:split]
    val = image_ids[split:]
    
    vocab_size = len(tokenizer.word_index)+1
    
    # Calculate steps per epoch
    total_train_samples = len(train) * len(mapping[train[0]])  # Total captions for training
    steps_per_epoch = total_train_samples // params['batch_size']
    total_val_samples = len(val) * len(mapping[val[0]])  # Total captions for validation
    validation_steps = total_val_samples // params['batch_size']
    
    # Log the parameters in MLflow
    mlflow.log_params(params)
    
    # Image feature input
    image_input = Input(shape=(2048,), name='image')
    image_dropout = Dropout(params['image_dropout'])(image_input)
    image_dense = Dense(params['dense_units'], activation='relu')(image_dropout)
    
    # Text input and embedding
    text_input = Input(shape=(max_length,), name='text')
    text_embedding = Embedding(input_dim=vocab_size, 
                                       output_dim=params['embedding_dim'], mask_zero=True)(text_input)
    text_dropout = Dropout(params['text_dropout'])(text_embedding)
    text_lstm = LSTM(params['dense_units'])(text_dropout)
    
    # Combine the image and text features
    combined = add([image_dense, text_lstm])
    combined_dense = Dense(params['dense_units'], activation='relu')(combined)
    combined_output = Dense(vocab_size, activation='softmax')(combined_dense)
    
    # Define the model
    model = Model(inputs=[image_input, text_input], outputs=combined_output)
    
    # Compile the model
    model.compile(optimizer=params['optimizer'], 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    
    # Define data generators
    train_generator = data_generator(
        data_keys=train,
        mapping=mapping,
        features=features,
        tokenizer=tokenizer,
        max_length=max_length,
        vocab_size=vocab_size,
        batch_size=params['batch_size']
    )

    val_generator = data_generator(
        data_keys=val,
        mapping=mapping,
        features=features,
        tokenizer=tokenizer,
        max_length=max_length,
        vocab_size=vocab_size,
        batch_size=params['batch_size']
    )
    lr_scheduler = ReduceLROnPlateau(
        monitor='val_loss',  # Reduce LR when validation loss plateaus
        factor=0.5,          # Reduce LR by half
        patience=2,          # Wait 2 epochs before reducing LR
        min_lr=1e-6          # Minimum learning rate
        )
    early_stopping = EarlyStopping(
        monitor='val_accuracy',  # Monitor validation accuracy
        patience=3,              # Stop after 3 epochs without improvement
        restore_best_weights=True  # Revert to the best weights
        )
    callbacks = [MLflowMetricsCallback(), lr_scheduler, early_stopping]
    # Train the model
    model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_generator,
        validation_steps=validation_steps,
        epochs=params['epochs'],
        callbacks=callbacks
    )
    
    mlflow.log_artifact(code_params['output_path']['features'])
    mlflow.log_artifact(code_params['output_path']['tokenizer'])
    
    # Example inputs for model signature
    example_image_id = train[0]  # Use the first training image
    example_image_feature = features[example_image_id + ".jpg"]  # Extract its feature vector

    # Use the first caption for this image
    example_caption = mapping[example_image_id][0]  # e.g., 'startseq a dog is running endseq'
    tokenized_caption = tokenizer.texts_to_sequences([example_caption])[0]
    padded_caption = pad_sequences([tokenized_caption], maxlen=max_length, padding='post')

    # Combine into an input example
    example_input = {
        "image": example_image_feature,
        "text": padded_caption
    }

    # Predict output to infer model signature
    example_output = model.predict(example_input)
    signature = infer_signature(example_input, example_output)
    
    mlflow.keras.log_model(model, artifact_path="model", signature=signature, pip_requirements="requirements.txt")
    return model


def objective(trial):
    # Sample hyperparameters
    params = {
        'embedding_dim': trial.suggest_categorical('embedding_dim', code_params['hyperparameters']['embedding_dim']),
        'dense_units': trial.suggest_categorical('dense_units', code_params['hyperparameters']['dense_units']),
        'batch_size': code_params['hyperparameters']['batch_size'],
        'epochs': code_params['hyperparameters']['epochs'],
        'optimizer': trial.suggest_categorical('optimizer', code_params['hyperparameters']['optimizer']),
        'image_dropout': trial.suggest_categorical('image_dropout', code_params['hyperparameters']['image_dropout']),
        'text_dropout': trial.suggest_categorical('text_dropout', code_params['hyperparameters']['text_dropout']),
    }
    
    # Start an MLflow run
    with mlflow.start_run():
        # Build and train the model
        model = build_and_train_model(params, clean_mapping, features, tokenizer, max_length)
        
        # Prepare validation data for BLEU evaluation
        image_ids = list(clean_mapping.keys())
        split = int(len(image_ids) * code_params['train_ratio'])
        evaluate_keys = random.sample(image_ids[split:], code_params['evaluate_sample'])
        evaluate_images = [features[key + ".jpg"] for key in evaluate_keys]
        evaluate_captions = [clean_mapping[key] for key in evaluate_keys]
        
        # Evaluate BLEU
        evaluate_bleu_score = evaluate_bleu(model, tokenizer, evaluate_images, evaluate_captions, max_length)
        
        # Log BLEU to MLflow
        mlflow.log_metric("BLEU Score", evaluate_bleu_score)
        print(f"Trial BLEU Score: {evaluate_bleu_score}")
        
        return evaluate_bleu_score


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


class MLflowMetricsCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        # Log training metrics
        mlflow.log_metric("train_loss", logs["loss"], step=epoch)
        mlflow.log_metric("train_accuracy", logs["accuracy"], step=epoch)

        # Log validation metrics
        if "val_loss" in logs:
            mlflow.log_metric("val_loss", logs["val_loss"], step=epoch)
            mlflow.log_metric("val_accuracy", logs["val_accuracy"], step=epoch)


def main():
    
    global code_params
    with open('params.yaml', 'r') as f:
        code_params = yaml.load(f, Loader=yaml.SafeLoader)
    
    # Set MLflow tracking
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Image Captioning BLEU Optimization")
    
    # Load CNN model for feature extraction
    cnn_model = load_cnn()
    
    global features, clean_mapping, tokenizer, max_length
    
    # Extract image features
    features = get_img_features(cnn_model=cnn_model, imgs_path=code_params['input_path']['images'], features_path=code_params['output_path']['features'])
    
    # Load and clean captions
    mapping = img_cap_mapping(caps_path=code_params['input_path']['captions'])
    clean_mapping = clean_caps(mapping, prefix=code_params['caption_preprocessing']['prefix'], suffix=code_params['caption_preprocessing']['suffix'])
    
    # Prepare tokenizer and compute max caption length
    all_captions = []
    for key in clean_mapping:
        for caption in clean_mapping[key]:
            all_captions.append(caption)
    
    max_length = max([len(caption.split()) for caption in all_captions])
    tokenizer = get_tokenizer(all_captions, code_params['output_path']['tokenizer'])
    
    # Run Optuna for hyperparameter optimization
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)  # Adjust number of trials as needed
    
    # Print the best parameters
    print(f"Best BLEU Score: {study.best_value}")
    print(f"Best Parameters: {study.best_params}")


if __name__ == "__main__":
    main()