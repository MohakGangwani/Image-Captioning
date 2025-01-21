import random
import mlflow
from mlflow.models.signature import infer_signature
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import EarlyStopping
import optuna
import yaml
from src.utils import *


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