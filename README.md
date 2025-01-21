
# Project Title

A brief description of what this project does and who it's for

# **📸 Image Captioning with Deep Learning 🧠**

This project implements an end-to-end **image captioning system** using **deep learning**. The model generates meaningful captions for input images by combining **CNNs (InceptionV3)** for image feature extraction and **LSTMs** for sequence generation. Hyperparameter tuning is conducted using **Optuna**, and all experiments are logged with **MLflow**. 🚀

---

## **📋 Table of Contents**
- 1. [📖 Overview](#overview)  
- 2. [✨ Features](#features)  
- 3. [🛠️ Technologies Used](#technologies-used)  
- 4. [📂 Project Structure](#project-structure)  
- 5. [⚙️ Setup Instructions](#setup-instructions)  
- 6. [🏋️‍♂️ Training the Model](#training-the-model)  
- 7. [📊 Results](#results)  

---

## **📖 Overview**

This project leverages **InceptionV3** to extract image features and **LSTM-based sequence models** to generate captions. The project includes hyperparameter tuning, regularization, and robust evaluation to achieve high-quality captions.

### **Key Highlights**
- 🔧 **Hyperparameter Optimization**: Used **Optuna** to tune embedding dimensions, dropout rates, and LSTM units.  
- 📏 **Evaluation Metrics**: Captions are evaluated using **BLEU-2**, ensuring relevance and accuracy.  
- 📊 **Experiment Tracking**: All runs, parameters, and metrics are logged using **MLflow**.  
- 🛠️ **Scalable Pipeline**: Modular code for data preprocessing, feature extraction, and model training.  

---

## **✨ Features**

- 🌟 **Image Feature Extraction**: Uses a pre-trained **InceptionV3** model.  
- 📝 **Caption Generation**: LSTM-based sequence generation with optimized hyperparameters.  
- 🔍 **Hyperparameter Tuning**: Integrated **Optuna** for automated parameter search.  
- 🗂️ **MLflow Integration**: Logs metrics, artifacts (models, tokenizer, features), and hyperparameters.  
- 📈 **BLEU Evaluation**: Evaluates the quality of generated captions.  

---

## **🛠️ Technologies Used**

- **Programming Language**: 🐍 Python  
- **Deep Learning Frameworks**: TensorFlow, Keras  
- **Experiment Tracking**: MLflow  
- **Hyperparameter Tuning**: Optuna  
- **Evaluation Metrics**: BLEU (NLTK)  
- **Configuration Management**: YAML  

---

## **📂 Data Source**

The dataset used for this project consists of images and their corresponding captions. The source of the data is:

- **Flickr8k Dataset**: A popular dataset used for image captioning, containing 8,000 images, each paired with five different captions.
- **Data Format**: Images are stored in `.jpg` format, and captions are stored in a `.txt` file, with each line representing an image's caption.

You can download the dataset from [Flickr8k dataset page](http://www.kaggle.com/datasets/adityajn105/flickr8k).

> **Note**: Please ensure that the images and captions are placed in the appropriate directories (`Data/Images/` and `Data/captions.txt`) before starting the training process.

---

## **📂 Project Structure**

```plaintext
.
├── Data/                     # Contains images and captions  
│   ├── captions.txt          # Captions for the images  
│   └── Images/               # Folder containing images  
├── main.py                   # Main script for training and evaluation  
├── mlruns/                   # MLflow logs and metadata  
├── Outputs/                  # Outputs like tokenizer and features.pkl  
│   ├── features.pkl  
│   └── tokenizer.pkl  
├── params.yaml               # YAML configuration for parameters  
├── README.md                 # Project documentation  
├── requirements.txt          # Python dependencies  
└── src/                      # Source code  
    ├── utils.py              # Helper functions for preprocessing, training, etc.  
    ├── test.ipynb            # Notebook for model testing  
```

---

## **⚙️Setup Instructions**
- 1. Clone the Repository

        
    ```
    git clone https://github.com/your_username/image-captioning.git  
    cd image-captioning
    ```
- 2. Create a Virtual Environment

    ```
    python -m venv env  
    source env/bin/activate  # For Linux/Mac  
    env\Scripts\activate     # For Windows  
    ```
- 3. Install Dependencies

    ```
    pip install -r requirements.txt  
    ```
---

## **🏋️‍♂️Training the Model**
- **1. Configure Parameters**

    Edit the params.yaml file to set paths, hyperparameters, and training configurations:
    ```
    input_path:
    images: "Data/Images/"
    captions: "Data/captions.txt"

    output_path:
    features: "Outputs/features.pkl"
    tokenizer: "Outputs/tokenizer.pkl"

    caption_preprocessing:
    prefix: "startseq"
    suffix: "endseq"

    train_ratio: 0.8
    evaluate_sample: 10

    hyperparameters:
    embedding_dim: [128, 256]
    dense_units: [256, 512]
    batch_size: 32
    epochs: 20
    optimizer: ["adam", "rmsprop"]
    image_dropout: [0.25, 0.35]
    text_dropout: [0.25, 0.35]
    ```
- **2. Run Training**

    Execute the main.py script to start training:

    ```
    python main.py  
    ```

    This will:

    - Extract image features using InceptionV3.
    - Preprocess captions and tokenize sequences.
    - Tune hyperparameters with Optuna.
    - Log artifacts and metrics to MLflow.

---

## **📊Results**

Instead of selecting a single model, this project explores various hyperparameter configurations. All models, metrics, and artifacts are logged and visualized using MLflow for performance comparison and analysis.