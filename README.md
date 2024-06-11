# Dental X-Ray Classifier

This project aims to develop a model capable of classifying dental X-ray images into two categories: "child" and "adult". The model is trained on a dataset consisting of a collection of dental panoramic X-ray (OPG) images segmented into children's and adults' X-rays. The dataset is preprocessed to extract relevant features and labels for training the model. The trained model can be used to make predictions on new X-ray images for classification. Additionally, the project provides scripts for evaluating model performance and saving the model for future use.

## How to Use

1. **Download the Repository**: Clone this repository to your local machine using the following command:

    ```bash
    git clone https://github.com/moeez-ktk/Child-Adult-Dental-XRay-Classifier-Model.git
    ```

2. **Install Dependencies**: Navigate to the project directory and install the required dependencies:

    ```bash
    cd dental-xray-classifier
    pip install -r requirements.txt
    ```

3. **Preprocess and Train the Model**: Run the `data_preparation.py` script to preprocess the dataset and split it into training and validation sets. Then, execute the `model_training.py` script to train the deep learning model:

    ```bash
    python data_preparation.py
    python model_training.py
    ```

4. **Evaluate Model Performance**: After training, you can evaluate the model's performance using the `model_evaluation.py` script:

    ```bash
    python model_evaluation.py
    ```

5. **Make Predictions**: Once the model is trained and evaluated, you can use it to make predictions on new X-ray images. Use the `predict.py` script. Before running the script, add appropriate path to the image in the code:

    ```bash
    python predict.py
    ```

## Model Training

This model is trained using a convolutional neural network (CNN) architecture. CNNs are well-suited for image classification tasks due to their ability to automatically learn hierarchical features from the input images. The model architecture consists of multiple convolutional layers followed by max-pooling layers for feature extraction, followed by fully connected layers for classification. The model is trained using the Adam optimizer with binary cross-entropy loss function, as it is a binary classification problem. The dataset is split into training and validation sets to monitor the model's performance during training and prevent overfitting.
