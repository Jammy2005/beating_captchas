
## Acknowledgement
A large portion of this code is taken from Python Lessons implementation of a captcha solver using TensorFlow OCR. Here are his provided links: 

https://youtu.be/z_6P0PilBmM?si=An5ASelzyYG6ng9z
https://pylessons.com/tensorflow-ocr-captcha
https://github.com/pythonlessons/mltu/tree/main/Tutorials/02_captcha_to_text

For a more in depth understanding of this code please check out these resources.

model.py -
    Defines the model architechure:

                            Model Architecture Description
    The model architecture is designed to handle tasks involving spatial and sequential data, such as OCR, speech recognition, or time-series predictions. It begins with an input layer that accepts data of shape input_dim, followed by a normalization step to scale pixel values between 0 and 1. The backbone of the model comprises a series of residual blocks, progressively increasing the number of filters from 16 to 256 while reducing spatial dimensions through downsampling. These residual blocks efficiently capture hierarchical features with the aid of skip connections and Leaky ReLU activations, complemented by dropout for regularization.

    After the residual blocks, the spatial feature maps are reshaped into a sequential format suitable for RNN processing. Two bidirectional LSTM (BLSTM) layers follow, each with 256 units in both forward and backward directions. These layers are designed to extract temporal dependencies and provide bidirectional context. Finally, the model outputs class probabilities for each time step using a dense layer with a softmax activation, where the output dimension corresponds to the number of classes plus one (to account for special tokens or padding). This architecture combines the strengths of convolutional feature extraction and recurrent sequence modeling, making it robust and versatile for sequence-based tasks.

train.py - 
    This script trains a deep learning model for CAPTCHA recognition using TensorFlow and a custom framework. It performs the following steps:

        Dataset Preparation: Downloads a CAPTCHA dataset, extracts images, and processes the labels. The labels are encoded, and the images are resized to a fixed shape. Padding and vocabulary configuration are also handled.

        Data Augmentation: Augments training data with random brightness adjustments, rotations, and erosion/dilation to improve model generalization.

        Model Architecture: Uses a convolutional-recurrent neural network (CRNN) with residual blocks for feature extraction and bidirectional LSTMs for sequence processing. The model outputs a sequence of characters representing the CAPTCHA text.

        Model Compilation: The model is compiled with a CTC loss function to handle variable-length outputs and a Character Word Error Rate (CWER) metric for evaluation.

        Training: The script splits the dataset into training and validation sets and trains the model using callbacks for early stopping, model saving, learning rate adjustments, logging, and exporting the trained model to ONNX format.

        Result Saving: Saves training and validation datasets to CSV files and the best-performing model to a specified directory.

evaluation_testing.py

    This script performs inference using a trained CAPTCHA recognition model in ONNX format. It includes a custom ImageToWordModel class, which preprocesses images, runs them through the model, and decodes the output into human-readable text using a CTC decoder. The script evaluates the model on a dataset of CAPTCHA images by calculating the Character Error Rate (CER) for each prediction and outputs the average CER as a performance metric. Additionally, it allows testing specific images for quick verification. This makes the script useful for evaluating the accuracy of the model and validating its performance on new or unseen CAPTCHA samples.

configs.py

    The ModelConfigs script defines essential configuration settings for training a machine learning model aimed at converting CAPTCHA images into text. It extends the BaseModelConfigs class from the mltu.configs module, providing customizable parameters for the model's training process. The script dynamically generates a unique model directory based on the current timestamp, ensuring organized versioning of trained models. Key configuration attributes include input image dimensions (height and width), vocabulary set, maximum text length, and critical training hyperparameters such as batch size, learning rate, number of training epochs, and the number of worker threads utilized during training. This script serves as a foundation for configuring and fine-tuning the model's performance based on specific requirements.

    *Generative AI was used for script descriptions*