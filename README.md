Handwritten Digit Recognition using OCR Systems

A deep learning-based OCR system designed to recognize handwritten digits using Convolutional Neural Networks (CNNs) and the MNIST dataset. This project integrates TensorFlow/Keras with a Streamlit web application to provide real-time digit prediction, performance visualization, and interactive evaluation.

🚀 Features
Handwritten digit recognition using CNN architecture
Real-time predictions through Streamlit interface
Image preprocessing with grayscale normalization and reshaping
Model evaluation using Accuracy, Precision, Recall, and F1-Score
Visualization tools including confusion matrix and per-class accuracy
High accuracy performance on MNIST dataset (~97.7%)
🛠️ Tech Stack
Python
TensorFlow / Keras
Streamlit
NumPy
Matplotlib / Seaborn
📂 Dataset
Dataset: MNIST Handwritten Digits Dataset
Training Images: 60,000
Testing Images: 10,000
Image Size: 28×28 grayscale images
🧠 Model Architecture

The CNN model includes:

Convolutional Layers
ReLU Activation Functions
MaxPooling Layers
Dropout Layers
Fully Connected Dense Layers
Softmax Output Layer for digit classification (0–9)
📊 Results

The trained model achieved strong performance on the MNIST test dataset:

Accuracy: 97.78%
Precision: 97.82%
Recall: 97.78%
F1 Score: 97.78%
💻 Streamlit Web Application

The application allows users to:

Upload handwritten digit images
Get instant predictions
Visualize model performance metrics in real time
📌 Future Enhancements
Full handwritten text recognition
Multilingual OCR support
Advanced image augmentation techniques
Deployment on cloud platforms
📖 Conclusion

This project demonstrates the effectiveness of deep learning in OCR systems by leveraging CNNs for accurate handwritten digit recognition. It provides a scalable foundation for intelligent document processing and automation applications.# Handwritten-Digit-Recognition-using-OCR-Systems-
