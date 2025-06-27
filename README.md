# 🎙️ Emotion Recognition from Speech using Deep Learning

This project involves building a deep learning model to recognize human emotions (e.g., happy, sad, angry) from speech audio files using speech signal processing techniques and neural networks.

---

## 🧠 Project Objective

To accurately classify human emotions from audio recordings by extracting relevant features (MFCCs) and training a deep learning model on labeled speech datasets.

---

## 📂 Dataset

- **RAVDESS** (Ryerson Audio-Visual Database of Emotional Speech and Song)
- Format: `.wav` audio files labeled with different emotions
- Download link: [RAVDESS Dataset](https://zenodo.org/record/1188976)

---

## 🧪 Technologies Used

| Category         | Tools / Libraries                      |
|------------------|----------------------------------------|
| Language         | Python                                 |
| ML Framework     | TensorFlow, Keras                      |
| Audio Processing | Librosa, Soundfile                     |
| Data Handling    | NumPy, Pandas, Scikit-learn            |
| Visualization    | Matplotlib                             |
| Environment      | Jupyter Notebook / Google Colab        |

---

## 🎯 Features

- Extracts **MFCC (Mel-Frequency Cepstral Coefficients)** from speech audio.
- Trains a **CNN / LSTM model** for multi-class emotion classification.
- Supports **datasets like RAVDESS, TESS, EMO-DB**.
- Evaluates performance using accuracy and confusion matrix.

---

## ⚙️ How It Works

1. **Preprocess Data**  
   - Load `.wav` files
   - Extract MFCC features using `librosa`
   - Map audio files to emotion labels

2. **Train Model**  
   - Build a CNN or LSTM using TensorFlow/Keras
   - Train and validate the model
   - Save the trained model as `.h5`

3. **Evaluate**  
   - Test the model on unseen data
   - Display accuracy, classification report, and confusion matrix

---

## 🚀 How to Run

1. **Install Requirements**
   ```bash
   pip install librosa soundfile numpy pandas scikit-learn matplotlib tensorflow
