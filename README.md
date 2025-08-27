🌱 Plant Disease Detection using CNN

A deep learning project that detects plant leaf diseases using Convolutional Neural Networks (CNNs). The model classifies healthy vs. diseased leaves with high accuracy, helping farmers and researchers identify plant health issues early.

📌 Features

🧠 CNN-based deep learning model for image classification.

🌿 Detects multiple plant diseases from leaf images.

📊 High training and validation accuracy.

💾 Trained model included (trained_model.keras).

🎯 Easy-to-use prediction script for testing new leaf images.

📂 Project Structure
Plant-Disease-Detection-using-CNN/
│── dataset/                # Training & testing images
│── trained_model.keras     # Saved trained model (~90MB)
│── requirements.txt        # Python dependencies
│── app.py / notebook.ipynb # Main training or prediction code
│── utils.py                # Helper functions (if any)
│── README.md               # Project documentation

⚙️ Installation

1️⃣ Clone the repository:

git clone https://github.com/vishal050505/Plant-Disease-Detection-using-CNN.git
cd Plant-Disease-Detection-using-CNN


2️⃣ Create a virtual environment (recommended):

python -m venv venv
venv\Scripts\activate     # On Windows
source venv/bin/activate  # On Linux/Mac


3️⃣ Install dependencies:

pip install -r requirements.txt

🚀 Usage
🔹 Train the model
python train.py

🔹 Test on an image
python predict.py --image sample_leaf.jpg

🔹 Run in Jupyter Notebook

Open plant_disease_detection.ipynb and run cells step by step.

🧠 Model Architecture

Convolutional Neural Networks (CNN)

ReLU activation + MaxPooling layers

Dropout for regularization

Dense layers with Softmax output

📊 Results

✅ Training Accuracy: ~XX%

✅ Validation Accuracy: ~XX%

✅ Works well on unseen leaf images

(You can update this with your actual metrics & graphs)

📌 Future Improvements

Add support for more plant species.

Deploy as a web/mobile app.

Use Transfer Learning (ResNet, EfficientNet, etc.) for higher accuracy.

Integrate with IoT for real-time field monitoring.

📜 License

This project is licensed under the MIT License – free to use and modify.
