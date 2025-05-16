# Plant-Disease-Detection
This project was created to understand the machine learning and data science by using python library Streamlit


✅ To run app.py (Streamlit web app):
🔧 Requirements:

Make sure you have these installed:

pip install streamlit tensorflow tensorflow_hub pillow numpy

▶️ Run the app:

streamlit run app.py

Then open the URL shown in your terminal (usually http://localhost:8501) in your browser.
✅ To run train.py (Model Training):
🔧 Requirements:

Make sure you have these installed:

pip install tensorflow

🧠 Data Folder Required:

The script expects the following folders:
        /content/Plant_Deasease_Detection/train
        /content/Plant_Deasease_Detection/valid

If you're not using Google Colab, update the paths to your local directory structure, e.g.:

train_dir = "data/train"
valid_dir = "data/valid"

▶️ Run the script:

python train.py

It will:

    Load your dataset.

    Train an EfficientNet model in two phases.

    Save the model as plant_disease_model.h5.


