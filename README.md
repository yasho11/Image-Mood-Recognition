# Facial Emotion Recognition Project

## Overview

This project implements a Convolutional Neural Network (CNN) using TensorFlow/Keras to detect facial emotions from images. It includes scripts/notebooks for training the model and Streamlit applications for real-time prediction via webcam or image upload.

For detailed information on the model architecture, training process, and limitations, please refer to the `Facial Emotion Recognition CNN Model Documentation` (if you saved the documentation provided earlier).

## Prerequisites

Before you begin, ensure you have the following installed:

* **Git:** For cloning the repository.
* **Python:** Version 3.8 or later recommended. ([Download Python](https://www.python.org/downloads/))
* **pip:** Python package installer (usually comes with Python).
* **python3-venv:** Module for creating virtual environments (usually comes with Python).

**For GPU Acceleration (Optional but Recommended):**

* **NVIDIA GPU:** A CUDA-compatible NVIDIA graphics card.
* **NVIDIA Drivers:** Appropriate drivers installed for your GPU and Linux distribution (e.g., Arch Linux).
* **CUDA Toolkit & cuDNN:** Compatible versions installed system-wide, matching the requirements for the TensorFlow version listed in `requirements.txt`. Refer to the [TensorFlow GPU documentation](https://www.tensorflow.org/install/gpu) for specific version compatibility and installation instructions for your system.

## Setup Instructions

1.  **Clone the Repository:**
    ```bash
    git clone <https://github.com/yasho11/Image-Mood-Recognition.git>
    ```

2.  **Create Virtual Environment:**
    It's highly recommended to use a virtual environment to manage dependencies.
    ```bash
    python -m venv myenv
    ```
    *(The `.gitignore` file is set up to ignore `myenv/`)*

3.  **Activate Virtual Environment:**
    * **Linux / macOS:**
        ```bash
        source myenv/bin/activate
        ```
    * **Windows (Git Bash):**
        ```bash
        source myenv/Scripts/activate
        ```
    * **Windows (Command Prompt/PowerShell):**
        ```bash
        .\myenv\Scripts\activate
        ```
    You should see `(myenv)` prefixed to your terminal prompt.

4.  **Install Dependencies:**
    Install all the required Python packages.
    ```bash
    pip install -r requirements.txt
    ```

5.  **Prepare the Dataset:**
    * The training and testing image data is **not included** in this repository (it's ignored by `.gitignore` due to potential size and licensing reasons).
    * You need to obtain the facial emotion dataset you intend to use (e.g., FER2013, CK+, AffectNet, RAF-DB, or your own collected data).
    * Create the `images/` directory in the project root:
        ```bash
        mkdir images
        ```
    * Organize your dataset within the `images/` directory following the structure expected by the training scripts and data loaders:
        ```
        images/
        ├── train/
        │   ├── angry/       (contains angry training images)
        │   ├── disgust/     (...)
        │   ├── fear/        (...)
        │   ├── happy/       (...)
        │   ├── neutral/     (...)
        │   ├── sad/         (...)
        │   └── surprise/    (...)
        └── test/
            ├── angry/       (contains angry testing images)
            ├── disgust/     (...)
            ├── fear/        (...)
            ├── happy/       (...)
            ├── neutral/     (...)
            ├── sad/         (...)
            └── surprise/    (...)
        ```

6.  **Place Pre-trained Model(s):**
    * The trained model files (e.g., `.keras` files) are also **ignored** by `.gitignore`.
    * If you have already trained a model, place the saved model file(s) (e.g., `best_emotion_detection_model.keras`, `emotion_detection_model.keras`) in the **root directory** of the project so the Streamlit applications (`app.py`, `app_comparison.py`) can find them.
    * If you haven't trained a model yet, proceed to the training step below (after setting up the data).

## Running the Application(s)

Make sure your virtual environment is activated (`source myenv/bin/activate` or similar).

* **To run the comparison Streamlit app:**
    ```bash
    streamlit run app_comparison.py
    ```
    *(This assumes you have placed two trained `.keras` model files in the root directory and updated the paths `MODEL_PATH_1` and `MODEL_PATH_2` inside the script if necessary)*

* **To run the single-model Streamlit app (if you have `app.py`):**
    ```bash
    streamlit run app.py
    ```
    *(This assumes you have placed one trained `.keras` model file in the root directory and updated `MODEL_PATH` inside the script if necessary)*

## Training the Model (Optional)

If you want to train the model yourself using the dataset you prepared in Step 5:

1.  Ensure the `images/` directory is correctly populated with `train` and `test` subdirectories.
2.  **(If applicable)** Convert the Jupyter notebook cells used for training into a Python script (e.g., `train.py`).
3.  Run the training script:
    ```bash
    python train.py
    ```
    *(Adjust the command if your training script has a different name)*
4.  The script should output training progress and save the trained model file(s) (e.g., `best_emotion_detection_model.keras`) to the location specified within the script (ideally the project root or a dedicated `models/` folder - ensure this folder is also in `.gitignore` if desired).

## Project Structure (Example)

```
.
├── .gitignore                 # Specifies intentionally untracked files
├── README.md                  # This setup guide
├── requirements.txt           # Python dependencies
├── app.py                     # Streamlit app (single model) - Optional
├── app_comparison.py          # Streamlit app (model comparison)
├── train.py                   # Model training script - Optional
├── best_emotion_detection_model.keras # Trained model file (Needs to be placed here)
├── emotion_detection_model.keras    # Trained model file (Needs to be placed here)
├── myenv/                     # Virtual environment directory (Ignored by Git)
└── images/                    # Dataset directory (Ignored by Git - Needs to be created)
    ├── train/
    │   └── ... (emotion folders)
    └── test/
        └── ... (emotion folders)
```

## Troubleshooting

* **GPU Not Detected by TensorFlow:** Ensure NVIDIA drivers, CUDA, and cuDNN are installed correctly and their versions are compatible with your TensorFlow version. Run the Python GPU verification script provided in previous discussions or check `nvidia-smi`.
* **`ModuleNotFoundError`:** Make sure your virtual environment (`myenv`) is activated before running `pip install` or executing any Python scripts/Streamlit apps.
* **Webcam Issues:** Ensure your browser has permission to access the webcam. Check if other applications are using the webcam. `streamlit-webrtc` issues can sometimes be complex depending on network/browser configurations.

---
