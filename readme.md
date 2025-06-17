
# HOUSE PRICE PREDICTION USING PYTORCH

This repository contains the code and resources for the House Price Prediction using pytorch, aiming to predict house prices based on various features.

## Project Overview

This project focuses on building a robust machine learning model to predict house prices. It leverages various data preprocessing techniques, feature engineering, and a trained model to provide accurate predictions.

## Features

  * **Data Preprocessing:** Handles missing values, encodes categorical features, and scales numerical features.
  * **Model Training:** Trains a machine learning model (likely a regression model) on the processed housing data.
  * **Prediction API/Application:** (Assumed based on `app.py`) Provides an interface to make predictions.
  * **Scalable Preprocessing:** Uses `_std.pkl` files for standardized preprocessing steps, allowing consistent data transformation.

## Installation

To set up the project locally, follow these steps:

1.  **Clone the repository:**

    ```bash
    git clone <your-repository-url>
    cd HOUSE_PRICE_PREDICTION_2.0
    ```

2.  **Create and activate the virtual environment:**
    The `environment.yml` file suggests using Conda.

    ```bash
    conda env create -f environment.yml
    conda activate house_price_prediction_2.0 # Or whatever name conda assigns
    ```

    Alternatively, if using `pip` with `venv`:

    ```bash
    python -m venv .venv
    source .venv/bin/activate # On Windows: .venv\Scripts\activate
    pip install -r requirements.txt # You might need to generate this from environment.yml or manually
    ```

3.  **Install dependencies:**
    If using `environment.yml` with Conda, the dependencies should be installed during `conda env create`. If not, you'll need to install them manually.

## Usage

### Running the Prediction Application (if `app.py` is a web app)

If `app.py` is a Flask/Streamlit/FastAPI application:

```bash
python app.py
```

Then, open your web browser and navigate to the address displayed in the console (e.g., `http://127.0.0.1:5000`).

### Running Notebooks

You can explore the data preprocessing and model training steps by running the Jupyter notebooks:

```bash
jupyter lab
```

Then open `preprocess.ipynb` and `model.ipynb`.

### Making Predictions Programmatically

You can load the trained model and scalers to make predictions within your own Python scripts.

```python
import joblib
import pandas as pd

# Load preprocessors
airconditioning_std = joblib.load('airconditioning_std.pkl')
furnish_std = joblib.load('furnish_std.pkl')
guestroom_std = joblib.load('guestroom_std.pkl')
mainroad_std = joblib.load('mainroad_std.pkl')
prefarea_std = joblib.load('prefarea_std.pkl')
scale_x = joblib.load('scale_x.pkl')
scale_y = joblib.load('scale_y.pkl') # For inverse transforming predictions

# Load the trained model
model = joblib.load('model.pth') # Assuming .pth is a pickled model, not necessarily a PyTorch model without further context

# Example prediction (replace with your actual data and preprocessing steps)
# This is a conceptual example and needs to be adapted based on your exact preprocessing in preprocess.ipynb
new_data = pd.DataFrame({
    'area': [6000],
    'bedrooms': [3],
    'bathrooms': [2],
    'stories': [2],
    'mainroad': ['yes'],
    'guestroom': ['no'],
    'basement': ['yes'],
    'hotwaterheating': ['no'],
    'airconditioning': ['yes'],
    'parking': [1],
    'prefarea': ['yes'],
    'furnishingstatus': ['furnished']
})

# Apply the same preprocessing steps as used during training
# (e.g., one-hot encoding for categorical, scaling for numerical)
# This part would typically be encapsulated in a function or class if app.py exists
# For demonstration:
# encoded_data = encoder.transform(new_data[categorical_cols])
# scaled_numerical = scale_x.transform(new_data[numerical_cols])
# processed_new_data = pd.concat([scaled_numerical, encoded_data], axis=1)

# For simplicity, assuming the model expects raw features after basic handling for demonstration
# You would apply the loaded _std.pkl files and scale_x.pkl here
# This requires understanding how your preprocess.ipynb exactly prepares the data.

# Placeholder for actual preprocessing
# You need to manually apply the transformations from your .pkl files here
# Example:
# new_data['mainroad'] = mainroad_std.transform(new_data[['mainroad']]) # If it's a simple scalar transformer
# For one-hot encoded features, you'd apply the OneHotEncoder that generated 'encoder' if it exists.

# Once your `processed_new_data` is ready:
# prediction_scaled = model.predict(processed_new_data)
# predicted_price = scale_y.inverse_transform(prediction_scaled.reshape(-1, 1))
# print(f"Predicted House Price: ${predicted_price[0][0]:,.2f}")
```

## File Structure

```
.
├── .ipynb_checkpoints/
├── .venv/                      # Python virtual environment
├── encoder/                    # Directory for custom encoders or related files
├── advertising.csv             # Potentially additional dataset or related data
├── airconditioning_std.pkl     # StandardScaler or similar for 'airconditioning' feature
├── app.py                      # Main application file (e.g., Flask/Streamlit app for prediction)
├── environment.yml             # Conda environment file
├── furnish_std.pkl             # StandardScaler or similar for 'furnishingstatus' feature
├── guestroom_std.pkl           # StandardScaler or similar for 'guestroom' feature
├── Housing.csv                 # Original raw housing dataset
├── mainroad_std.pkl            # StandardScaler or similar for 'mainroad' feature
├── model.ipynb                 # Jupyter notebook for model training and evaluation
├── model.pth                   # Trained machine learning model (e.g., serialized using joblib or PyTorch model)
├── prefarea_std.pkl            # StandardScaler or similar for 'prefarea' feature
├── preprocess.ipynb            # Jupyter notebook for data preprocessing and feature engineering
├── process_house.csv           # Processed housing dataset after `preprocess.ipynb`
├── readme.md                   # This README file
├── scale_x.pkl                 # StandardScaler or MinMaxScaler for input features (X)
└── scale_y.pkl                 # StandardScaler or MinMaxScaler for target variable (y - price)
```

## Data

The primary dataset used is `Housing.csv`. This file contains the raw features of houses and their corresponding prices.

  * `Housing.csv`: The initial dataset used for training and testing.
  * `process_house.csv`: The clean and preprocessed dataset, ready for model training, generated by `preprocess.ipynb`.

## Model Training

The `model.ipynb` notebook details the steps for training the machine learning model. This includes:

  * Loading the preprocessed data (`process_house.csv`).
  * Splitting data into training and testing sets.
  * Training the model using the selected algorithm.
  * Evaluating model performance.
  * Saving the trained model (`model.pth`) and scalers (`scale_x.pkl`, `scale_y.pkl`, and individual feature scalers).

## Prediction

Predictions can be made using the `app.py` (if it's a web application) or by loading the `model.pth` and the various `_std.pkl` and `scale_x.pkl`/`scale_y.pkl` files for programmatic predictions.

## Contributing

Contributions are welcome\! If you have suggestions for improvements, new features, or bug fixes, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the LICENSE.md file for details (if you have one, otherwise remove this section or specify your license).

## Contact

For any questions or inquiries, please contact:

  * AVIJIT BHADRA (riyabha4566@gmail.com)

-----