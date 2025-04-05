# Maintenance Failure Prediction and Monitoring Dashboard

This project presents a predictive maintenance system using a synthetic dataset of milling machine operations. It includes a neural network classifier for predicting machine failure and a Streamlit dashboard for real-time monitoring.

## Project Structure

```
├── ai4i2020.csv            # Dataset
├── app.py                  # Streamlit dashboard
├── main.ipynb              # Training script
├── maintenance_model.keras # Saved Keras model
├── README.md               # This documentation
├── scaler.pkl              # Saved StandardScaler
└── requirements.txt        # Required Python packages
```

## Dataset Description

The dataset simulates machine operations with the following key features:

- **Type**: Product quality level (L/M/H)
- **Air Temperature [K]**
- **Process Temperature [K]**
- **Rotational Speed [rpm]**
- **Torque [Nm]**
- **Tool Wear [min]**
- **Failure Modes**: TWF, HDF, PWF, OSF, RNF
- **Target**: Machine Failure (1 = Failed, 0 = Normal)

## Streamlit Dashboard

A simple dashboard interface where users can:

- Input operational parameters manually
- View prediction in real-time
- Uses loaded model (`maintenance_model.keras`) and scaler (`scaler.pkl`)

## How to Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Dashboard

```bash
streamlit run app.py
```

### 3. Training

```bash
python main.ipynb
```
*build by [Robertino Gladden Narendra]*