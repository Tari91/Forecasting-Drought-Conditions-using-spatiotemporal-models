# ConvLSTM Drought Forecasting 

## Overview

This project demonstrates a spatiotemporal forecasting model for synthetic drought data using **ConvLSTM** networks in TensorFlow/Keras. The model predicts future drought conditions (SPI values) based on historical data.

## Features

* Synthetic data generation simulating SPI values with spatial and temporal correlations.
* ConvLSTM-based deep learning model to capture spatiotemporal dependencies.
* Visualization of model predictions versus ground truth.

## Requirements

* Python 3.8+
* TensorFlow 2.x
* NumPy
* Matplotlib

Install dependencies using pip:

```bash
pip install tensorflow numpy matplotlib
```

## Usage

1. Save the script `convlstm_drought_forecast.py` locally.
2. Run the script:

```bash
python convlstm_drought_forecast.py
```

3. The script will:

   * Generate synthetic spatiotemporal drought data.
   * Train a ConvLSTM model to predict the next month.
   * Display visualizations comparing last observed month, ground truth, and model prediction.

## File Structure

* `convlstm_drought_forecast.py` : Main script containing data generation, model definition, training, and visualization.

## Notes

* The data is synthetic for demonstration purposes. For real-world applications, replace the data generation function with actual SPI datasets.
* Model and training parameters (filters, timesteps, epochs) can be tuned for different datasets.

## License

This project is provided under the MIT License.
---
Enjoy experimenting with spatiotemporal forecasting using ConvLSTM!

## Author

William Tarinabo, williamtarinabo@gmail.com
