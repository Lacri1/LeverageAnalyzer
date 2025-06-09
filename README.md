# Leverage Analyzer

A web-based dashboard application that visualizes the prediction model for TQQQ's actual leverage ratio. 
It utilizes a deep learning model trained in [LeverageGenerator](https://github.com/Lacri1/LeverageGenerator) to calculate and visualize TQQQ's expected returns.


## Key Features

- **Custom Date Range Analysis**: Analyze data from March 10, 1999 (QQQ listing date) to present
- **Real-time Predictions**: Predict TQQQ leverage ratios using a trained deep learning model
- **Visualization**: Compare actual TQQQ returns with predicted returns through interactive charts
- **Responsive Design**: Optimized UI/UX for both desktop and mobile environments

## Technology Stack

- **Backend**: Python Flask
- **Frontend**: HTML5, CSS3, JavaScript (Chart.js)
- **Data Processing**: pandas, numpy
- **Machine Learning**: TensorFlow, scikit-learn

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/tqqq-leverage-analyzer.git
cd tqqq-leverage-analyzer
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# or
source .venv/bin/activate  # macOS/Linux
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
python main.py
```

5. Access in your web browser:
```
http://127.0.0.1:5000
```

## How to Use

1. Select your desired date range (default: last 6 months)
2. Click the "Analyze" button
3. Compare actual TQQQ returns with predicted returns in the chart

## Model Information

This application uses the following model files:

- `leverage_model.keras`: Trained deep learning model
- `leverage_scaler.pkl`: Scaler for feature scaling
- `model_input_features.json`: List of model input features

For more details about the model, please refer to the [LeverageGenerator](https://github.com/Lacri1/LeverageGenerator) repository.

## License

This project is licensed under the MIT License.

