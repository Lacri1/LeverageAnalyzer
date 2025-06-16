# Leverage Analyzer

A web-based dashboard application that visualizes the prediction model for TQQQ's actual leverage ratio. 
It utilizes a deep learning model trained in [LeverageGenerator](https://github.com/Lacri1/LeverageGenerator) to calculate and visualize TQQQ's expected returns, even for periods before TQQQ's inception.


## Screenshot

![Leverage Analyzer Dashboard](assets/predicted_TQQQ_backtest_1999_2025.png)

## Key Features

- **Custom Date Range Analysis**: Analyze data from March 10, 1999 (QQQ listing date) to present
- **Backtesting Capability**: Simulate TQQQ performance for periods before its actual inception (February 9, 2010)
- **Real-time Predictions**: Predict TQQQ leverage ratios using a trained deep learning model
- **Visualization**: Compare actual TQQQ returns with predicted returns through interactive charts
- **Responsive Design**: Optimized UI/UX for both desktop and mobile environments
- **Historical Simulation**: Understand how TQQQ would have performed during major market events before its launch

## Technology Stack

- **Backend**: Python Flask
- **Frontend**: HTML5, CSS3, JavaScript (Chart.js)
- **Data Processing**: pandas, numpy
- **Machine Learning**: TensorFlow, scikit-learn

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Lacri1/tqqq-leverage-analyzer.git
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

### Backtesting Before TQQQ's Inception
For periods before February 9, 2010 (TQQQ's launch date), the application simulates TQQQ's performance using:
- Historical QQQ data
- The trained leverage prediction model
- Daily rebalancing to maintain 3x leverage

This allows you to analyze how TQQQ would have performed during significant market events like the Dot-com bubble and the 2008 financial crisis.

## Recent Updates

- Added backtesting capability for periods before TQQQ's inception
- Improved model accuracy for extreme market conditions
- Enhanced visualization of predicted vs actual leverage ratios
- Added support for custom date range analysis

## Model Information

This application uses the following model files:

- `leverage_model.keras`: Trained deep learning model
- `leverage_scaler.pkl`: Scaler for feature scaling
- `model_input_features.json`: List of model input features

For more details about the model, please refer to the [LeverageGenerator](https://github.com/Lacri1/LeverageGenerator) repository.

## Deployment

### AWS EKS Deployment

This application can be deployed on AWS Elastic Kubernetes Service (EKS). Here's a general overview of the steps involved:

1.  **Prepare Docker Image**
    -   Build your Docker image:
        ```bash
        docker build -t leverage-analyzer:latest .
        ```
    -   Authenticate Docker to your Amazon ECR registry:
        ```bash
        aws ecr get-login-password --region <your-region> | docker login --username AWS --password-stdin <your-account-id>.dkr.ecr.<your-region>.amazonaws.com
        ```
    -   Tag your image for ECR:
        ```bash
        docker tag leverage-analyzer:latest <your-account-id>.dkr.ecr.<your-region>.amazonaws.com/leverage-analyzer:latest
        ```
    -   Push the image to ECR:
        ```bash
        docker push <your-account-id>.dkr.ecr.<your-region>.amazonaws.com/leverage-analyzer:latest
        ```

2.  **Set Up AWS EKS Cluster**
    -   Create an EKS cluster (e.g., using `eksctl` or AWS Console).
    -   Configure `kubectl` to connect to your EKS cluster.

3.  **Deploy to EKS**
    -   Create Kubernetes deployment and service YAML files (e.g., `deployment.yaml`, `service.yaml`).
        *Example `deployment.yaml` (simplified):*
        ```yaml
        apiVersion: apps/v1
        kind: Deployment
        metadata:
          name: leverage-analyzer-deployment
        spec:
          replicas: 1
          selector:
            matchLabels:
              app: leverage-analyzer
          template:
            metadata:
              labels:
                app: leverage-analyzer
            spec:
              containers:
              - name: leverage-analyzer
                image: <your-account-id>.dkr.ecr.<your-region>.amazonaws.com/leverage-analyzer:latest
                ports:
                - containerPort: 5000
        ```
        *Example `service.yaml` (simplified for LoadBalancer):*
        ```yaml
        apiVersion: v1
        kind: Service
        metadata:
          name: leverage-analyzer-service
        spec:
          selector:
            app: leverage-analyzer
          type: LoadBalancer
          ports:
            - protocol: TCP
              port: 80
              targetPort: 5000
        ```
    -   Apply the Kubernetes configurations:
        ```bash
        kubectl apply -f deployment.yaml
        kubectl apply -f service.yaml
        ```

4.  **Verify Deployment**
    -   Check the status of your pods:
        ```bash
        kubectl get pods
        ```
    -   Get the external URL of your service:
        ```bash
        kubectl get service leverage-analyzer-service
        ```
    -   Access the application via the provided external URL.

### Service Verification

![AWS Deployment Verification](assets/AWS.png)
*Screenshot showing successful deployment on AWS*

## License

This project is licensed under the MIT License.

