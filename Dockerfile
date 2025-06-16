FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Waitress as production server
RUN pip install --no-cache-dir waitress

# Copy model files first
COPY leverage_model.keras .
COPY leverage_scaler.pkl .
COPY model_input_features.json .

# Copy all necessary files
COPY . .

# Set environment variable to ensure proper model loading
ENV PYTHONPATH=/app

# Expose the port the app runs on
EXPOSE 5000

# Command to run the application with Waitress in production
CMD ["waitress-serve", "--port=5000", "--url-scheme=http", "main:app"]
