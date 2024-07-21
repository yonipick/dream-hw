# Stage 1: Training
FROM python:3.8-slim AS train

# Install dependencies
RUN pip install pandas scikit-learn joblib

# Copy code
COPY data_processing.py model_training.py /app/

# Set working directory
WORKDIR /app

# Run training
CMD ["python", "-c", "from data_processing import load_data, preprocess_data, split_data; from model_training import train_model; url = 'https://drive.google.com/uc?export=download&id=1kqnB4J8FuF1k8xLIvfbPE1jsqUwd3wVH'; dataset = load_data(url); df_final = preprocess_data(dataset); X, _, Y = split_data(df_final); train_model(X, Y, 'RFR')"]

# Stage 2: Serving
FROM python:3.8-slim AS serve

# Install dependencies
RUN pip install flask pandas scikit-learn joblib

# Copy code and model from the training stage
COPY --from=train /app/RFR_model.pkl /app/
COPY app.py data_processing.py model_inference.py /app/

# Set working directory
WORKDIR /app

# Run Flask app
CMD ["python", "app.py"]
