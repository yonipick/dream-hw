# Stage 1: Training
FROM python:3.9-slim AS train

# Install dependencies
RUN pip install pandas scikit-learn joblib

# Copy code
COPY dream/service/data_processing.py dream/service/model_training.py /dream/service/
RUN mkdir -p /dream/models
RUN mkdir -p /dream/dataframes

# Set working directory
WORKDIR /dream

# Run training
CMD ["python", "/dream/service/model_training.py"]


# Stage 2: Serving
FROM python:3.9-slim AS serve

# Install dependencies
RUN pip install flask pandas scikit-learn joblib

# Copy code and model from the training stage
COPY --from=train dream/models/model.pkl dream/models/
COPY --from=train dream/models/encoder.pkl dream/models/
COPY dream/app/app.py /dream/app/
COPY dream/service/data_processing.py dream/service/model_inference.py /dream/service/

# Set working directory
WORKDIR /dream

# Run Flask app
CMD ["python", "/dream/app/app.py"]
