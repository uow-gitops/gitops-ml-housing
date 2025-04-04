# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code, including the 'model', 'templates', and 'static' folders.
COPY . .

# Expose port 5000 (or whichever port your Flask app uses)
EXPOSE 5000

# Use Gunicorn to serve the Flask application
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
