# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy the dependencies file and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the content of the local src directory to the working directory
COPY processor.py .

# Command to run the application
CMD ["python", "processor.py"]