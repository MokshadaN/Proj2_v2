# Use a slim version of Python 3.11 as the base image
FROM python:3.11-slim

# Set the working directory inside the container to /app
WORKDIR /app

# Copy the requirements.txt file from your local machine into the container at /app
COPY requirements.txt .

# Run pip to install all the dependencies listed in requirements.txt
# --no-cache-dir ensures we don't store the download cache, keeping the image smaller
# --upgrade ensures we get the latest patch versions of the libraries
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# This image is now a self-contained environment with all our tools pre-installed.
