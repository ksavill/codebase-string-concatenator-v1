# Use a slim Python 3.8 image as the base
FROM python:3.8-slim

# Install Git, required for cloning and pulling repositories
RUN apt-get update && apt-get install -y git

# Copy the requirements file and install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

# Copy the application code and index.html
COPY main.py /app/main.py
COPY index.html /app/index.html

# Set the working directory
WORKDIR /app

# Expose the port the app runs on
EXPOSE 32546

# Command to run the application
CMD ["python", "main.py"]