# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . /app/

# Create a non-root user (security best practice)
RUN useradd --create-home --shell /bin/bash app
USER app

# Expose port (Cloud Run will override this)
EXPOSE 8080

# Run the application
CMD ["python", "app.py"]