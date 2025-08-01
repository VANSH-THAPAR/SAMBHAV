# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory
WORKDIR /code

# Copy and install requirements system-wide as root
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Create a non-root user for security
RUN useradd --create-home --shell /bin/bash appuser

# Create necessary directories AND give ownership to the new user
RUN mkdir -p /code/model_cache && chown -R appuser:appuser /code/model_cache
RUN mkdir -p /code/temp_docs && chown -R appuser:appuser /code/temp_docs

# Copy the rest of the application code and give ownership
COPY --chown=appuser:appuser . /code/

# Switch to the non-root user
USER appuser

# Run the Gunicorn server
CMD ["gunicorn", "-w", "1", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:7860", "main:app"]