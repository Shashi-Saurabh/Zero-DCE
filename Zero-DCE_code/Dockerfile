# Use official Python image
FROM python:3.9-slim

WORKDIR /app
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port (FastAPI default)
EXPOSE 8000

# Launch
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
