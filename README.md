# Specialist_Recommender

A simple FastAPI-based API that recommends a medical specialist based on user symptoms, age, gender, severity, and duration. It uses the Hugging Face model `OzzeY72/biobert-medical-specialities` and also normalizes common symptom words like **sinus**, **acidity**, **breathlessness**, etc. before prediction. 

## Features

* FastAPI REST API
* Symptom normalization
* Specialist recommendation
* Confidence score in response
* CLI testing support
* CORS enabled for testing

## Project File

* `app.py` → main FastAPI application

## Requirements

Install the required packages:

```bash
pip install fastapi uvicorn torch transformers pydantic
```

If you want to test using a CLI Python script:

```bash
pip install requests
```

## Run the API

Start the FastAPI server with:

```bash
uvicorn app:app --reload
```

If successful, the server will run at:

```text
http://127.0.0.1:8000
```

## Available Endpoints

### 1. Root

```http
GET /
```

Returns a simple message showing that the API is running.

### 2. Health Check

```http
GET /health
```

Returns API status and device info.

### 3. Recommend Specialist

```http
POST /recommend
```

Accepts patient details and returns the predicted specialist.

## Example Request

```json
{
  "age": 56,
  "gender": "f",
  "symptoms": "sinus, headache, fatigue",
  "severity": 4,
  "duration": "3 days"
}
```

## Example Response

```json
{
  "recommended_specialist": "Otolaryngologist (Ear, Nose and Throat Specialist)",
  "confidence": 0.87,
  "model_label": "Otorhinolaryngology",
  "normalized_symptoms": "sinusitis headache fatigue nasal congestion facial pressure"
}
```

## API Testing

### Option 1: Swagger UI

Once the server is running, open:

```text
http://127.0.0.1:8000/docs
```

You can test the `/recommend` endpoint directly from the browser.

### Option 2: CLI Test Script

Create a file like `test_app.py` and send a POST request to:

```text
http://127.0.0.1:8000/recommend
```

Make sure the FastAPI server is running before executing the test script.

## Notes

* This project currently runs on **CPU**. 
* CORS is enabled for testing purposes. 
* There is **no HTML frontend required** for using this API.
* You can test using Swagger UI or a Python CLI script.

## Important

This API is for **recommendation/testing purposes only** and should **not** be treated as a final medical diagnosis.
