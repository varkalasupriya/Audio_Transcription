# FastAPI Audio Transcription and Summarization Service

## Overview
This FastAPI application uses the Whisper model by OpenAI for automatic speech recognition (ASR) and Hugging Face Transformers for text summarization.

## Features
- Transcribe audio files into text with high accuracy.
- Generate concise summaries of the transcribed text.
- Extract timestamps for each segment of the transcription.

## Endpoints
- **POST /transcribe**: Upload an audio file to transcribe it into text and receive a summary along with timestamps.

### Setup
1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/yourrepository.git
    ```

2. Create a virtual environment and activate it:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

4. Run the application:
    ```bash
    uvicorn main:app --reload
    ```

## Usage
1. Navigate to `http://127.0.0.1:8000/docs` in your web browser.
2. Use the interactive API documentation to upload audio files and get transcriptions, summaries, and timestamps.

