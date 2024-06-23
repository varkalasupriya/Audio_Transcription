from fastapi import FastAPI, File, UploadFile, HTTPException
import whisper
import uvicorn
import os
import shutil
from transformers import pipeline

# Initialize FastAPI app
app = FastAPI()

# Ensure the temporary directory for uploaded files exists
os.makedirs("temp", exist_ok=True)

# Load the Whisper model for audio transcription
# Whisper is a state-of-the-art ASR (Automatic Speech Recognition) model by OpenAI
model = whisper.load_model("base")

# Initialize the summarization pipeline from Hugging Face Transformers
# This pipeline leverages models like BART, T5, etc., for text summarization
summarizer = pipeline("summarization")

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Upload Audio File: Select an audio file from your device to upload.

    Transcription Process: The uploaded audio will be processed using the Whisper model to convert speech into text.

    Summarization: The transcribed text will be summarized using Hugging Face's summarization pipeline.
    
    Timestamps Extraction: Timestamps will be extracted for each segment of the transcription to indicate when each segment starts and ends.

    Returns: A dictionary containing the transcription text, summary, and timestamps.
    """
    try:
        # Save the uploaded file to a temporary directory
        with open(f"temp/{file.filename}", "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Transcribe the audio file using Whisper
        result = model.transcribe(f"temp/{file.filename}")

        # Extract the transcription text from the result
        transcription_text = result["text"]

        # Extract segments with timestamps from the result
        segments = result["segments"]
        timestamps = [
            {
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"]
            }
            for segment in segments
        ]

        # Summarize the transcription text using the Hugging Face summarization pipeline
        summarized_text = summarizer(transcription_text, max_length=150, min_length=30, do_sample=False)[0]['summary_text']

        # Remove the temporary audio file after processing
        os.remove(f"temp/{file.filename}")

        # Return the transcription, summary, and timestamps as the API response
        return {
            "transcription": transcription_text,
            "summary": summarized_text,
            "timestamps": timestamps
        }

    except Exception as e:
        # Raise an HTTP exception in case of any error during the process
        raise HTTPException(status_code=500, detail=str(e))

# Entry point for running the FastAPI application with Uvicorn
if __name__ == "__main__":
    # Start the Uvicorn server to serve the FastAPI application
    uvicorn.run(app, host="0.0.0.0", port=8000)
