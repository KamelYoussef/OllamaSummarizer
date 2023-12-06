"""A FastAPI web application with two endpoints. The application receives input in the form of a human-generated text
or a PDF file, processes it using a language model (Zephyr) using Ollama, and provides the summary of the PDF file.
"""

# Import necessary libraries
from fastapi import FastAPI, File, UploadFile, HTTPException
import fitz
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import Ollama
import os
from src.PDFProcessor import PDFProcessor


# Define data models using Pydantic
class Input(BaseModel):
    human_input: str


class Output(BaseModel):
    output: str


# Create a FastAPI application instance
app = FastAPI()

# Configure Cross-Origin Resource Sharing (CORS) middleware
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Defining our locAL LLM using Ollama
llm = Ollama(
    model="zephyr",  # Specify the language model to use
    verbose=True,
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
)


# Function to handle conversation processing using llm
def conversation(human_input):
    prompt = human_input

    output = llm(prompt)
    return output


# Define the "/conversation" endpoint to handle text input
@app.post("/conversation")
async def input(input: Input):
    # Process the input using the conversation function
    output = Output(output=conversation(input.human_input))
    return output


# Define the "/file/upload" endpoint to handle PDF file uploads
@app.post("/file/upload")
async def upload_file(uploaded_file: UploadFile = File(...)):
    # Check if the uploaded file is a PDF
    if uploaded_file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="File must be a PDF")

    # Define the directory to save the PDF file
    save_directory = "uploaded_pdfs"

    # Create the directory if it doesn't exist
    os.makedirs(save_directory, exist_ok=True)

    # Define the file path to save the PDF
    save_path = os.path.join(save_directory, uploaded_file.filename)

    # Open the PDF file using PyMuPDF
    text = fitz.open(save_path)

    # Instantiate PDFProcessor
    pdf_processor = PDFProcessor(text, llm)

    # Call the process method to perform PDF processing
    pdf_processor.process()

    # Return a JSON response indicating success along with the filename
    return {"filename": uploaded_file.filename, "message": "success"}
