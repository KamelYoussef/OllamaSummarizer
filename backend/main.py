"""A FastAPI frontend application with two endpoints. The application receives input in the form of a human-generated text
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
from backend.PDFProcessor import PDFProcessor


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
    """
    Endpoint to upload and process a PDF file.

    Parameters:
    - uploaded_file: UploadFile object representing the uploaded file.

    Returns:
    - dict: A dictionary with a message or an error in case of failure.
    """
    try:
        # Check if the uploaded file is a PDF
        if uploaded_file.content_type != "application/pdf":
            raise HTTPException(status_code=400, detail="File must be a PDF")

        # Read the content of the uploaded PDF file as bytes
        pdf_bytes = await uploaded_file.read()

        # Open the PDF file using PyMuPDF
        pdf_document = fitz.open("pdf", pdf_bytes)

        # Instantiate PDFProcessor
        pdf_processor = PDFProcessor(pdf_document, llm)

        # Call the process method to perform PDF processing
        output = pdf_processor.process()

        # Close the PDF document
        pdf_document.close()

        # Return a response
        return {"message": "Processing successful", "result": output}

    except Exception as e:
        return {"error": f"Error during file upload and processing: {e}"}
