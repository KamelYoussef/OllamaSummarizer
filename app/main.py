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
from backend.PDFProcessor import PDFProcessor
import logging
from frontend.shared.config_loader import load_config

config = load_config()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
    model=config["MODEL"],  # Specify the language model to use
    verbose=True,
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
)


# Function to handle conversation processing using llm
def conversation(human_input, llm=llm):
    """
    Process a human input in a conversation context using the language model.

    Parameters:
    - human_input (str): The input provided by the user.

    Returns:
    - str: The model's response to the input.
    """
    prompt = human_input

    try:
        output = llm(prompt)
        return output
    except Exception as e:
        logger.error(f"Error during conversation processing: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


# Define the "/conversation" endpoint to handle text input
@app.post("/conversation")
async def input(input: Input):
    """
    Endpoint to handle text input for a conversation.

    Parameters:
    - input (Input): An Input object containing the human input.

    Returns:
    - Output: An Output object containing the model's response.
    """
    # Process the input using the conversation function
    output = Output(output=conversation(input.human_input))
    return output


# Define the "/file/upload" endpoint to handle PDF file uploads
@app.post("/file/upload", response_model=dict)
async def upload_file(uploaded_file: UploadFile = File(...), llm=llm):
    """
    Endpoint to upload and process a PDF file.

    Parameters:
    - uploaded_file (UploadFile): The uploaded PDF file.

    Returns:
    - dict: A dictionary with information about the processing status.

    Raises:
    - HTTPException: If the uploaded file is not a PDF (status_code=400).
    - Exception: If an error occurs during file upload and processing.

    Example:
    - {"message": "Processing successful", "result": "output_summary_text"}
    - {"error": "Error during file upload and processing: error_details"}
    """
    try:
        # Check if the uploaded file is a PDF
        if uploaded_file.content_type != "application/pdf":
            raise HTTPException(status_code=400, detail="Invalid PDF file")

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

    except HTTPException as http_exception:
        raise http_exception

    except Exception as e:
        logger.error(f"Error during file upload and processing: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
