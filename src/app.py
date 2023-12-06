from fastapi import FastAPI, File, UploadFile, HTTPException
import fitz
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import Ollama
import os
from src.Functions import processing


class Input(BaseModel):
    human_input: str


class Output(BaseModel):
    output: str


app = FastAPI()

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
    model="zephyr", verbose=True, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
)


def conversation(human_input):
    prompt = human_input

    output = llm(prompt)
    return output


@app.post("/conversation")
async def input(input: Input):
    output = Output(output=conversation(input.human_input))
    return output


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

    text = fitz.open(save_path)

    # Main function
    processing(text, llm)

    return {"filename": uploaded_file.filename, "message": "success"}





