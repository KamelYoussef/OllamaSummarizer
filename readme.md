# PDF Summarizer

PDF Summarizer is a web application with a FastAPI backend and a Streamlit frontend that processes and summarizes PDF documents. The backend utilizes the Ollama language model through the LangChain library to generate summaries based on human-generated text or uploaded PDF files.

## Features

- **Text Input Processing**: Send human-generated text to the backend for summarization.
- **PDF Upload and Summarization**: Upload PDF documents for automatic summarization using the Ollama language model.
- **Cross-Origin Resource Sharing (CORS)**: Middleware configured for handling requests from different origins.
- **Conversation Processing**: Utilizes the Ollama language model for processing text input in a conversation context.
- **Error Handling**: Proper handling of errors during text and PDF processing.

## Getting Started

### Prerequisites

Ensure you have Python and pip installed on your machine. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Running the Application

### Backend (FastAPI):

Run the FastAPI backend using the following command:

```bash
uvicorn app/main:app --reload
```
## Frontend (Streamlit):

Run the Streamlit application using the following command:

```bash
streamlit run frontend/app.py
```

## API Endpoints

### `/conversation`

- **Method**: POST
- **Parameters**:
  - `input` (Input): An Input object containing the human input.
- **Returns**:
  - `output` (Output): An Output object containing the model's response.

### `/file/upload`

- **Method**: POST
- **Parameters**:
  - `uploaded_file` (UploadFile): The uploaded PDF file.
- **Returns**:
  - `result` (dict): A dictionary with information about the processing status.

    ## Contributing

Contributions are welcome! If you would like to contribute to the project, please follow these steps:

1. Fork the repository.
2. Create a new branch for your changes.
3. Make your changes and commit them.
4. Push your changes to your fork.
5. Submit a pull request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [FastAPI](https://fastapi.tiangolo.com/)
- [Streamlit](https://streamlit.io/)
- [LangChain](https://github.com/Menelause/LangChain)
- [Ollama](https://github.com/KamelYoussef/OllamaSummarizer)
