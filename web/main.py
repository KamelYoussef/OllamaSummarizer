# Import necessary libraries
import logging
import time
import streamlit as st
import requests
from htmlTemplates import css
from config_loader import load_config

# Read configuration from the config file
config = load_config()

# Constants
PDF_FILE_NAME = "document.pdf"
PDF_CONTENT_TYPE = "application/pdf"


def process_pdf(file):
    """
    Process the uploaded PDF file.

    Parameters:
    - file: File object representing the uploaded PDF.

    Returns:
    - None
    """
    try:
        # Send the PDF file to the FastAPI backend
        files = {"uploaded_file": (PDF_FILE_NAME, file, PDF_CONTENT_TYPE)}
        response = requests.post(config["domain"] + "/file/upload", files=files)

        # Check if the request was successful
        response.raise_for_status()

    except requests.exceptions.RequestException as e:
        logging.error(f"Error during processing: {e}")
        st.error("An error occurred during processing. Please try again.")


def display_summary(output_path):
    """
    Display the summary obtained from the FastAPI backend.

    Parameters:
    - output_path: Path to the directory where output files are saved.

    Returns:
    - None
    """
    with open(output_path + "Translation.txt", "r") as file:
        output = file.read()
        st.write(output)


def homepage(output_path="output/"):
    """
    Streamlit application.

    Displays a web-based interface allowing users to upload a PDF file,
    sends it to a FastAPI backend for processing, and displays the generated summary.

    Parameters:
    - output_path: Path to the directory where output files will be saved.

    Returns:
    - None
    """
    # Configure Streamlit page settings
    st.set_page_config(page_title="Résumé PDF", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)  # Apply custom CSS style

    # Display a header in the main section
    st.header("Résumé du PDF")

    # Sidebar section for file uploading
    with st.sidebar:
        pdf_doc = st.file_uploader("Selectionner votre fichier PDF ", type=["pdf"])

        # Button to initiate processing when clicked
        if st.button("Commencer"):
            with st.spinner("Traitement..."):
                # Measure the execution time
                start = time.time()

                # Process the uploaded PDF
                process_pdf(pdf_doc)

                end = time.time()
                logging.info(f"Execution time in seconds: {end - start}")

    # Display the Summary obtained from the FastAPI backend
    display_summary(output_path)

    # Clear the dashboard (commented out for now)
    # f = open('output/Summary.txt', 'r+')
    # f.truncate(0)


if __name__ == "__main__":
    # Run the Streamlit application
    homepage()
