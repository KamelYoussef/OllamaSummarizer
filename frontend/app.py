# Import necessary libraries
import logging
import time
import streamlit as st
import requests
from shared.config_loader import load_config


# Constants
PDF_FILE_NAME = "document.pdf"
PDF_CONTENT_TYPE = "application/pdf"


def process_pdf(file):
    """
    Process the uploaded PDF file.

    Parameters:
    - file: File object representing the uploaded PDF.

    Returns:
    - result: The result or summary of processing, or None in case of an error.
    """
    try:
        # Send the PDF file to the FastAPI backend
        files = {"uploaded_file": (file.name, file.read(), PDF_CONTENT_TYPE)}
        response = requests.post(config["domain"] + "/file/upload", files=files)

        # Check if the request was successful
        response.raise_for_status()

        # Parse the JSON response
        result = response.json().get("result")

        return result

    except requests.exceptions.RequestException as e:
        logging.error(f"Error during processing: {e}")
        st.error("An error occurred during processing. Please try again.")
        return None


def app():
    """
    Streamlit application.

    Displays a frontend-based interface allowing users to upload a PDF file,
    sends it to a FastAPI backend for processing, and displays the generated summary.

    Returns:
    - None
    """
    # Configure Streamlit page settings
    st.set_page_config(page_title="Résumé PDF", page_icon=":books:")

    # Load external CSS file
    css = open("frontend/pages/styles.css").read()

    # Apply the styles using st.markdown
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

    # Display a header in the main section
    st.header("Résumé du PDF")

    # Initialize summary variable
    summary = None

    # Sidebar section for file uploading
    with st.sidebar:
        pdf_doc = st.file_uploader("Selectionner votre fichier PDF ", type=["pdf"])

        # Button to initiate processing when clicked
        if st.button("Commencer"):
            with st.spinner("Traitement..."):
                # Measure the execution time
                start = time.time()

                # Process the uploaded PDF
                summary = process_pdf(pdf_doc)

                end = time.time()
                logging.info(f"Execution time in seconds: {end - start}")

    # Display the Summary obtained from the FastAPI backend
    if summary is not None:
        st.write(summary)

if __name__ == "__main__":
    # Read configuration from the config file
    config = load_config()

    # Run the Streamlit app
    app()