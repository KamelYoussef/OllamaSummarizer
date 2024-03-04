# Import necessary libraries
import logging
import time
import streamlit as st
import requests
from shared.config_loader import load_config
import base64
from streamlit_pdf_viewer import pdf_viewer
import fitz

logging.basicConfig(level=logging.INFO)


@st.cache_data(show_spinner=False)
def process_pdf(file, start=1, end=1):
    """
    Process the uploaded PDF file.

    Parameters:
    - file: File object representing the uploaded PDF.

    Returns:
    - result: The result or summary of processing, or None in case of an error.
    """
    try:
        if not is_valid_pdf(file):
            st.error("Veuillez sélectionner un fichier PDF valide.")
            return None

        # Send the PDF file to the FastAPI backend
        files = {
            "uploaded_file": (file.name, file.read(), "application/pdf"),
        }
        params = {
            "start": start,
            "end": end
        }

        response = requests.post(config["DOMAIN"] + "/file/upload", files=files, params=params)

        # Check if the request was successful
        response.raise_for_status()

        # Parse the JSON response
        result = response.json().get("result")
        return result

    except requests.exceptions.RequestException as e:
        logging.error(f"Error during processing: {e}")
        st.error("Une erreur s'est produite pendant le traitement. Veuillez réessayer.")
        return None

    except Exception as e:
        logging.exception(f"Unexpected error: {e}")
        st.error("Une erreur inattendue s'est produite. Veuillez contacter l'administrateur.")
        return None


def is_valid_pdf(file):
    """
    Check if the uploaded file is a valid PDF.
    """
    return file is not None and file.type == "application/pdf" and file.size > 0


def display_summary(text):
    """
    Display the summary.

    Parameters:
    - text: The input text for displaying the summary.
    """
    words = len(text.split())
    st.text_area(label=f"Résumé du PDF: ({words} mots)", value=text, height=485)
    st.columns(6)[5].button("Copier")


def display_pdf(pdf_doc):
    pdf_viewer(input=pdf_doc.getvalue(), height=590)


def pdf_num_pages(doc):
    with fitz.open("pdf", doc.read()) as file:
        num = file.page_count
    doc.seek(0)  # repopulate the UploadedFile stream
    return num


def app():
    """
    Streamlit application.

    Displays a frontend-based interface allowing users to upload a PDF file,
    sends it to a FastAPI backend for processing, and displays the generated summary.

    Returns:
    - None
    """

    # Configure Streamlit page settings
    st.set_page_config(page_title="Résumé PDF", page_icon=":books:", layout="wide")

    # Load external CSS file
    css = open("frontend/pages/styles.css").read()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

    # Initialize summary variable
    summary = None

    # Split the web page into two columns to display the PDF and its summary
    col1, col2 = st.columns(spec=[1.33, 1], gap="small")

    # Sidebar section for file uploading
    with st.sidebar:
        st.sidebar.title("Résumé de documents")
        # Upload the PDF and display it
        pdf_doc = st.file_uploader("Selectionner votre fichier PDF ", type=["pdf"], accept_multiple_files=False,
                                   key="pdf_uploader", help="Taille maximale de fichier: 200Mo")
        if pdf_doc:
            with col1:
                display_pdf(pdf_doc)

            # choose type of summary
            types = st.radio("Type du résumé", ["Résumé long", "Résumé court"])

            # select pages from pdf
            num_pages = pdf_num_pages(pdf_doc)
            pages = (1, 1)
            if num_pages > 1:
                pages = st.slider('Selection des pages ', 1, num_pages, (1, num_pages))

            # Button to initiate processing when clicked
            if st.button("Commencer"):
                with st.spinner("Traitement..."):
                    # Measure the execution time
                    start = time.time()

                    # Process the uploaded PDF
                    summary = process_pdf(pdf_doc, pages[0], pages[1])

                    end = time.time()
                    logging.info(f"Execution time in seconds: {end - start}")

        st.markdown("---")
        if st.button("Refaire"):
            # Clear values from *all* all in-memory and on-disk data cache
            st.cache_data.clear()

    # Display the Summary obtained from the FastAPI backend
    if summary is not None:
        with col2:
            if types == "Résumé long":
                display_summary(summary[0])
            elif types == "Résumé court":
                display_summary(summary[1])


if __name__ == "__main__":
    # Read configuration from the config file
    config = load_config()

    # Run the Streamlit app
    app()
