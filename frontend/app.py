# Import necessary libraries
import logging
import time
import streamlit as st
import requests
from shared.config_loader import load_config
from wordcloud import WordCloud
from PIL import Image
from spacy.lang.fr.stop_words import STOP_WORDS as fr_stop
from io import BytesIO

logging.basicConfig(level=logging.INFO)


@st.cache_data(show_spinner=False)
def process_pdf(file):
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
        files = {"uploaded_file": (file.name, file.read(), "application/pdf")}
        response = requests.post(config["DOMAIN"] + "/file/upload", files=files)

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
    return file is not None and file.type == "application/pdf"


def generate_wordcloud(text):
    """
    Generate a word cloud from the given text.

    Parameters:
    - text: The input text for generating the word cloud.

    Returns:
    - wordcloud_image: Word cloud image.
    """

    stop_words = list(fr_stop)
    wordcloud = WordCloud(width=800, height=400, background_color="white", stopwords=stop_words).generate(text)

    # Convert to image
    img = wordcloud.to_image()

    # Convert image to bytes
    img_bytes = BytesIO()
    img.save(img_bytes, format="PNG")

    return img_bytes.getvalue()


def display_wordcloud(text):
    """
    Display the word cloud image.

    Parameters:
    - text: The input text for generating the word cloud.
    """
    st.subheader("Nuage de mots")
    st.image(generate_wordcloud(text))


def display_summary(text):
    """
    Display the summary.

    Parameters:
    - text: The input text for displaying the summary.
    """
    st.header("Résumé du PDF")
    st.write(text)


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
        st.markdown("---")
        if st.button("Vider le cache"):
            # Clear values from *all* all in-memory and on-disk data cache
            st.cache_data.clear()

    # Display the Summary obtained from the FastAPI backend
    if summary is not None:
        # Display the summary
        display_summary(summary)

        # Display the wordcloud
        display_wordcloud(summary)


if __name__ == "__main__":
    # Read configuration from the config file
    config = load_config()

    # Run the Streamlit app
    app()
