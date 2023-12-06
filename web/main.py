"""Using Streamlit framework to create a simple web-based interface. The application allows users to upload a PDF
file, which is then sent to a FastAPI backend for processing. The backend extracts information from the PDF and
generates a summary.
"""

# Import necessary libraries
import time
import streamlit as st
from htmlTemplates import css
import requests # Library for making HTTP requests

# FastAPI backend endpoint for processing PDF files
domain = "http://localhost:8000"
url = f"{domain}/file/upload"


def homepage():
    """
    Streamlit application.

    Displays a web-based interface allowing users to upload a PDF file,
    sends it to a FastAPI backend for processing, and displays the generated summary.
    """
    # Configure Streamlit page settings
    st.set_page_config(page_title="Résumé PDF", page_icon=":books:")
    st.write(css, unsafe_allow_html=True) # Apply custom CSS style

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

                # Send the PDF file to the FastAPI backend
                files = {"uploaded_file": ("document.pdf", pdf_doc, "application/pdf")}
                response = requests.post(url, files=files)

                end = time.time()
                print(f"\nExecution time in seconds: {end - start}")

    # Display the Summary obtained from the FastAPI backend
    with open("Output/Translation.txt", "r") as file:
        output = file.read()
        st.write(output)

    # Clear the dashboard (commented out for now)
    # f = open('Output/Summary.txt', 'r+')
    # f.truncate(0)


if __name__ == "__main__":
    # Run the Streamlit application
    homepage()
