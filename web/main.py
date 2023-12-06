import time
import streamlit as st
from htmlTemplates import css
import requests

domain = "http://localhost:8000"
url = f"{domain}/file/upload"


def homepage():
    st.set_page_config(page_title="Résumé PDF", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    st.header("Résumé du PDF")

    with st.sidebar:
        pdf_doc = st.file_uploader("Selectionner votre fichier PDF ", type=["pdf"])
        if st.button("Commencer"):
            with st.spinner("Traitement..."):

                start = time.time()

                # Send the PDF file using FastAPI
                files = {"uploaded_file": ("document.pdf", pdf_doc, "application/pdf")}
                response = requests.post(url, files=files)

                end = time.time()
                print(f"\nExecution time in seconds: {end - start}")

    # Display the Summary
    with open("Output/Translation.txt", "r") as file:
        output = file.read()
        st.write(output)

    # Clear the dashboard
    # f = open('Output/Summary.txt', 'r+')
    # f.truncate(0)


if __name__ == "__main__":
    homepage()
