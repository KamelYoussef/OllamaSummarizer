import time
import streamlit as st
from htmlTemplates import css

from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import Ollama
from Functions import *

llm = Ollama(
    model="zephyr", verbose=True, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
)


def main():
    st.set_page_config(page_title="Résumé PDF", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    st.header("Résumé du PDF")

    with st.sidebar:
        pdf_doc = st.file_uploader(
            "Selectionner votre fichier PDF ", type=["pdf"]
        )
        if st.button("Commencer"):
            with st.spinner("Traitement..."):
                start = time.time()

                # retrieve the pdf doc
                doc = text_upload(pdf_doc)

                # all process
                processing(doc, llm)

                end = time.time()
                print(f"\nExecution time in seconds: {end - start}")

    # Display the Summary
    output = open_file("Output/Translation")
    st.write(output)

    # f = open('Output/Summary.txt', 'r+')
    # f.truncate(0)


if __name__ == "__main__":
    main()
