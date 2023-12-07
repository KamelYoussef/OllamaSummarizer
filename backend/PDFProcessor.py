import os
import logging
from backend.functions import *

class PDFProcessor:
    def __init__(self, pdf_doc, llm, output_path="output/"):
        self.pdf_doc = pdf_doc
        self.llm = llm
        self.output_path = output_path

    def _extract_and_clean_text(self):
        text = merge_text(self.pdf_doc)
        logging.info(f"Number of tokens in the extracted text: {get_num_tokens(self.llm, text)}")
        return text

    def _process_text(self, text):
        docs = chunking(text)
        vectors = embedding(docs)
        selected_indices = clustering(vectors)
        summaries = chunks_summaries(docs, selected_indices, self.llm)
        return summaries

    def _save_and_log(self, file_name, content):
        file_path = os.path.join(self.output_path, file_name)
        save_file(file_path, content)
        logging.info(f"Number of tokens in {file_name}: {get_num_tokens(self.llm, content)}")

    def process(self):
        try:
            # Extract and clean text
            text = self._extract_and_clean_text()

            # Process text
            summaries = self._process_text(text)

            # Save Summaries
            self._save_and_log("Summaries", summaries)

            # Convert summaries to Document
            summaries_docs = convert_to_document(summaries)

            # Summary of the summaries
            output = combine_summary(summaries_docs, self.llm)

            # Save Summary
            self._save_and_log("Summary", output)

            # Translation
            translation = translation_to_french(output, self.llm)

            # Save Translation
            self._save_and_log("Translation", translation)

            logging.info("Processing completed successfully.")

        except Exception as e:
            logging.error(f"Error during processing: {e}")
            # Add additional error handling or raise the exception based on your requirements
