import logging
from backend.functions import *


class PDFProcessor:
    def __init__(self, pdf_doc, llm):
        self.pdf_doc = pdf_doc
        self.llm = llm

    def _extract_and_clean_text(self):
        """
        Extracts and cleans text from the PDF document.

        Returns:
            str: Cleaned text.
        """
        text = merge_text(self.pdf_doc)
        logging.info(f"Number of tokens in the extracted text: {get_num_tokens(self.llm, text)}")
        return text

    def _process_text(self, text):
        """
        Processes the input text.

        Args:
            text (str): Input text.

        Returns:
            list: Processed summaries.
        """
        self.docs = chunking(text)
        vectors = embedding(self.docs)
        self.selected_indices = clustering(vectors)
        summaries = chunks_summaries(self.docs, self.selected_indices, self.llm)
        return summaries

    def process(self):
        """
        Process the PDF document.

        Returns:
            str: Translated output.
        """
        try:
            # Extract and clean text
            text = self._extract_and_clean_text()

            # Process text
            summaries = self._process_text(text)

            # Convert summaries to Document
            summaries_docs = convert_to_document(summaries)

            # Summary of the summaries
            output = combine_summary(summaries_docs, self.llm)

            # Translation
            # translation = translation_to_french(output, self.llm)

            logging.info("Processing completed successfully.")

            selected_docs = [self.docs[doc] for doc in self.selected_indices]

            return summaries, output, selected_docs

        except Exception as e:
            logging.error(f"Error during processing: {e}")
            # Add additional error handling or raise the exception based on your requirements
