class PDFAnnotator:
    """A class for annotating PDF documents."""

    def __init__(self, pdf_doc):
        """
        Initialize the PDFAnnotator object.

        Args:
            pdf_doc (PDFDocument): The PDF document to be annotated.
        """
        self.document = pdf_doc

    def search_and_highlight(self, search_text, highlight_color=(0.5, 1, 0)):
        """
        Search for a text string in the PDF document and highlight occurrences.

        Args:
            search_text (str): The text to search for in the document.
            highlight_color (tuple, optional): RGB tuple representing the highlight color. Default is (0.5, 1, 0).

        Returns:
            PDFDocument: The annotated PDF document.
        """
        for page_num in range(len(self.document)):
            try:
                page = self.document.load_page(page_num)
                text_instances = page.search_for(search_text)

                for inst in text_instances:
                    # Create highlight annotation
                    highlight = page.add_highlight_annot(inst)
                    highlight.set_colors(stroke=highlight_color)
            except Exception as e:
                print(f"Error processing page {page_num}: {e}")

        # Save changes
        self.document.save("frontend/anno.pdf")