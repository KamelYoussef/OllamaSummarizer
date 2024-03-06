class PDFAnnotator:
    def __init__(self, pdf_doc):
        self.document = pdf_doc

    def search_and_highlight(self, search_text, highlight_color=(0.5, 1, 0)):
        for page_num in range(len(self.document)):
            page = self.document.load_page(page_num)
            text_instances = page.search_for(search_text)

            for inst in text_instances:
                # Create highlight annotation
                highlight = page.add_highlight_annot(inst)
                highlight.set_colors(stroke=highlight_color)

        # Save changes
        self.document.save("anno.pdf")