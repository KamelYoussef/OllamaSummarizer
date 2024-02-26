# Import necessary libraries
import re
import string
import unicodedata
import contractions
import fitz
import numpy as np
import pandas as pd
import yake
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.embeddings import OllamaEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sklearn.cluster import KMeans
from unidecode import unidecode
from frontend.shared.config_loader import load_config

config = load_config()


def extract_text(doc):
    """
    Extracts plain text from a PyMuPDF document.

    Parameters:
    - doc: PyMuPDF document object.

    Returns:
    - raw: Concatenated plain text extracted from the document.
    """
    try:
        output = [page.get_text("blocks") for page in doc]
        raw = "".join([unidecode(block[4]) for block in output if block[6] == 0])
        return raw
    except Exception as e:
        print(f"Error during text extraction: {e}")
        return ""


def extract_dict(doc):
    """
    Extracts text block information from a PyMuPDF document and organizes it into a dictionary.

    Parameters:
    - doc: PyMuPDF document object.

    Returns:
    - block_dict: Dictionary containing text block information for each page.
    """
    if not isinstance(doc, fitz.Document):
        raise ValueError("Invalid input: 'doc' must be a PyMuPDF document object.")

    block_dict = {}  # Dictionary to store text block information
    page_num = 1  # Initialize page number

    try:
        # Iterate through all pages in the document
        for page in doc:
            file_dict = page.get_text("dict")  # Get the page dictionary
            block = file_dict["blocks"]  # Get the block information
            block_dict[page_num] = block  # Store in the block dictionary
            page_num += 1  # Increase the page value by 1

        return block_dict
    except Exception as e:
        print(f"Error during text block extraction: {e}")
        return {}


def extract_spans(doc):
    """
    Extracts text spans from a document and returns a DataFrame with span information.

    Parameters:
    - doc: PyMuPDF document object.

    Returns:
    - span_df: DataFrame containing information about text spans (xmin, ymin, xmax, ymax, text, is_upper, is_bold,
               span_font, font_size).
    """

    if not isinstance(doc, fitz.Document):
        raise ValueError("Invalid input: 'doc' must be a PyMuPDF document object.")

    try:
        spans = pd.DataFrame(
            columns=["xmin", "ymin", "xmax", "ymax", "text", "is_upper", "is_bold", "span_font", "font_size"])
        rows = []

        # Iterate through pages and blocks using the extract_dict function
        for page_num, blocks in extract_dict(doc).items():
            for block in blocks:
                if block["type"] == 0:  # Check if it's a text block
                    for line in block["lines"]:
                        for span in line["spans"]:
                            xmin, ymin, xmax, ymax = list(span["bbox"])
                            font_size = span["size"]
                            text = unidecode(span["text"])
                            span_font = span["font"]
                            is_upper = False
                            is_bold = False

                            # Check if the span text is in uppercase
                            if re.sub("[\(\[].*?[\)\]]", "", text).isupper():
                                is_upper = True

                            # Check if the span font has bold attribute
                            if "bold" in span_font.lower():
                                is_bold = True

                            # Append span information to the rows list
                            if text.replace(" ", "") != "":
                                rows.append((
                                    xmin,
                                    ymin,
                                    xmax,
                                    ymax,
                                    text,
                                    is_upper,
                                    is_bold,
                                    span_font,
                                    font_size,
                                ))

        # Create a DataFrame from the collected span information
        span_df = pd.DataFrame(
            rows,
            columns=[
                "xmin",
                "ymin",
                "xmax",
                "ymax",
                "text",
                "is_upper",
                "is_bold",
                "span_font",
                "font_size",
            ],
        )

        return span_df
    except Exception as e:
        print(f"Error during span extraction: {e}")
        return pd.DataFrame(
            columns=["xmin", "ymin", "xmax", "ymax", "text", "is_upper", "is_bold", "span_font", "font_size"])


def score_span(doc):
    """
    Scores and tags text spans based on font size, boldness, and uppercase characteristics.

    Parameters:
    - doc: PyMuPDF document object.

    Returns:
    - span_df: DataFrame containing information about text spans along with an additional "tag" column.
    """
    span_scores = []  # List to store span scores
    span_num_occur = {}  # Dictionary to count occurrences of each span score
    special = "[(_:/,#%\=@)&]"

    # Iterate through each span in the DataFrame obtained from extract_spans function
    for index, span_row in extract_spans(doc).iterrows():
        score = round(span_row.font_size)
        text = span_row.text

        # Check for special characters in the text
        if not re.search(special, text):
            # Adjust score based on bold and uppercase attributes
            if span_row.is_bold:
                score += 1
            if span_row.is_upper:
                score += 1

        # Append the calculated score to the list
        span_scores.append(score)

    # Count occurrences of each span score
    values, counts = np.unique(span_scores, return_counts=True)
    style_dict = {}
    for value, count in zip(values, counts):
        style_dict[value] = count

    # Sort style_dict based on counts (not inplace)
    sorted(style_dict.items(), key=lambda x: x[1])

    # Determine the primary font size
    p_size = max(style_dict, key=style_dict.get)

    idx = 0
    tag = {}

    # Assign tags based on font size
    for size in sorted(values, reverse=True):
        idx += 1
        if size == p_size:
            idx = 0
            tag[size] = "p"
        if size > p_size:
            tag[size] = f"h{idx}"
        if size < p_size:
            tag[size] = f"s{idx}"

    # Assign tags to each span based on the calculated scores
    span_tags = [tag[score] for score in span_scores]

    # Create a DataFrame with additional "tag" column
    span_df = extract_spans(doc)
    span_df["tag"] = span_tags

    return span_df


def correct_end_line(line):
    """
    Checks if a line of text ends with a hyphen, indicating a continuation to the next line.

    Parameters:
    - line: The input line of text.

    Returns:
    - True if the line ends with a hyphen, indicating a continuation.
    - False otherwise.
    """
    "".join(line.rstrip().lstrip())  # Remove leading and trailing whitespaces

    # Check if the line ends with a hyphen
    if line[-1] == "-":
        return True
    else:
        return False


def category_text(doc):
    """
    Extracts and categorizes text content based on heading tags (h) and paragraph tags (p).

    Parameters:
    - doc: PyMuPDF document object.

    Returns:
    - text_df: DataFrame containing extracted headings and corresponding content.
    """
    headings_list = []  # List to store extracted headings
    text_list = []  # List to store corresponding content
    tmp = []  # Temporary list for accumulating content
    heading = ""

    # Obtain a DataFrame with scored and tagged spans using score_span function
    span_df = score_span(doc)

    # Filter spans to include only headings (h) and paragraphs (p)
    span_df = span_df.loc[span_df.tag.str.contains("h|p")]

    # Iterate through each span in the DataFrame
    for index, span_row in span_df.iterrows():
        text = span_row.text
        tag = span_row.tag

        # Check if the span represents a heading (h)
        if "h" in tag:
            headings_list.append(text)
            text_list.append("".join(tmp))
            tmp = []
            heading = text
        else:
            # Check for hyphen indicating line continuation
            if correct_end_line(text):
                tmp.append(text[:-1])
            else:
                tmp.append(text + " ")

    # Append the last accumulated content
    text_list.append("".join(tmp))
    text_list = text_list[1:]  # Exclude the first empty element
    text_df = pd.DataFrame(
        zip(headings_list, text_list), columns=["heading", "content"]
    )

    return text_df


def merge_text(doc):
    """
    Merges and cleans text content extracted from a document.

    Parameters:
    - doc: PyMuPDF document object.

    Returns:
    - Merged and cleaned text content as a string.
    """
    s = ""  # String to accumulate merged text

    # Iterate through each row in the DataFrame obtained from category_text function
    for index, row in category_text(doc).iterrows():
        s += "".join((row["heading"], "\n", row["content"]))

    # Clean and normalize the merged text
    return clean_text(" ".join(s.split()))


def clean_text(text, cleaning_functions=None):
    """
    Perform text cleaning operations on the input text.

    Args:
        text (str): Input text to be cleaned.
        cleaning_functions (list, optional): List of cleaning functions to apply in order.

    Returns:
        str: Cleaned text.
    """
    if cleaning_functions is None:
        cleaning_functions = [
            to_lowercase,
            standardize_accented_chars,
            remove_url,
            expand_contractions,
            remove_mentions_and_tags,
            remove_special_characters,
            remove_spaces,
        ]

    cleaned_text = text
    for cleaning_function in cleaning_functions:
        try:
            cleaned_text = cleaning_function(cleaned_text)
        except Exception as e:
            print(f"Error in {cleaning_function.__name__}: {e}")

    return cleaned_text


def to_lowercase(text):
    """
    Converts a given text to lowercase.

    Parameters:
    - text: The input text to be converted.

    Returns:
    - Lowercase version of the input text.
    """
    return text.lower()


def standardize_accented_chars(text):
    """
    Standardizes accented characters in a text by converting them to their closest ASCII representation.

    Parameters:
    - text: The input text with accented characters.

    Returns:
    - Standardized text with accented characters converted to ASCII representation.
    """
    return (
        unicodedata.normalize("NFKD", text)
        .encode("ascii", "ignore")
        .decode("utf-8", "ignore")
    )


def remove_url(text):
    """
    Removes URLs from a given text.

    Parameters:
    - text: The input text containing URLs.

    Returns:
    - Text with URLs removed.
    """
    return re.sub(r"https?:\S*", "", text)


def expand_contractions(text):
    """
    Expands contractions in a given text.

    Parameters:
    - text: The input text with contractions.

    Returns:
    - Text with contractions expanded.
    """
    expanded_words = []
    for word in text.split():
        expanded_words.append(contractions.fix(word))
    return " ".join(expanded_words)


def remove_mentions_and_tags(text):
    """
    Removes mentions (@) and tags (#) from a given text.

    Parameters:
    - text: The input text containing mentions and tags.

    Returns:
    - Text with mentions and tags removed.
    """
    return re.sub(r"[@#]\S*", "", text)


def remove_special_characters(text):
    """
    Removes special characters from a given text, keeping only alphanumeric characters and selected symbols.

    Parameters:
    - text: The input text with special characters.

    Returns:
    - Text with special characters removed.
    """
    pat = r"[^a-zA-z0-9.,!?/:;\"\'\s]"
    return re.sub(pat, "", text)


def remove_spaces(text):
    """
    Removes extra spaces from a given text.

    Parameters:
    - text: The input text with extra spaces.

    Returns:
    - Text with extra spaces removed.
    """
    return re.sub(" +", " ", text)


def remove_punctuation(text):
    """
    Removes punctuation from a given text.

    Parameters:
    - text: The input text with punctuation.

    Returns:
    - Text with punctuation removed.
    """
    return "".join([c for c in text if c not in string.punctuation])


def extract_keywords(text):
    """
    Extracts keywords from a given text using the YAKE (Yet Another Keyword Extractor) algorithm.

    Parameters:
    - text: The input text from which keywords will be extracted.

    Returns:
    - List of extracted keywords.
    """
    kw_extractor = yake.KeywordExtractor(top=20, stopwords=None)  # Initialize YAKE extractor
    keywords = kw_extractor.extract_keywords(text)  # Extract keywords using YAKE
    return [kw for kw, v in keywords]  # Return a list of extracted keywords


def extract_info(input_file: str):
    """
    Extracts file info
    """
    # Open the PDF
    pdfDoc = fitz.open(input_file)
    output = {
        "File": input_file,
        "Encrypted": ("True" if pdfDoc.isEncrypted else "False"),
    }
    # If PDF is encrypted the file metadata cannot be extracted
    if not pdfDoc.isEncrypted:
        for key, value in pdfDoc.metadata.items():
            output[key] = value

    # To Display File Info
    print("## File Information ##################################################")
    print("\n".join("{}:{}".format(i, j) for i, j in output.items()))
    print("######################################################################")

    return True, output


def clustering(vectors, num_clusters=10):
    """
    Performs K-means clustering on a set of vectors and returns the indices of representative points.

    Parameters:
    - vectors: List of vectors representing embeddings.

    Returns:
    - selected_indices: Indices of representative points after clustering.
    """
    if len(vectors) >= num_clusters:
        kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(vectors)
        labels = kmeans.labels_
        idx = [labels.tolist().index(i) for i in range(num_clusters)]  # need improvement
    else:
        idx = list(range(len(vectors)))

    selected_indices = sorted(idx)
    return selected_indices


def get_num_tokens(llm, text):
    """
    Returns the number of tokens in a given text using a language model.

    Parameters:
    - llm: Language model object with a 'get_num_tokens' method.
    - text: The input text for token count.

    Returns:
    - The number of tokens.
    """
    return llm.get_num_tokens(text)


def chunking(text):
    """
    Splits a given text into chunks using RecursiveCharacterTextSplitter.

    Parameters:
    - text: The input text to be chunked.

    Returns:
    - docs: List of documents obtained after splitting the input text.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config["CHUNK_SIZE"],
        chunk_overlap=config["CHUNK_OVERLAP"]
    )

    # Create documents by splitting the input text
    docs = text_splitter.create_documents([text])

    # Print the number of documents created
    print(f"Now our text is split up into {len(docs)} documents")

    return docs


def embedding(docs):
    """
    Generates document embeddings using Ollama embeddings.

    Parameters:
    - docs: List of documents with 'page_content' attribute containing text for embedding.

    Returns:
    - vectors: Document embeddings generated using Ollama embeddings.
    """
    embeddings = OllamaEmbeddings(base_url=config["OLLAMA_URL"], model=config["MODEL"])
    vectors = embeddings.embed_documents([x.page_content for x in docs])
    return vectors


def chunks_summaries(docs, selected_indices, llm):
    """
    Generates summaries for selected document chunks using a language model.

    Parameters:
    - docs: List of documents containing chunks for summarization.
    - selected_indices: List of indices indicating the selected document chunks.
    - llm: Language model with summarization capabilities.

    Returns:
    - summaries: Summaries for the selected document chunks.
    """
    map_prompt = """
                    Give a title and a summary in one paragraph for the article : ```{text}```
                    """
    map_prompt_template = PromptTemplate(
        template=map_prompt, input_variables=["text"]
    )
    map_chain = load_summarize_chain(
        llm=llm, chain_type="stuff", prompt=map_prompt_template
    )

    selected_docs = [docs[doc] for doc in selected_indices]

    # Make an empty list to hold your summaries
    summary_list = []

    # Loop through a range of the length of your selected docs
    for i, doc in enumerate(selected_docs):
        # Go get a summary of the chunk
        chunk_summary = map_chain.run([doc])
        chunk_summary = re.sub("Title: ", "", chunk_summary)
        chunk_summary = re.sub("Summary: ", "\n", chunk_summary)

        # Append that summary to your list
        summary_list.append(chunk_summary)

    summaries = "\n\n\n".join(summary_list)
    return summaries


def convert_to_document(text):
    """
    Converts a given text to a document object.

    Parameters:
    - text: The input text to be converted to a document.

    Returns:
    - doc: Document object with 'page_content' attribute set to the input text.
    """
    doc = Document(page_content=text)
    return doc


def combine_summary(summaries, llm):
    """
    Combines multiple summaries into a verbose summary using a language model.

    Parameters:
    - summaries: Summaries to be combined.
    - llm: Language model with summarization capabilities.

    Returns:
    - output: Verbose summary combining the input summaries.
    """
    combine_prompt = """
                        Give a summary in three sentences of the article : ```{text}```
                        """
    combine_prompt_template = PromptTemplate(
        template=combine_prompt, input_variables=["text"]
    )

    reduce_chain = load_summarize_chain(
        llm=llm,
        chain_type="stuff",
        prompt=combine_prompt_template,
    )
    output = reduce_chain.run([summaries])
    return output


def translation_to_french(text, llm):
    """
    Translates a given text to French using a language model.

    Parameters:
    - text: The input text to be translated.
    - llm: Language model object with translation capabilities.

    Returns:
    - translation: Translated text in French.
    """
    translation = llm(f"Give a translation in French of the article : ```{text}```")
    return translation


def bullet_points_summaries(text, llm):
    """
    Translates a given text to French using a language model.

    Parameters:
    - text: The input text to be summarized.
    - llm: Language model object with summarization capabilities.

    Returns:
    - translation: summary in bullet points.
    """
    bullets = llm(f"Give very short bullet points that summarizes the article : ```{text}```")
    return bullets