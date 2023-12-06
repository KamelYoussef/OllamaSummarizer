import os
import re
import string
import unicodedata
from tempfile import NamedTemporaryFile
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


def extract_text(doc):
    output = []
    raw = ""
    for page in doc:
        output += page.get_text("blocks")
    for block in output:
        if block[6] == 0:  # We only take the text
            plain_text = unidecode(block[4])  # Encode in ASCII
            raw += plain_text
    return raw


def extract_dict(doc):
    block_dict = {}
    page_num = 1
    for page in doc:  # Iterate all pages in the document
        file_dict = page.get_text("dict")  # Get the page dictionary
        block = file_dict["blocks"]  # Get the block information
        block_dict[page_num] = block  # Store in block dictionary
        page_num += 1  # Increase the page value by 1
    return block_dict


def extract_spans(doc):
    spans = pd.DataFrame(columns=["xmin", "ymin", "xmax", "ymax", "text", "tag"])
    rows = []
    for page_num, blocks in extract_dict(doc).items():
        for block in blocks:
            if block["type"] == 0:
                for line in block["lines"]:
                    for span in line["spans"]:
                        xmin, ymin, xmax, ymax = list(span["bbox"])
                        font_size = span["size"]
                        text = unidecode(span["text"])
                        span_font = span["font"]
                        is_upper = False
                        is_bold = False
                        if "bold" in span_font.lower():
                            is_bold = True
                        if re.sub("[\(\[].*?[\)\]]", "", text).isupper():
                            is_upper = True
                        if text.replace(" ", "") != "":
                            rows.append(
                                (
                                    xmin,
                                    ymin,
                                    xmax,
                                    ymax,
                                    text,
                                    is_upper,
                                    is_bold,
                                    span_font,
                                    font_size,
                                )
                            )
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


def score_span(doc):
    span_scores = []
    span_num_occur = {}
    special = "[(_:/,#%\=@)&]"
    for index, span_row in extract_spans(doc).iterrows():
        score = round(span_row.font_size)
        text = span_row.text
        if not re.search(special, text):
            if span_row.is_bold:
                score += 1
            if span_row.is_upper:
                score += 1
        span_scores.append(score)
    values, counts = np.unique(span_scores, return_counts=True)
    style_dict = {}
    for value, count in zip(values, counts):
        style_dict[value] = count
    sorted(style_dict.items(), key=lambda x: x[1])
    p_size = max(style_dict, key=style_dict.get)
    idx = 0
    tag = {}
    for size in sorted(values, reverse=True):
        idx += 1
        if size == p_size:
            idx = 0
            tag[size] = "p"
        if size > p_size:
            tag[size] = "h{0}".format(idx)
        if size < p_size:
            tag[size] = "s{0}".format(idx)
    span_tags = [tag[score] for score in span_scores]
    span_df = extract_spans(doc)
    span_df["tag"] = span_tags
    return span_df


def correct_end_line(line):
    "".join(line.rstrip().lstrip())
    if line[-1] == "-":
        return True
    else:
        return False


def category_text(doc):
    headings_list = []
    text_list = []
    tmp = []
    heading = ""
    span_df = score_span(doc)
    span_df = span_df.loc[span_df.tag.str.contains("h|p")]

    for index, span_row in span_df.iterrows():
        text = span_row.text
        tag = span_row.tag
        if "h" in tag:
            headings_list.append(text)
            text_list.append("".join(tmp))
            tmp = []
            heading = text
        else:
            if correct_end_line(text):
                tmp.append(text[:-1])
            else:
                tmp.append(text + " ")

    text_list.append("".join(tmp))
    text_list = text_list[1:]
    text_df = pd.DataFrame(
        zip(headings_list, text_list), columns=["heading", "content"]
    )
    return text_df


def merge_text(doc):
    s = ""
    for index, row in category_text(doc).iterrows():
        s += "".join((row["heading"], "\n", row["content"]))
    return clean_text(" ".join(s.split()))
    # return s


def clean_text(text):
    out = to_lowercase(text)
    out = standardize_accented_chars(out)
    out = remove_url(out)
    out = expand_contractions(out)
    out = remove_mentions_and_tags(out)
    out = remove_special_characters(out)
    out = remove_spaces(out)
    return out


def to_lowercase(text):
    return text.lower()


def standardize_accented_chars(text):
    return (
        unicodedata.normalize("NFKD", text)
        .encode("ascii", "ignore")
        .decode("utf-8", "ignore")
    )


def remove_url(text):
    return re.sub(r"https?:\S*", "", text)


def expand_contractions(text):
    expanded_words = []
    for word in text.split():
        expanded_words.append(contractions.fix(word))
    return " ".join(expanded_words)


def remove_mentions_and_tags(text):
    text = re.sub(r"@\S*", "", text)
    return re.sub(r"#\S*", "", text)


def remove_special_characters(text):
    # define the pattern to keep
    pat = r"[^a-zA-z0-9.,!?/:;\"\'\s]"
    return re.sub(pat, "", text)


def remove_spaces(text):
    return re.sub(" +", " ", text)


def remove_punctuation(text):
    return "".join([c for c in text if c not in string.punctuation])


def remove_stopwords(text, nlp):
    filtered_sentence = []
    doc = nlp(text)
    for token in doc:
        if token.is_stop == False:
            filtered_sentence.append(token.text)
    return " ".join(filtered_sentence)


def lemmatize(text, nlp):
    doc = nlp(text)
    lemmatized_text = []
    for token in doc:
        lemmatized_text.append(token.lemma_)
    return " ".join(lemmatized_text)


def extract_keywords(text):
    kw_extractor = yake.KeywordExtractor(top=20, stopwords=None)
    keywords = kw_extractor.extract_keywords(text)
    return [kw for kw, v in keywords]


def save_file(name, doc):
    with open(name + ".txt", "w") as file:
        file.write(doc)


def open_file(name):
    with open(name + ".txt", "r") as file:
        return file.read()


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


def is_valid_path(path):
    """
    Validates the path inputted and checks whether it is a file path or a folder path
    """
    if not path:
        raise ValueError(f"Invalid Path")
    if os.path.isfile(path):
        return path
    elif os.path.isdir(path):
        return path
    else:
        raise ValueError(f"Invalid Path {path}")


def clustering(vectors):
    if len(vectors) > 10:
        # Choose the number of clusters, this can be adjusted based on the book's content.
        num_clusters = 10
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(vectors)
        labels = kmeans.labels_
        # Find the closest embeddings to the centroids
        # Create an empty list that will hold your closest points
        closest_indices = []
        # Loop through the number of clusters you have
        for i in range(num_clusters):
            # Get the list of distances from that particular cluster center
            distances = np.linalg.norm(vectors - kmeans.cluster_centers_[i], axis=1)

            # Find the list position of the closest one (using argmin to find the smallest distance)
            closest_index = np.argmin(distances)

            # Append that position to your closest indices list
            closest_indices.append(closest_index)
    else:
        closest_indices = [k for k in range(len(vectors))]

    selected_indices = sorted(closest_indices)
    return selected_indices


def text_upload(pdf_doc):
    with NamedTemporaryFile(dir="../tmp", suffix=".pdf") as f:
        f.write(pdf_doc.getbuffer())
        text = fitz.open(f.name)
    return text


def get_num_tokens(llm, text):
    print(f"This text has {llm.get_num_tokens(text)} tokens in it")


def chunking(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=3000
    )
    docs = text_splitter.create_documents([text])
    print(f"Now our book is split up into {len(docs)} documents")
    return docs


def embedding(docs):
    embeddings = OllamaEmbeddings(base_url="http://localhost:11434", model="zephyr")
    vectors = embeddings.embed_documents([x.page_content for x in docs])
    return vectors


def chunks_summaries(docs, selected_indices, llm):
    map_prompt = """
                        You will be given a part from an article enclosed in triple backticks (```)
                        Your goal is to give a summary of this part.

                        ```{text}```
                        FULL SUMMARY:
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

        # Append that summary to your list
        summary_list.append(chunk_summary)

    summaries = "\n".join(summary_list)
    return summaries


def convert_to_document(text):
    doc = Document(page_content=text)
    return doc


def combine_summary(summaries, llm):
    combine_prompt = """
                        You will be given a parts of an article enclosed in triple backticks (```)
                        Your goal is to give a verbose summary and make it look like an article.

                        ```{text}```
                        VERBOSE SUMMARY:
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
    translation = llm(f"translate in french this {text}")
    return translation

def processing(pdf_doc, llm):
    # Text extraction/cleaning
    text = merge_text(pdf_doc)

    get_num_tokens(llm, text)

    # Text chunking
    docs = chunking(text)

    # Embedding
    vectors = embedding(docs)

    # Clustering
    selected_indices = clustering(vectors)

    # Summary of selected chunks
    summaries = chunks_summaries(docs, selected_indices, llm)

    save_file("Output/Summaries", summaries)

    # Convert summaries to Document
    summaries_docs = convert_to_document(summaries)

    get_num_tokens(llm, summaries_docs.page_content)

    # Summary of the summaries
    output = combine_summary(summaries_docs, llm)

    save_file("Output/Summary", output)

    # Translation
    translation = translation_to_french(output, llm)

    save_file("Output/translation", translation)
