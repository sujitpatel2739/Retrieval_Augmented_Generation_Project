from tensorflow import Variable as tf_Variable, math as tf_math
from RAGutils import *
import pickle
import pandas as pd
import re
import sys
import os


from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER


# Objects initialization and calling
GCS = GoogleCustomSearch(
    api_key='AIzaSyDpsYxahm-USx131yjLGctwFzvufj7Yee8', cx="94086cd10dad34239")

PC = PineconeDB(api_key='1bad4ac6-a29a-4dfd-9dd6-b8a5803a7901')
def reset_pinecone_conn():
    global PC
    PC = PineconeDB(api_key='1bad4ac6-a29a-4dfd-9dd6-b8a5803a7901')

TFUtils = TransformersUtils(
    model_id='bert-base-uncased', SentenceTransformer_id='multi-qa-MiniLM-L6-cos-v1')


# Function to scrape data from a webpage
def get_scraped_data(link):
    response = requests.get(link)
    if response.status_code != 200:
        return ""
    soup = BeautifulSoup(response.content, 'html.parser')
    paragraphs = soup.find_all("p")
    paragraph_texts = [re.sub(r'\[.*?\]', '', p.get_text(separator=' ', strip=False))
                       for p in paragraphs if len(p) > 2]
    return paragraph_texts


def scrape_data(web_links):
    documents = []
    for link in web_links:
        try:
            documents.append(get_scraped_data(link))
        except:
            pass
    return documents


def retrive_context(query, vectorDB_index, top_k=10):
    sparse_values, dense_vector = TFUtils.vectorise(query, get_sparse=True)
    hdense_vec, hsparse_vals = TFUtils.hybrid_scale(
        dense=dense_vector, sparse=sparse_values, alpha=1)
    retrived_context = PC.retrive_docs_for_context(
        vectorDB_index, hsparse_vals, hdense_vec, top_K=top_k)
    return retrived_context


def summarize(context):
    inputs = TFUtils.sumtokenizer(context, return_tensors="tf").input_ids
    context_len = len(context)
    print(context_len)
    min_length = context_len / 5 if 1000 >= context_len >= 500 else context_len / \
        3 if 500 >= context_len >= 250 else context_len / \
        2 if 250 >= context_len >= 150 else context_len - 1
    outputs = TFUtils.sumModel.generate(inputs, min_length=int(
        min_length), max_length=context_len, do_sample=False)
    return TFUtils.sumtokenizer.decode(outputs[0], skip_special_tokens=True)


def answer_question(question, answer_text):
    '''
    Takes a `question` string and an `answer_text` string (which contains the
    answer), and identifies the words within the `answer_text` that are the
    answer. Prints them out.
    '''
    # ======== Tokenize ========
    # Apply the tokenizer to the input text, treating them as a text-pair.
    input_ids = TFUtils.qatokenizer.encode(question, answer_text)

    # Report how long the input sequence is.
    print('Query has {:,} tokens.\n'.format(len(input_ids)))

    # ======== Set Segment IDs ========
    # Search the input_ids for the first instance of the `[SEP]` token.
    sep_index = input_ids.index(TFUtils.qatokenizer.sep_token_id)

    # The number of segment A tokens includes the [SEP] token istelf.
    num_seg_a = sep_index + 1

    # The remainder are segment B.
    num_seg_b = len(input_ids) - num_seg_a

    # Construct the list of 0s and 1s.
    segment_ids = [0]*num_seg_a + [1]*num_seg_b

    # There should be a segment_id for every input token.
    assert len(segment_ids) == len(input_ids)

    # ======== Evaluate ========
    # Run our example question through the model.
    start_scores, end_scores = TFUtils.qaModel(tf_Variable([input_ids]),  # The tokens representing our input text.
                                               token_type_ids=tf_Variable([segment_ids]))  # The segment IDs to differentiate question from answer_text

    # ======== Reconstruct Answer ========
    # Find the tokens with the highest `start` and `end` scores.
    answer_start = tf_math.argmax(start_scores)
    answer_end = tf_math.argmax(end_scores)

    # Get the string versions of the input tokens.
    tokens = TransformersUtils.qatokenizer.convert_ids_to_tokens(input_ids)

    # Start with the first token.
    answer = tokens[answer_start]

    # Select the remaining answer tokens and join them with whitespace.
    for i in range(answer_start + 1, answer_end + 1):

        # If it's a subword token, then recombine it with the previous token.
        if tokens[i][0:2] == '##':
            answer += tokens[i][2:]

        # Otherwise, add a space then the token.
        else:
            answer += ' ' + tokens[i]

    return answer


def export_pdf(data, filename='Output.pdf'):
    doc = SimpleDocTemplate(filename, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph(f"<strong>{filename}</strong>", styles["Heading1"]))
    story.append(HRFlowable(width="100%", thickness=1,
                 lineCap='round', color=colors.black))
    for document in data:
        p = Paragraph(document, styles["Normal"])
        story.append(p)
        story.append(HRFlowable(width="100%", thickness=1,
                            lineCap='round', color=colors.black))

    doc.build(story)


# Main function
def main():
    os.system('cls')
    # Define your search queries
    
    query = input("Enter a query to perform RAG: ")

    global PC
    index_name = str(''.join(re.findall(r'[a-zA-Z]', query)).lower())

    print(f"Searching information on {query}, Please wait!")

    if not index_name in PC.PCVectorDB.list_indexes().names():
        for idx in PC.PCVectorDB.list_indexes().names():
            PC.delete_index(idx)

        vectorIndex = PC(index_name=index_name)

        web_links = GCS(query, link_limit=5)
        scraped_data = scrape_data(web_links)

        print(f"Inserting data into the vector database for simmilarity search! (Efficient Search)!")
        vectorIndex = PC.insert_into_vectorDB(
            scraped_data=scraped_data, PCVectorDB_index=vectorIndex, vectorisor_fun=TFUtils.vectorise)

    reset_pinecone_conn()
    vectorIndex = PC(index_name=index_name)

    r_query = input("Input a query to search data from vector database: ")
    retrived_context = retrive_context(r_query, vectorIndex, top_k=50)
    # Storing as combined PDF file.

    res = input("Data retrived! Do you want to export to PDF ?    Y/N ?")
    if res.lower() == 'y':
        export_pdf(retrived_context, filename=f'{query}.pdf')
        print("PDF exported successfully!")

    # summarized = []
    # for doc in retrived_context:
    #     summary = summarize(doc)
    #     print(summary)


    # retrive_answers = answer_question("who is optimus prime", summarized_context)
    # print(retrive_answers)


if __name__ == "__main__":
    main()
