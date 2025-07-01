import requests
from bs4 import BeautifulSoup
from gensim.models import Word2Vec
from transformers import BertModel, BertForQuestionAnswering, BertTokenizer, BertTokenizerFast, AutoTokenizer, TFAutoModelForSeq2SeqLM, pipeline
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, PodSpec
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter


class PineconeDB():
    def __init__(self, api_key = '1bad4ac6-a29a-4dfd-9dd6-b8a5803a7901'):
        # bert_model = BertModel.from_pretrained('bert-base-uncased')
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        self.PCVectorDB = Pinecone(api_key = api_key)

    def __call__(self, index_name):
        if not index_name in self.PCVectorDB.list_indexes().names():
            self.PCVectorDB.create_index(
                name = index_name,
                dimension = 512,
                metric = 'dotproduct',
                spec = PodSpec(
                    environment="gcp-starter"
                )
            )
        return self.PCVectorDB.Index(index_name)

    
    def insert_into_vectorDB(self, scraped_data, PCVectorDB_index, vectorisor_fun):
        original_and_vectors = []
        doc_id = 0
        for documents in scraped_data:
            for doc in documents:
                if(len(doc) < 3):
                    continue
                doc = str(doc)
                sparse_values, dense_vector = vectorisor_fun(doc, get_sparse = True)
                entry = {
                    'id': f"doc_{doc_id}",
                    'sparse_values': sparse_values,
                    'values': dense_vector,
                    'metadata' : {'context': doc}
                    }
                original_and_vectors.append(entry)
                doc_id += 1
        try:
            PCVectorDB_index.upsert(original_and_vectors)
            return PCVectorDB_index
        except:
            print("Error in upserting documents !")
        return None


    def retrive_docs_for_context(self, PCVectorDB_index = None, hsparse_vals = None, hdense_vec = None, top_K = 5): 
        # Perform similarity search
        results = PCVectorDB_index.query(
            sparse_vector= hsparse_vals,
            vector = hdense_vec,
            top_k=top_K,
            include_metadata= True)
        documents = []
        for match in results['matches']:
            documents.append(match['metadata']['context'])
        return documents
    
    def delete_index(self, index_name = None):
        if index_name in self.PCVectorDB.list_indexes().names():
            self.PCVectorDB.delete_index(index_name)
            return True
        else:
            print("Error: Index not found!")
            return False


class GoogleCustomSearch():
    def __init__(self, api_key = None, cx = None):
        self.api_key = api_key
        self.cx = cx

    # Call function to search using Google Custom Search API
    def __call__(self, query, link_limit):
        url = f"https://www.googleapis.com/customsearch/v1?q={query}&key={self.api_key}&cx={self.cx}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            return list(set([item['link'] for item in data.get('items', [])]))[:link_limit]
        else:
            print(f"Failed to search for url: {query}")
        return []
    

class TransformersUtils():
    def __init__(self, model_id = 'bert-base-uncased', SentenceTransformer_id = 'multi-qa-MiniLM-L6-cos-v1'):
        self.tokenizer = BertTokenizerFast.from_pretrained(model_id)
        self.sentenceModel = SentenceTransformer(SentenceTransformer_id)
        self.sumtokenizer = AutoTokenizer.from_pretrained("stevhliu/my_awesome_billsum_model")
        self.sumModel = TFAutoModelForSeq2SeqLM.from_pretrained("stevhliu/my_awesome_billsum_model", from_pt = True)
        self.qatokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
        self.qaModel = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

    def vectorise(self, context, get_sparse = False):
        input_ids = self.tokenizer(
            context, padding=True, truncation=True,
            max_length=512
        )['input_ids']
        dense_vector = self.sentenceModel.encode(context).tolist()
        dense_vector = dense_vector + ([float(0)] * (512 - len(dense_vector)))
        if get_sparse:
            sparse_vector = dict(Counter(input_ids))
            sparse_values = {}
            sparse_values['indices'] = []
            sparse_values['values'] = []
            for vec in sparse_vector.items():
                sparse_values['indices'].append(vec[0])
                sparse_values['values'].append(float(vec[1]))
            return sparse_values, dense_vector
        return dense_vector
    
    def hybrid_scale(self, dense = None, sparse = None, alpha = 0.5):
        # check alpha value is in range
        if alpha < 0 or alpha > 1:
            raise ValueError("Alpha must be between 0 and 1")
        # scale sparse and dense vectors to create hybrid search vecs
        hsparse = {
            'indices': sparse['indices'],
            'values':  [v * (1 - alpha) for v in sparse['values']]
        }
        hdense = [v * alpha for v in dense]
        return hdense, hsparse
