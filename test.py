import unittest
import os
from dotenv import load_dotenv
from sklearn.metrics import precision_score, recall_score, f1_score
from fuzzywuzzy import fuzz
from RAGutils import GoogleCustomSearch, PineconeDB, TransformersUtils
from main import retrive_context, answer_question, scrape_data
import warnings

warnings.filterwarnings("ignore")

# === Load environment variables ===
load_dotenv()
GCS_API_KEY = os.getenv("GCS_API_KEY")
GCS_CX = os.getenv("GCS_CX")
PC_API_KEY = os.getenv("PINECONE_API_KEY")

# === Define test dataset ===
# You can later switch this to load from CSV for >10 queries
TEST_CASES = [
    {
        "query": "Who is the Prime Minister of India?",
        "expected_answer": "Narendra Modi"
    },
    {
        "query": "What is the capital of France?",73
        "expected_answer": "Paris"
    },
    {
        "query": "What is the boiling point of water in Celsius?",
        "expected_answer": "100"
    },
    {
        "query": "What does CPU stand for?",
        "expected_answer": "Central Processing Unit"
    }
]

class TestRAGPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.GCS = GoogleCustomSearch(api_key=GCS_API_KEY, cx=GCS_CX)
        cls.PC = PineconeDB(api_key=PC_API_KEY)
        cls.TFUtils = TransformersUtils()

        cls.results = []  # Store results per test

        for case in TEST_CASES:
            query = case['query']
            expected_answer = case['expected_answer']
            index_name = ''.join(filter(str.isalnum, query)).lower()

            # Clear if exists
            for idx in cls.PC.PCVectorDB.list_indexes().names():
                cls.PC.delete_index(idx)

            # Create index
            vector_index = cls.PC(index_name=index_name)

            # Scrape and insert
            links = cls.GCS(query, link_limit=3)
            data = scrape_data(links)
            cls.PC.insert_into_vectorDB(data, vector_index, cls.TFUtils.vectorise)

            # Retrieve and answer
            retrieved = retrive_context(query, vector_index, top_k=5)
            combined_context = " ".join(retrieved[:3])  # Use 3 contexts max
            predicted_answer = answer_question(query, combined_context)

            # Append to result set
            cls.results.append({
                "query": query,
                "expected": expected_answer,
                "predicted": predicted_answer,
                "retrieved_contexts": retrieved
            })

    def test_accuracy_metrics(self):
        total = len(self.results)
        correct_exact = 0
        fuzzy_scores = []

        expected_labels = []
        predicted_labels = []

        print("\n==== Test Results ====\n")

        for result in self.results:
            exp = result["expected"].strip().lower()
            pred = result["predicted"].strip().lower()

            is_correct = exp in pred or pred in exp
            fuzzy_ratio = fuzz.partial_ratio(exp, pred)

            fuzzy_scores.append(fuzzy_ratio)
            expected_labels.append(1)
            predicted_labels.append(1 if fuzzy_ratio >= 80 else 0)

            print(f"Query: {result['query']}")
            print(f"Expected: {exp}")
            print(f"Predicted: {pred}")
            print(f"Match: {'✔️' if is_correct else '❌'} | Fuzzy: {fuzzy_ratio}%\n")

            if is_correct:
                correct_exact += 1

        # === Compute Metrics ===
        accuracy = correct_exact / total
        precision = precision_score(expected_labels, predicted_labels, zero_division=0)
        recall = recall_score(expected_labels, predicted_labels, zero_division=0)
        f1 = f1_score(expected_labels, predicted_labels, zero_division=0)
        avg_fuzzy = sum(fuzzy_scores) / total

        print("==== Final Evaluation ====")
        print(f"Exact Match Accuracy: {accuracy*100:.2f}%")
        print(f"Avg Fuzzy Match Score: {avg_fuzzy:.2f}%")
        print(f"Precision: {precision*100:.2f}%")
        print(f"Recall:    {recall*100:.2f}%")
        print(f"F1 Score:  {f1*100:.2f}%\n")

        # Optional assertions
        self.assertGreaterEqual(accuracy, 0.5, "Accuracy below acceptable level")
        self.assertGreaterEqual(avg_fuzzy, 70, "Fuzzy average too low")

if __name__ == "__main__":
    unittest.main()

import unittest
import os
from dotenv import load_dotenv
from sklearn.metrics import precision_score, recall_score, f1_score
from fuzzywuzzy import fuzz
from RAGutils import GoogleCustomSearch, PineconeDB, TransformersUtils
from main import retrive_context, answer_question, scrape_data
import warnings

warnings.filterwarnings("ignore")

# === Load environment variables ===
load_dotenv()
GCS_API_KEY = os.getenv("GCS_API_KEY")
GCS_CX = os.getenv("GCS_CX")
PC_API_KEY = os.getenv("PINECONE_API_KEY")

# === Define test dataset ===
# You can later switch this to load from CSV for >10 queries
TEST_CASES = [
    {
        "query": "Who is the Prime Minister of India?",
        "expected_answer": "Narendra Modi"
    },
    {
        "query": "What is the capital of France?",73
        "expected_answer": "Paris"
    },
    {
        "query": "What is the boiling point of water in Celsius?",
        "expected_answer": "100"
    },
    {
        "query": "What does CPU stand for?",
        "expected_answer": "Central Processing Unit"
    }
]

class TestRAGPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.GCS = GoogleCustomSearch(api_key=GCS_API_KEY, cx=GCS_CX)
        cls.PC = PineconeDB(api_key=PC_API_KEY)
        cls.TFUtils = TransformersUtils()

        cls.results = []  # Store results per test

        for case in TEST_CASES:
            query = case['query']
            expected_answer = case['expected_answer']
            index_name = ''.join(filter(str.isalnum, query)).lower()

            # Clear if exists
            for idx in cls.PC.PCVectorDB.list_indexes().names():
                cls.PC.delete_index(idx)

            # Create index
            vector_index = cls.PC(index_name=index_name)

            # Scrape and insert
            links = cls.GCS(query, link_limit=3)
            data = scrape_data(links)
            cls.PC.insert_into_vectorDB(data, vector_index, cls.TFUtils.vectorise)

            # Retrieve and answer
            retrieved = retrive_context(query, vector_index, top_k=5)
            combined_context = " ".join(retrieved[:3])  # Use 3 contexts max
            predicted_answer = answer_question(query, combined_context)

            # Append to result set
            cls.results.append({
                "query": query,
                "expected": expected_answer,
                "predicted": predicted_answer,
                "retrieved_contexts": retrieved
            })

    def test_accuracy_metrics(self):
        total = len(self.results)
        correct_exact = 0
        fuzzy_scores = []

        expected_labels = []
        predicted_labels = []

        print("\n==== Test Results ====\n")

        for result in self.results:
            exp = result["expected"].strip().lower()
            pred = result["predicted"].strip().lower()

            is_correct = exp in pred or pred in exp
            fuzzy_ratio = fuzz.partial_ratio(exp, pred)

            fuzzy_scores.append(fuzzy_ratio)
            expected_labels.append(1)
            predicted_labels.append(1 if fuzzy_ratio >= 80 else 0)

            print(f"Query: {result['query']}")
            print(f"Expected: {exp}")
            print(f"Predicted: {pred}")
            print(f"Match: {'✔️' if is_correct else '❌'} | Fuzzy: {fuzzy_ratio}%\n")

            if is_correct:
                correct_exact += 1

        # === Compute Metrics ===
        accuracy = correct_exact / total
        precision = precision_score(expected_labels, predicted_labels, zero_division=0)
        recall = recall_score(expected_labels, predicted_labels, zero_division=0)
        f1 = f1_score(expected_labels, predicted_labels, zero_division=0)
        avg_fuzzy = sum(fuzzy_scores) / total

        print("==== Final Evaluation ====")
        print(f"Exact Match Accuracy: {accuracy*100:.2f}%")
        print(f"Avg Fuzzy Match Score: {avg_fuzzy:.2f}%")
        print(f"Precision: {precision*100:.2f}%")
        print(f"Recall:    {recall*100:.2f}%")
        print(f"F1 Score:  {f1*100:.2f}%\n")

        # Optional assertions
        self.assertGreaterEqual(accuracy, 0.5, "Accuracy below acceptable level")
        self.assertGreaterEqual(avg_fuzzy, 70, "Fuzzy average too low")

if __name__ == "__main__":
    unittest.main()
