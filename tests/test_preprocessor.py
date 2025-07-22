from src.components.preprocessor import SmartAdaptiveChunker
from pprint import pprint

sample_blocks = [
    # 1. Normal short sentence
    "Artificial Intelligence is transforming industries globally.",

    # 2. Previously noisy JSON-like block now cleaned
    "Data received successfully. Status is OK. Timestamp: 2025-07-18 10:00:00",

    # 3. Symbol-heavy but meaningful block
    "Header Alert: System failure occurred due to overload. Please restart immediately!",

    # 4. Repetitive phrasing (noise handled previously)
    "AI improves healthcare. AI improves healthcare. AI improves healthcare by assisting in diagnosis.",

    # 5. Very long academic paragraph
    "Artificial Intelligence, often abbreviated as AI, refers to the simulation of human intelligence in machines that are programmed to think and act like humans. It can be applied in a variety of fields such as medical diagnosis, robotics, stock trading, and more. AI technologies include machine learning, natural language processing, and computer vision.",

    # 6. A cleaned up instruction block from previously delimited HTML/docs
    "Steps to reset your password: Go to settings. Click 'Reset Password'. Enter your registered email. Follow instructions sent to your inbox.",

    # 7. Text from FAQ/help article
    "To update your billing info, navigate to the 'Billing & Subscription' tab and select 'Update Payment Method'. Ensure your card details are current.",

    # 8. Header-style block cleaned from markup
    "Release Notes - Version 2.3.4: New features added. Major bug fixes. Performance improvements implemented.",

    # 9. News-style text
    "The Prime Minister addressed the nation on Friday, outlining new economic measures to support startups and innovation in the technology sector.",

    # 10. Long unstructured (was noisy, now readable)
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Quisque fermentum, lorem ut hendrerit interdum, nunc eros tempor libero, sit amet sodales risus ipsum in purus.",

    # 11. Previously symbol/JSON/delimiter-heavy block
    "User feedback report: Positive responses increased 27% post-update. Critical issues decreased. Overall satisfaction improved.",

    # 12. Help desk alert
    "Reminder: Please update your security questions. This helps recover your account if you forget your password or lose access.",

    # 13. Legal notice style
    "This document is confidential and intended solely for the recipient. Unauthorized sharing or distribution is strictly prohibited.",

    # 14. Very long single topic (for token-aware chunking)
    "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It involves algorithms that can learn from and make predictions on data. These algorithms operate by building a model based on sample data, known as training data, in order to make predictions or decisions without being specifically programmed to perform the task. Examples include recommendation engines, spam filters, and fraud detection systems.",

    # 15. Cleaned sentence from marketing or blog block
    "Unlock your team’s potential with cutting-edge AI tools that streamline your workflow and boost efficiency by over 50%."
]



# Instantiate chunker with desired parameters
chunker = SmartAdaptiveChunker(
        model_name = "bert-base-uncased",
        max_tokens = 28,
        min_tokens = 4,
        similarity_threshold = 0.80
)

# Step-by-step outputs
print("========== STEP 3: Delimiter-Based Logical Splitting ==========\n")
step3_blocks = chunker._delimiter_split(sample_blocks)
for i, blk in enumerate(step3_blocks, 1):
    print(f"[{i}] {blk}\n")

print("\n========== STEP 5: Semantic Chunk Refinement ==========\n")
step5_chunks = chunker._semantic_refine(step3_blocks)
for i, chunk in enumerate(step5_chunks, 1):
    print(f"[{i}] {chunk}\n")

print("\n========== FINAL OUTPUT WITH METADATA ==========\n")
final_chunks = chunker.process(sample_blocks)
pprint(final_chunks)
