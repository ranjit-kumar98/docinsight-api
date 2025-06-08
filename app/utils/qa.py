from transformers import pipeline

# Load the QA pipeline once
qa_pipeline = pipeline(
    "question-answering",
    model="distilbert-base-uncased-distilled-squad",
    tokenizer="distilbert-base-uncased-distilled-squad"
)

def answer_question(question: str, context: str) -> str:
    """
    Returns the extracted answer for `question` given the `context`.
    """
    result = qa_pipeline(question=question, context=context)
    return result.get("answer", "")
