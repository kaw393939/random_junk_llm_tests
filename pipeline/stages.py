import spacy
from transformers import AutoTokenizer

# Load SpaCy and tokenizer globally for performance
nlp = spacy.load("en_core_web_sm")
tokenizer = AutoTokenizer.from_pretrained("gpt2", model_max_length=1024)

def chunk_text_by_sentences(text, max_tokens, overlap):
    """Chunks text into token-limited chunks while respecting sentence boundaries."""
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    current_chunk = []
    current_token_count = 0

    for sentence in sentences:
        sentence_tokens = tokenizer.encode(sentence, add_special_tokens=False)
        sentence_token_count = len(sentence_tokens)

        if current_token_count + sentence_token_count <= max_tokens:
            current_chunk.append(sentence)
            current_token_count += sentence_token_count
        else:
            yield " ".join(current_chunk)
            # Include overlap
            current_chunk = current_chunk[-overlap:] if overlap else []
            current_chunk.append(sentence)
            current_token_count = sum(len(tokenizer.encode(s, add_special_tokens=False)) for s in current_chunk)
    
    if current_chunk:
        yield " ".join(current_chunk)
