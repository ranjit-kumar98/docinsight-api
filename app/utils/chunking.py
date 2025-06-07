def chunk_text(text, max_tokens=200):
    import re

    # Split by double newline or sentence end
    raw_chunks = re.split(r'\n\n|\.\s', text)

    chunks = []
    current_chunk = ""
    current_tokens = 0

    for part in raw_chunks:
        part = part.strip()
        token_count = len(part.split())

        if current_tokens + token_count > max_tokens:
            chunks.append(current_chunk.strip())
            current_chunk = part
            current_tokens = token_count
        else:
            current_chunk += " " + part
            current_tokens += token_count

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks
