import re

def chunk_text(text, lines_per_chunk=10):
    # 1. Normalize line breaks, remove empty lines
    lines = [ln.strip() for ln in text.replace('\r','\n').split('\n') if ln.strip()]
    
    chunks = []
    # 2. Header chunk (first 5 lines, typically account details + summary)
    if len(lines) > 5:
        header = " ".join(lines[:5])
        chunks.append(header)
        start_idx = 5
    else:
        start_idx = 0

    # 3. Group the rest into fixed-size chunks
    for i in range(start_idx, len(lines), lines_per_chunk):
        group = lines[i : i + lines_per_chunk]
        chunks.append(" ".join(group))

    return chunks
