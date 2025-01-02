def paragraph_maker(text, maxpadding=2):
    lines = text.splitlines()
    paragraphs = []
    current_paragraph = []
    empty_line_count = 0
    
    for i, line in enumerate(lines):
        # Skip image tags and table rows
        if line.strip().startswith('<!-- image -->') or line.strip().startswith('|'):
            continue
            
        # If line starts with ##, treat it as a separate paragraph
        if line.strip().startswith('##'):
            if current_paragraph:
                paragraphs.append(' '.join(current_paragraph))
                current_paragraph = []
            paragraphs.append(line.strip())
            empty_line_count = 0
            continue
            
        if not line.strip():
            empty_line_count += 1
            if empty_line_count >= maxpadding and current_paragraph:
                paragraphs.append(' '.join(current_paragraph))
                current_paragraph = []
        else:
            empty_line_count = 0
            current_paragraph.append(line.strip())
            
    # Add the last paragraph if there is one
    if current_paragraph:
        paragraphs.append(' '.join(current_paragraph))
    
    # Filter out empty paragraphs and clean up
    return [p.strip() for p in paragraphs if p.strip()]


def paragraph_merger(paragraphs,min_par_size,threshold) :
    newparagraphs = []
    for i,paragraph in enumerate(paragraphs) :
        if len(paragraph) < min_par_size :
            if paragraph != paragraphs[-1] :
                if len(paragraphs[i+1]) > threshold :
                    if not (paragraphs[i+1].startswith('##') or paragraphs[i+1].startswith('|') ) :
                        paragraphs[i+1] = paragraph + paragraphs[i+1]
                        continue
        newparagraphs.append(paragraph)
    return newparagraphs
