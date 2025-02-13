import spacy

spacy_nlp = spacy.load('en_core_web_lg')

def spacy_get_ents(batch_texts, is_noun_chunks=True, is_named_entities=True):
    results = []
    for spacy_doc in spacy_nlp.pipe(batch_texts):
        result = []
        if is_noun_chunks:
            result += [noun_chunk.text for noun_chunk in spacy_doc.noun_chunks]
        if is_named_entities:
            result += [entity.text for entity in spacy_doc.ents]
        results.append(result)
    return results
