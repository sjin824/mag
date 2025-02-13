import stanza

stanza_nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency')

def get_phrases(tree, label):
    if tree.is_leaf():
        return []
    results = []
    for child in tree.children:
        results += get_phrases(child, label)

    if tree.label == label:
        return [' '.join(tree.leaf_labels())] + results
    else:
        return results

def stanza_get_ents(batch_texts, is_NP=True, is_VP=True, is_upos=True):
    results = []
    for text in batch_texts:
        stanza_doc = stanza_nlp(text)
        result = []
        for sent in stanza_doc.sentences:
            if is_NP:
                result.extend(get_phrases(sent.constituency, 'NP'))
            if is_VP:
                result.extend(get_phrases(sent.constituency, 'VP'))
            if is_upos:
                result.extend(
                    [word.text for word in sent.words if word.upos in ['NOUN', 'VERB', 'ADJ', 'ADV']]
                )
        results.append(result)
    return results
