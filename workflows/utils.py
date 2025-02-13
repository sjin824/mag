# Need some work on there: the logic of deduplication.
def entity_deduplicate(ents_tuple):
    merged_list = []
    for ents in ents_tuple:
        merged_list.extend(ents)
    return list(dict.fromkeys(merged_list))