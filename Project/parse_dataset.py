import json
import re
from collections import defaultdict, Counter

# Regex patterns
ENTITY_ID_PATTERN = re.compile(r"(i\d+)(?:%(\d+))?")
RELATION_PATTERN = re.compile(r"([A-Za-z-]+):(i\d+)")

def parse_dataset(file_path):
    sentences = []
    current_sentence = []
    sentence_id = 0

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                # Sentence boundary
                if current_sentence:
                    sentence_id += 1
                    sentences.append({
                        "sentence_id": sentence_id,
                        "tokens": current_sentence
                    })
                    current_sentence = []
            else:
                parts = line.split('\t')
                if len(parts) < 1:
                    continue
                word = parts[0]
                cols = parts[1:]

                # Extract mention_spans (from column 1, mention encoding)
                mention_spans = []
                if len(cols) > 0 and cols[0] and cols[0] != "_":
                    matches = ENTITY_ID_PATTERN.findall(cols[0])
                    for match in matches:
                        eid, pos = match
                        position = int(pos) if pos else 0
                        mention_spans.append((eid, position))

                # Extract head_entity_ids (from column 2, headword mapping)
                head_entity_ids = []
                if len(cols) > 1 and cols[1] and cols[1] != "_":
                    matches = ENTITY_ID_PATTERN.findall(cols[1])
                    head_entity_ids = [match[0] for match in matches]  # ignore pos for heads
                    head_entity_ids = list(set(head_entity_ids))

                # Extract all relations (from all columns)
                relations = []
                for col in cols:
                    if col and col != "_":
                        matches = RELATION_PATTERN.findall(col)
                        for match in matches:
                            relation_type, target_entity = match
                            relations.append({
                                "type": relation_type,
                                "target": target_entity
                            })

                token = {
                    "word": word,
                    "mention_spans": mention_spans,
                    "head_entity_ids": head_entity_ids,
                    "relations": relations
                }
                current_sentence.append(token)

    # Handle last sentence if no trailing blank line
    if current_sentence:
        sentence_id += 1
        sentences.append({
            "sentence_id": sentence_id,
            "tokens": current_sentence
        })

    return sentences

def reconstruct_spans(sentences):
    spans = []
    for sentence in sentences:
        sid = sentence['sentence_id']
        tokens = sentence['tokens']
        # Group by eid
        eid_to_indices = defaultdict(list)
        eid_to_positions = defaultdict(list)
        for idx, token in enumerate(tokens):
            for eid, pos in token['mention_spans']:
                eid_to_indices[eid].append(idx)
                eid_to_positions[eid].append((idx, pos))

        for eid, indices in eid_to_indices.items():
            indices.sort()
            # Group consecutive
            groups = []
            current_group = [indices[0]]
            for i in indices[1:]:
                if i == current_group[-1] + 1:
                    current_group.append(i)
                else:
                    groups.append(current_group)
                    current_group = [i]
            groups.append(current_group)

            for group in groups:
                start = group[0]
                end = group[-1]
                # Find head: where pos == 1, else last
                head = end
                for idx, pos in eid_to_positions[eid]:
                    if idx in group and pos == 1:
                        head = idx
                        break
                span_text = ' '.join(tokens[i]['word'] for i in group)
                spans.append({
                    "sentence": sid,
                    "eid": eid,
                    "start": start,
                    "end": end,
                    "head": head,
                    "text": span_text
                })
    return spans

def build_true_clusters(spans, sentences):
    # Collect all eids
    all_eids = set(span['eid'] for span in spans)

    # Initialize union-find
    parent = {eid: eid for eid in all_eids}
    rank = {eid: 0 for eid in all_eids}

    # Add edges from relations
    for sentence in sentences:
        for token in sentence['tokens']:
            for rel in token['relations']:
                target = rel['target']
                if target in all_eids:
                    # Source is eids in mention_spans of this token
                    sources = [eid for eid, _ in token['mention_spans']]
                    for source in sources:
                        if source in all_eids:
                            union(parent, rank, source, target)

    # Find roots
    eid_to_root = {eid: find(parent, eid) for eid in all_eids}

    # Group spans by root
    clusters = defaultdict(list)
    for span in spans:
        root = eid_to_root[span['eid']]
        clusters[root].append(span)

    return dict(clusters)

def find(parent, i):
    if parent[i] == i:
        return i
    parent[i] = find(parent, parent[i])
    return parent[i]

def union(parent, rank, x, y):
    xroot = find(parent, x)
    yroot = find(parent, y)
    if rank[xroot] < rank[yroot]:
        parent[xroot] = yroot
    elif rank[xroot] > rank[yroot]:
        parent[yroot] = xroot
    else:
        parent[yroot] = xroot
        rank[xroot] += 1

def build_entity_clusters(sentences):
    spans = reconstruct_spans(sentences)
    clusters = build_true_clusters(spans, sentences)
    return clusters, spans

def compute_statistics(sentences, clusters, spans):
    total_sentences = len(sentences)
    total_tokens = sum(len(s['tokens']) for s in sentences)
    total_mention_spans = len(spans)
    unique_entities = len(clusters)

    # Top 5 largest clusters
    cluster_sizes = [(root, len(spans_list)) for root, spans_list in clusters.items()]
    cluster_sizes.sort(key=lambda x: x[1], reverse=True)
    top_5_clusters = cluster_sizes[:5]

    # Relation distribution
    relation_types = []
    for sentence in sentences:
        for token in sentence['tokens']:
            for rel in token['relations']:
                relation_types.append(rel['type'])
    relation_dist = Counter(relation_types)

    # New stats
    if clusters:
        avg_cluster_size = sum(len(spans_list) for spans_list in clusters.values()) / len(clusters)
        max_cluster_size = max(len(spans_list) for spans_list in clusters.values())
    else:
        avg_cluster_size = 0
        max_cluster_size = 0

    # Max span length
    max_span_length = max((span['end'] - span['start'] + 1) for span in spans) if spans else 0

    # Distribution of mention types: single-token vs multi-token
    single_token_spans = sum(1 for span in spans if span['start'] == span['end'])
    multi_token_spans = total_mention_spans - single_token_spans

    return {
        'total_sentences': total_sentences,
        'total_tokens': total_tokens,
        'total_mention_spans': total_mention_spans,
        'unique_entities': unique_entities,
        'top_5_clusters': top_5_clusters,
        'relation_distribution': dict(relation_dist),
        'avg_cluster_size': avg_cluster_size,
        'max_cluster_size': max_cluster_size,
        'max_span_length': max_span_length,
        'single_token_spans': single_token_spans,
        'multi_token_spans': multi_token_spans
    }

if __name__ == "__main__":
    file_path = "Project/dataset.norm"
    sentences = parse_dataset(file_path)
    clusters, spans = build_entity_clusters(sentences)
    stats = compute_statistics(sentences, clusters, spans)

    # Print statistics
    print("Total Sentences:", stats['total_sentences'])
    print("Total Tokens:", stats['total_tokens'])
    print("Total Mention Spans:", stats['total_mention_spans'])
    print("Unique Entities:", stats['unique_entities'])
    print("Top 5 Largest Clusters:")
    for eid, size in stats['top_5_clusters']:
        print(f"  {eid}: {size} spans")
    print("Relation Distribution:")
    for rel_type, count in stats['relation_distribution'].items():
        print(f"  {rel_type}: {count}")
    print("Average Cluster Size:", f"{stats['avg_cluster_size']:.2f}")
    print("Max Cluster Size:", stats['max_cluster_size'])
    print("Max Span Length:", stats['max_span_length'])
    print("Single-token Spans:", stats['single_token_spans'])
    print("Multi-token Spans:", stats['multi_token_spans'])

    # Print first 3 sentences as JSON
    # first_three = sentences[:3]
    # print("\nFirst 3 Sentences:")
    # print(json.dumps(first_three, ensure_ascii=False, indent=2))
    with open("all_sentences.json", "w", encoding="utf-8") as f:
        json.dump(sentences, f, ensure_ascii=False, indent=2)