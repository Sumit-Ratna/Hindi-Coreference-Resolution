import re
# import os
# print("Current working directory:", os.getcwd())

INPUT_FILE = "dataset.norm"
OUTPUT_FILE = "dataset.conll"

def extract_ids(cols):
    """Extract coreference IDs like i1, i2, etc. from columns."""
    ids = []
    for c in cols:
        if c and c != "_":
            # Match patterns like "i1%1:t1", "i1:t1", or "i1"
            matches = re.findall(r'([iI]\d+)', c)
            ids.extend(matches)
    return ids

def convert_to_conll(lines):
    conll = []
    doc = "doc1"
    part = 0
    token_id = 0
    open_mentions = {}

    for line in lines:
        line = line.strip()

        # Sentence break
        if not line:
            conll.append("")
            token_id = 0
            continue

        parts = line.split("\t")
        token = parts[0]
        cols = parts[1:]

        # Extract IDs
        ids = extract_ids(cols)
        ids = [int(i[1:]) for i in ids if i.lower().startswith("i")]

        # Build coreference column
        coref_col = "_"
        if ids:
            notation = []
            for cid in ids:
                if cid not in open_mentions:
                    open_mentions[cid] = True
                    notation.append(f"({cid}")
                else:
                    notation.append(f"{cid})")
                    open_mentions.pop(cid, None)
            coref_col = "|".join(notation)

        conll.append(f"{doc}\t{part}\t{token_id}\t{token}\t_\t_\t_\t_\t{coref_col}")
        token_id += 1

    return "\n".join(conll)

# ---- run conversion ----
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    lines = f.readlines()

conll_text = convert_to_conll(lines)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write(conll_text)

print("Conversion complete!")
print(f"Saved to: {OUTPUT_FILE}")
