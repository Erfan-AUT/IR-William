import json
import pandas as pd

half_space = "\u200c"

def tokenize(plain: str) -> set:
    return set(plain.split())

def create_inv_idx(tokens: set, documents: pd.DataFrame):
    inv_idx = dict()
    for token in tokens:
        inv_idx[token] = set()
        for id, doc in zip(documents["id"], documents["content"]):
            if token in doc.split():
                inv_idx[token].add(id)
    return inv_idx

def single_query(q: str, inv_idx: dict[str, set]) -> set:
    return inv_idx.get(q, set())

def multi_query(q: str, inv_idx: dict[str, set]) -> set:
    scores = dict()
    for item in q.split():
        for id in inv_idx.get(item, set()):
            if id in scores.keys():
                scores[id] += 1
            else:
                scores[id] = 1
    return {k for k, _ in sorted(scores.items(), key=lambda item: item[1])}

class Normalization:

    def __init__(self):
        with open("words.json", encoding="utf-8") as f:
            norm_words = json.load(f)
        self.pronouns = norm_words["pronouns"]
        self.prepositions = norm_words["prepositions"]
        self.punctuations = norm_words["punctuations"]
        self.suffixes = norm_words["suffixes"]

    def normalize(self, inv_idx: dict[str, set]):

        keys = list(inv_idx.keys())

        for token in keys:
            
            new_token = token
            changed = False
            
            for pun in self.punctuations:
                if pun in new_token:
                    new_token = new_token.replace(pun, "")
                    changed = True

            if half_space in token:
                for suff in self.suffixes:
                    if suff in new_token:
                        new_token = new_token.replace(suff, "")
                new_token = new_token.replace(half_space, "")
                changed = True

            if new_token in inv_idx.keys():
                inv_idx[new_token] = inv_idx[new_token] | inv_idx[token]
            else:
                inv_idx[new_token] = inv_idx[token]

            if changed:
                inv_idx.pop(token)
        
    def clean_tokens(self, tokens: set) -> set:
        for p in (self.prepositions + self.punctuations + self.pronouns):
            tokens.discard(p)
        return tokens            

def main():

    print("Search engine started, wait for initialization...")

    data = pd.read_excel("data.xlsx")
    normal = Normalization()

    data_head = data.head(50)

    # print(data_head)

    tokens = set()
    for content in data_head["content"]:
        tokens |= normal.clean_tokens(tokenize(content))

    inv_idx = create_inv_idx(tokens, data_head)
    normal.normalize(inv_idx)

    while(True):
        in_str = input("Enter your query, !q to exit \n")
        if in_str == "!q":
            break

        if len(in_str.split()) == 1:
            ids = single_query(in_str, inv_idx)
        else:
            ids = multi_query(in_str, inv_idx)
        
        for i in ids:
            print([i, data_head["url"][i-1]])
        # result = {id: data_head["url"][id-1] for id in ids}
        # print(result)


if __name__ == "__main__":
    main()