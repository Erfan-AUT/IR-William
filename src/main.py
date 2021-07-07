import json
import pandas as pd
from math import log2, sqrt
from heapq import heappop, heappush


half_space = "\u200c"


def reverse_sorted_dict(nary: dict):
    return {k: v for k, v in sorted(nary.items(), key=lambda item: item[1], reverse=True)}


class Processing:
    def __init__(self, docs: pd.DataFrame) -> None:
        self.docs = docs
        self.normal = Normalization()
        self.doc_lengths = dict()
        self.champions = dict()

        self.gen_tokens()
        self.create_inv_idx()
        self.gen_champion_list()

    # Generate for each dictionary term t, the r docs of highest weight in t’s postings. (here r is all of them because we don't know k)
    def gen_champion_list(self):
        for token in self.tokens:
            self.champions[token] = dict()
            for id in self.inv_idx[token].keys():
                self.champions[token][id] = self.inv_idx[token][id] / \
                    self.doc_lengths[id]
            self.champions[token] = reverse_sorted_dict(self.champions[token])

    # Generate and normalize tokens from input dataset
    def gen_tokens(self):
        tokens = set()
        for content in self.docs['content']:
            tokens |= self.normal.normalize_tokens(self.tokenize(content))
        self.tokens = tokens

    def tokenize(self, text: str) -> set:
        return set(text.split())

    def create_inv_idx(self):
        self.inv_idx = dict()
        for token in self.tokens:
            self.inv_idx[token] = dict()
            for id, doc, _ in self.docs.itertuples(index=False):
                if token in doc.split():
                    self.inv_idx[token][id-1] = self.tf(token, doc)

        for id, doc, _ in self.docs.itertuples(index=False):
            self.doc_lengths[id-1] = self.gen_doc_lengths(id-1, doc)

    def gen_doc_lengths(self, id: int, doc: str):
        # Normalize to remove useless tokens, get(id, 0) to discard non-existing terms in docs.
        return sqrt(sum([self.inv_idx[token].get(id, 0) ** 2 for token in self.normal.normalize_tokens(self.tokenize(doc))]))

    # Single word query from phase 1
    def single_query(self, q: str) -> set:
        return self.inv_idx.get(q, set())

    # Multi word query from phase 1
    def multi_query(self, q: str) -> set:
        scores = dict()
        for item in q.split():
            for id in self.inv_idx.get(item, set()):
                if id in scores.keys():
                    scores[id] += 1
                else:
                    scores[id] = 1
        return {k for k, _ in sorted(scores.items(), key=lambda item: item[1])}

    # Term Frequency
    def tf(self, term: str, doc: str) -> float:
        return 1 + log2(doc.split().count(term))

    # Inverse Document Frequency
    def idf(self, term: str) -> float:
        return log2(len(self.docs) / len(self.inv_idx[term]))

    def tf_idf(self, term: str, doc: str):
        return self.tf(term, doc) * self.idf(term)

    def cos_similarity(self, q: str, id: int, doc: str):
        return sum([self.tf_idf(token, doc) if token in doc.split() else 0 for token in q.split()]) / self.doc_lengths[id]

    # Generate scores for each document given the query
    def gen_scores(self, q: str):
        scores = dict()
        q_words = q.split()
        for word, champ in self.champions.items():
            if word in q_words:
                for champ_id in champ.keys():
                    if champ_id not in scores.keys():
                        scores[champ_id] = 0
                    scores[champ_id] += self.cos_similarity(
                        word, champ_id, self.docs['content'][champ_id])

        self.scores = scores

    def best_k_heap(self, k: int):
        heap = []
        for id, score in self.scores.items():
            heappush(heap, (score, id))

        worst = dict()
        for _ in range(k):
            score, id = heappop(heap)
            worst[id] = score

        return reverse_sorted_dict(worst)

    def best_k_sort(self, k: int):
        return reverse_sorted_dict(self.scores)[:k]

    def best_k(self, k: int, heap_or_sort=True):
        if heap_or_sort:
            return self.best_k_heap(k)
        else:
            return self.best_k_sort(k)



class Normalization:

    def __init__(self):
        with open("data/words.json", encoding="utf-8") as f:
            norm_words = json.load(f)
        self.pronouns = norm_words["pronouns"]
        self.prepositions = norm_words["prepositions"]
        self.punctuations = norm_words["punctuations"]
        self.suffixes = norm_words["suffixes"]
        self.arabic_plurals = norm_words["arabic_plurals"]

    def normalize_tokens(self, tokens: set) -> set:
        for p in (self.prepositions + self.punctuations + self.pronouns):
            tokens.discard(p)

        new_tokens = set()

        for tkn in tokens:
            new_token = tkn

            for pun in self.punctuations:
                if pun in new_token:
                    new_token = new_token.replace(pun, "")

            if half_space in new_token:
                for suff in self.suffixes:
                    if new_token.endswith(suff):
                        new_token = new_token[:-len(suff)]
                new_token = new_token.replace(half_space, "")

            if tkn in self.arabic_plurals.keys():
                new_token = self.arabic_plurals[tkn]

            new_tokens.add(new_token)

        return new_tokens


def main():

    length = input(
        "Enter the number of docs you want to search through (starts from the 1st document), m for max \n")
    print("Search engine started, wait for initialization...")

    data = pd.read_excel("data/data.xlsx")

    if length == "m":
        data_head = data
    else:
        data_head = data.head(int(length))

    p = Processing(data_head)

    while(True):
        in_str = input("Enter your query, !q to exit \n")
        if in_str == "!q":
            break
        # if len(in_str.split()) == 1:
        #     ids = p.single_query(in_str)
        # else:
        #     ids = p.multi_query(in_str)

        # for i in ids:
        #     print([i, data_head["url"][i-1]])
        p.gen_scores(in_str)
        k = min(5, len(p.scores))
        print("The documents with the best scores are in this order:")
        print(p.best_k(k, heap_or_sort=True))


if __name__ == "__main__":
    main()
