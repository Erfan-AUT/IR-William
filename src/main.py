import json
import pandas as pd
from math import log2, sqrt
from heapq import heappop, heappush
import time
from typing import Any, Union


half_space = "\u200c"


def reverse_sorted_dict(nary: dict):
    return {k: v for k, v in sorted(nary.items(), key=lambda item: item[1], reverse=True)}


class Clustering:
    def __init__(self):
        self.data = data
        self.clusters = {}
        self.cluster_count = 0
        self.cluster_distances = {}
        self.cluster_centers = {}
    
    # K-means clustering on a given data set in the form of dictionaries
    def k_means(self):       
        self.cluster_centers = 
        # Calculate the distance between each point and each cluster center
        for i in range(len(self.data)):       
            for j in range(self.cluster_count):       
                self.cluster_distances[j] = self.calculate_distance(self.data[i], self.cluster_centers[j])       
            # Find the smallest distance and assign the point to that cluster       
            self.clusters[min(self.cluster_distances, key=self.cluster_distances.get)] = self.clusters.get(min(self.cluster_distances, key=self.cluster_distances.get), [])       
            self.clusters[min(self.cluster_distances, key=self.cluster_distances.get)].append(i)       
            # Calculate the new cluster center       
            for j in range(self.cluster_count):       
                self.cluster_centers[j] = self.calculate_center(self.clusters[j])       
        # Calculate the average distance between each point and its cluster center       
        for i in range(len(self.data)):       
            for j in range(self.cluster_count):       
                self.cluster_distances[j] = self.calculate_distance(self.data[i], self.cluster_centers[j])       
        return self.clusters, self.cluster_distances, self.cluster_centers

    # Calculating the distance between a point and a cluster center
    def calculate_distance(self, point: dict, center: dict) -> float:
        distance = 0.0
        for key in point:
            distance += (point[key] - center[key]) ** 2
        return sqrt(distance)




class Processing:
    def __init__(self, docs: pd.DataFrame, has_champion=True, load=False, load_addr=None) -> None:
        self.normal = Normalization()
        self.doc_lengths = dict()
        self.has_champion = has_champion

        if load:
            self.docs = pd.read_pickle(load_addr)
        else:
            self.docs = docs

        print("Tokenizing docs")
        start_time = time.time()
        self.docs["idx"] = docs.apply(
            lambda row: self.gen_doc_idx(row['content']), axis=1)
        print("--- %s seconds ---" % (time.time() - start_time))

        print("Generating tokens")
        start_time = time.time()
        self.gen_tokens()
        print("--- %s seconds ---" % (time.time() - start_time))

        print("Creating inverse index")
        start_time = time.time()
        self.create_inv_idx()
        print("--- %s seconds ---" % (time.time() - start_time))

        if self.has_champion:
            self.champions = dict()
            print("Generating champion list")
            start_time = time.time()
            self.gen_champion_list()
            print("--- %s seconds ---" % (time.time() - start_time))

    def gen_doc_idx(self, content):
        tokens = self.doc_tokens(content, list)
        return {token: tokens.count(token) for token in tokens}

    # Generate for each dictionary term t, the r docs of highest weight in t’s postings. (here r is all of them because we don't know k)
    def gen_champion_list(self, l=None):
        for token in self.tokens:
            self.champions[token] = dict()
            for id in self.inv_idx[token].keys():
                self.champions[token][id] = self.inv_idx[token][id] / \
                    self.doc_lengths[id]
            self.champions[token] = reverse_sorted_dict(self.champions[token])
            if type(l) is int:
                self.champions[token] = self.champions[token][:l]

    def doc_tokens(self, doc: str, tokens_type: type):
        return self.normal.normalize_tokens(self.tokenize(doc, tokens_type))

    # Generate and normalize tokens from input dataset
    def gen_tokens(self):
        self.tokens = set()
        for idx in self.docs['idx']:
            self.tokens |= set(idx.keys())

    def tokenize(self, text: str, tokens_type: type):
        return tokens_type(text.split())

    # Save term frequency in inverse index
    def create_inv_idx(self):
        inv_idx = {token: dict() for token in self.tokens}
        for id, _, _, idx in self.docs.itertuples(index=False):
            for key, value in idx.items():
                inv_idx[key][id-1] = self.tf(value)
            self.doc_lengths[id-1] = self.gen_doc_length(idx)
        self.inv_idx = inv_idx

    def gen_doc_length(self, idx: dict):
        return sqrt(sum([value ** 2 for _, value in idx.items()]))

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
    def tf(self, count: int) -> float:
        return 1 + log2(count)

    # Inverse Document Frequency
    def idf(self, term: str) -> float:
        return log2(len(self.docs) / len(self.inv_idx[term]))

    def tf_idf(self, term: str, idx: dict):
        return self.tf(idx.get(term)) * self.idf(term)

    def cos_similarity(self, q_word: str, id: int):
        return sum([self.tf_idf(q_word, self.docs['idx'][id])]) / self.doc_lengths[id]

    # Generate scores for each document given the query
    def gen_scores(self, q: str):
        scores = dict()
        q_words = self.doc_tokens(q, set)
        search_area = self.champions if self.has_champion else self.inv_idx
        for q_word in q_words:
            # If query word is not in champion list, then ignore it
            for champ_id in search_area.get(q_word, dict()).keys():
                scores[champ_id] = scores.get(champ_id, 0) + self.cos_similarity(q_word, champ_id)
            
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

    def save(self, addr):
        self.docs.to_pickle(addr)

ListSet = Union[list, set]

class Normalization:

    def __init__(self):
        with open("data/words.json", encoding="utf-8") as f:
            norm_words = json.load(f)
        self.pronouns = norm_words["pronouns"]
        self.prepositions = norm_words["prepositions"]
        self.punctuations = norm_words["punctuations"]
        self.suffixes = norm_words["suffixes"]
        self.arabic_plurals = norm_words["arabic_plurals"]

    def add(self, collection: ListSet, item: Any):
        if type(collection) is list:
            collection.append(item)
        elif type(collection) is set:
            collection.add(item)
        else:
            pass

    def remove(self, collection: ListSet, item: Any):
        if type(item) is list:
            try:
                collection.remove(item)
            except:
                pass
        elif type(item) is set:
            collection.discard(item)
        else:
            pass

    def normalize_tokens(self, tokens: ListSet):
        for p in (self.prepositions + self.punctuations + self.pronouns):
            self.remove(tokens, p)

        new_tokens = type(tokens)()

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

            self.add(new_tokens, new_token)

        return new_tokens


def main():

    length = input(
        "Enter the number of docs you want to search through (starts from the 1st document), m for max \n")
    print("Search engine started, wait for initialization...")

    data = pd.read_excel("data/phase2/data.xlsx")

    if length == "m":
        data_head = data
    else:
        data_head = data.head(int(length))

    p = Processing(data_head, has_champion=False)

    while(True):
        in_str = input("Enter your query, !q to exit \n")
        if in_str == "!q":
            break

        start_time = time.time()

        p.gen_scores(in_str)
        k = min(5, len(p.scores))
        print("The documents with the best scores are in this order (add +1 to the ids):")
        print(p.best_k(k, heap_or_sort=True))
        print("Querying took:")
        print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__":
    main()
