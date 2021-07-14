import json
import pandas as pd
from math import log2, sqrt
from heapq import heappop, heappush
import time
from typing import Any, Union
from sklearn.model_selection import KFold


half_space = "\u200c"
ListSet = Union[list, set]


def reverse_sorted_dict(nary: dict):
    return {k: v for k, v in sorted(nary.items(), key=lambda item: item[1], reverse=True)}


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


class Processing:
    def __init__(self, docs: pd.DataFrame, length='m', has_champion=True, load=False, load_addr=None) -> None:
        self.normal = Normalization()
        self.doc_lengths = dict()
        self.has_champion = has_champion

        if load:
            self.docs = pd.read_pickle(load_addr)
        else:
            self.docs = docs

        if length == "m":
            pass
        else:
            self.docs = self.docs.sample(int(length))
            # self.docs = self.docs.head(int(length))

        self.docs['id'] -= 1

        print("Tokenizing docs")
        start_time = time.time()
        self.docs["idx"] = self.docs.apply(
            lambda row: self.gen_doc_idx(row['content']), axis=1)
        print("--- %s seconds ---" % (time.time() - start_time))

        if 'topic' not in self.docs.columns:
            self.docs['topic'] = ''

        # Inferred Category
        self.docs['i_cat'] = ''

        # Sort columns to be compliant with the previous code
        self.docs = self.docs[['id', 'content', 'url', 'idx', 'topic', 'i_cat']]

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
        for id, _, _, idx, _, _ in self.docs.itertuples(index=False):
            for key, value in idx.items():
                inv_idx[key][id] = self.tf(value)
            self.doc_lengths[id] = self.gen_doc_length(idx)
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

    def word_cos_similarity(self, q_word: str, id: int):
        return self.tf_idf(q_word, self.docs['idx'][id]) / self.doc_lengths[id]

    def doc_tf_idf(self, term: str, id1: int, id2: int):
        return self.idf(term) * self.docs['idx'][id1][term] * self.docs['idx'][id2][term]

    def doc_cos_similarity(self, id1: int, id2: int):
        k1 = set(self.docs['idx'][id1].keys())
        k2 = set(self.docs['idx'][id2].keys())
        keys = k1 & k2
        return sum([self.doc_tf_idf(k, id1, id2) for k in keys]) / (self.doc_lengths[id1] * self.doc_lengths[id2]) + 1

    # Generate scores for each document given the query
    def gen_scores(self, q: str):
        scores = dict()
        q_words = self.doc_tokens(q, set)
        search_area = self.champions if self.has_champion else self.inv_idx
        for q_word in q_words:
            # If query word is not in champion list, then ignore it
            for champ_id in search_area.get(q_word, dict()).keys():
                scores[champ_id] = scores.get(
                    champ_id, 0) + self.word_cos_similarity(q_word, champ_id)

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


class Clustering:
    def __init__(self, p3: Processing, cluster_count: int = 2, p2: Processing = None):
        self.p3 = p3
        self.p2 = p2
        self.clusters = {}
        self.cluster_count = cluster_count
        self.cluster_distances = {}
        self.cluster_centers = {}

    # K-means clustering on a given data set in the form of dictionaries
    def k_means(self, max_iter=100):
        self.cluster_centers = self.p3.docs.sample(n=self.cluster_count)
        data = self.p3.docs['id']

        for _ in range(max_iter):
            # Calculate the distance between each point and each cluster center
            for i in range(len(data)):
                for j in range(self.cluster_count):
                    self.cluster_distances[j] = self.calculate_distance(
                        data[i], self.cluster_centers[j])
                # Find the smallest distance and assign the point to that cluster
                self.clusters[min(self.cluster_distances, key=self.cluster_distances.get)] = self.clusters.get(
                    min(self.cluster_distances, key=self.cluster_distances.get), [])
                self.clusters[min(self.cluster_distances,
                                  key=self.cluster_distances.get)].append(i)
                # Calculate the new cluster center
                for j in range(self.cluster_count):
                    self.cluster_centers[j] = self.calculate_center(
                        self.clusters[j])
            # Calculate the average distance between each point and its cluster center
            for i in range(len(data)):
                for j in range(self.cluster_count):
                    self.cluster_distances[j] = self.calculate_distance(
                        self.data[i], self.cluster_centers[j])

        # return self.clusters, self.cluster_distances, self.cluster_centers

    # Calculating the distance between a point and a cluster center
    def calculate_distance(self, point, center) -> float:
        return 1 / self.p3.doc_cos_similarity(point, center)

    # Finds the document with the least distance to the average of the cluster
    def calculate_center(self, cluster) -> str:
        acc = dict()
        for id in cluster:
            for word in self.p3.docs['idx'][id].keys():
                acc[word] = acc.get(word, 0) + \
                    self.p3.docs['idx'][id][word] / len(cluster)

        self.p3.docs.loc['tmp_cl'] = ['tmp_cl', '', '', acc]

        sim = dict()
        for id in cluster:
            sim[id] = self.calculate_distance(id, 'tmp_cl')

        self.p3.docs.drop('tmp_cl', inplace=True)
        return min(sim, key=sim.get)

    def search_docs(self, q: str, search_area: list):
        scores = dict()
        q_words = self.p3.doc_tokens(q, set)
        for center in search_area:
            for q_word in q_words:
                scores[center] = scores.get(
                    center, 0) + self.p3.word_cos_similarity(q_word, center)
        return scores

    def query_k_means(self, q, b=2):

        cluster_scores = self.search_docs(q, self.cluster_centers)

        relevant_centers = reverse_sorted_dict(cluster_scores)[:b]
        relevant_clusters = [self.clusters[self.cluster_centers.index(
            center)] for center in relevant_centers]
        relevant_docs = [self.p3.docs[id]
                         for clust in relevant_clusters for id in clust]

        scores = self.search_docs(q, relevant_docs)
        return scores

    # def query_knn(self, cat, q):

    # This function performs knn classification algorithm on the given data set.
    def knn_iteration(self, data: pd.DataFrame, query_id, k, choice_fn=lambda x: max(x, key=x.count)):
        neighbor_distances = dict()

        # 3. For each example in the data
        for id in data:
            # 3.1 Calculate the distance between the query example and the current
            # example from the data.
            distance = self.calculate_distance(id, query_id)

            # 3.2 Add the distance and the index of the example to an ordered collection
            neighbor_distances[id] = distance

        # 4. Sort the ordered collection of distances and indices from
        # smallest to largest (in ascending order) by the distances
        sorted_neighbor_distances = reverse_sorted_dict(neighbor_distances)

        # 5. Pick the first K entries from the sorted collection
        first_k_keys = list(sorted_neighbor_distances.keys())[:k]
        k_nearest_distances = {key: sorted_neighbor_distances[key] for key in first_k_keys}
        # k_nearest_distances = sorted_neighbor_distances[:k]

        # 6. Get the labels of the selected K entries
        k_nearest_labels = [self.p3.docs['topic'][i] for i in k_nearest_distances.keys()]

        # 7. If regression (choice_fn = mean), return the average of the K labels
        # 8. If classification (choice_fn = mode), return the mode of the K labels
        return choice_fn(k_nearest_labels)

    def knn_learning(self):
        k_fold = KFold(10, shuffle=True, random_state=1)
        k_scores = dict()
        k = 0
        for train_ids, test_ids in k_fold.split(self.p3.docs):
            k += 1
            train = self.p3.docs.iloc[train_ids]
            test = self.p3.docs.iloc[test_ids]
            for id in test['id']:
                # self.p3.docs.iloc[id]['i_cat'] = self.knn_iteration(train_ids, id, k=k)
                test['i_cat'][id] = self.knn_iteration(train['id'], id, k=k)
            k_scores[k] = len(
                train[train['topic'] == train['i_cat']]) / len(train)
            test['i_cat'] = ''

        self.k = max(k_scores, key=k_scores.get)

    def knn_classification(self):
        for id in self.p2.docs['id']:
            self.p2.docs['i_cat'][id] = self.knn_iteration(
                self.p3.docs['id'], id, self.k)
        self.p2.docs.to_excel("phase2_knn.xlsx")
        self.p2.save("phase2_knn.pickle")


def main():

    length = input(
        "Enter the number of docs you want to search through (starts from the 1st document), m for max \n")
    print("Search engine started, wait for initialization...")

    data1 = pd.read_excel("data/phase3/1.xlsx")
    data2 = pd.read_excel("data/phase3/2.xlsx")
    data3 = pd.read_excel("data/phase3/3.xlsx")

    data = pd.concat([data1, data2, data3])
    p3 = Processing(data, length=length, has_champion=False)

    data = pd.read_excel("data/phase3/data.xlsx")
    p2 = Processing(data, length=length, has_champion=False)

    c = Clustering(p3=p3, p2=p2)

    while(True):
        in_str = input("Enter your query, !q to exit \n")
        if in_str == "!q":
            break

        start_time = time.time()

        p3.gen_scores(in_str)
        k = min(5, len(p3.scores))
        print("The documents with the best scores are in this order (add +1 to the ids):")
        print(p3.best_k(k, heap_or_sort=True))
        print("Querying took:")
        print("--- %s seconds ---" % (time.time() - start_time))


def main_2():
    length = 100
    data1 = pd.read_excel("data/phase3/1.xlsx")
    data2 = pd.read_excel("data/phase3/2.xlsx")
    data3 = pd.read_excel("data/phase3/3.xlsx")

    data = pd.concat([data1, data2, data3])
    p3 = Processing(data, length=length, has_champion=False)

    data = pd.read_excel("data/phase2/data.xlsx")
    p2 = Processing(data, length=length, has_champion=False)

    c = Clustering(p3=p3, p2=p2)

    c.knn_learning()
    c.knn_classification()
    print(c.k)


if __name__ == "__main__":
    main_2()
