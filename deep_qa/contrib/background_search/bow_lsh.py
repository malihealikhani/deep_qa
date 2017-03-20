'''
The retrieval method encodes both the background and the query sentences by averaging their, and does retrieval using
a Locality Sensitive Hash (LSH). We use ScikitLearn's LSH.
'''
import sys
import os
import argparse
import gzip
import numpy
import sklearn
import pickle

import spacy
from sklearn.neighbors import LSHForest


class BOWLSH:
    def __init__(self, serialization_prefix='lsh'):
        self.embeddings = {}
        self.lsh = None
        self.embedding_dim = None
        self.serialization_prefix = serialization_prefix
        # We'll keep track of the min and max vector values so that we can sample values from this
        # interval for UNK if needed.
        self.vector_max = -float("inf")
        self.vector_min = float("inf")
        self.en_nlp = spacy.load('en')
        self.indexed_background = {}  # index -> background sentence

    def read_embeddings_file(self, embeddings_file: str):
        with gzip.open(embeddings_file, 'rb') as embeddings_file:
            for line in embeddings_file:
                fields = line.decode('utf-8').strip().split(' ')
                self.embedding_dim = len(fields) - 1
                word = fields[0]
                vector = numpy.asarray(fields[1:], dtype='float32')
                vector_min = min(vector)
                vector_max = max(vector)
                if vector_min < self.vector_min:
                    self.vector_min = vector_min
                if vector_max > self.vector_max:
                    self.vector_max = vector_max
                self.embeddings[word] = vector

    def load_model(self):
        pickled_embeddings_file = "%s/embeddings.pkl" % self.serialization_prefix
        pickled_lsh_file = "%s/lsh.pkl" % self.serialization_prefix
        self.embeddings = pickle.load(open(pickled_embeddings_file, 'rb'))
        indexed_background_file = open("%s/background.tsv" % self.serialization_prefix, "r")
        for vector in self.embeddings.values():
            if self.embedding_dim is None:
                self.embedding_dim = len(vector)
                vector_min = min(vector)
                vector_max = max(vector)
                if vector_min < self.vector_min:
                    self.vector_min = vector_min
                if vector_max > self.vector_max:
                    self.vector_max = vector_max 
        self.lsh = pickle.load(open(pickled_lsh_file, 'rb'))
        for line in indexed_background_file:
            parts = line.strip().split('\t')
            self.indexed_background[int(parts[0])] = parts[1]

    def save_model(self):
        if not os.path.exists(self.serialization_prefix):
            os.makedirs(self.serialization_prefix)
        pickled_embeddings_file = "%s/embeddings.pkl" % self.serialization_prefix
        pickled_lsh_file = "%s/lsh.pkl" % self.serialization_prefix
        indexed_background_file = open("%s/background.tsv" % self.serialization_prefix, "w")
        pickle.dump(self.embeddings, open(pickled_embeddings_file, 'wb'))
        pickle.dump(self.lsh, open(pickled_lsh_file, 'wb'))
        for index, sentence in self.indexed_background.items():
            sentence = sentence.replace('\t', ' ')  # Sanitizing sentences before making a tsv.
            print("%d\t%s" % (index, sentence), file=indexed_background_file)

    def get_word_vector(self, word, random_for_unk=False):
        if word in self.embeddings:
            return self.embeddings[word]
        else:
            # If this is for the background data, we'd want to make new vectors (uniformly sampling
            # from the range (vector_min, vector_max)). If this is for the queries, we'll return a zero vector
            # for UNK because this word doesn't exist in the background data either.
            if random_for_unk:
                vector = numpy.random.uniform(low=self.vector_min, high=self.vector_max, size=(self.embedding_dim,))
                self.embeddings[word] = vector
            else:
                vector = numpy.zeros((self.embedding_dim,))
            return vector

    def encode_sentence(self, sentence: str, for_background=False):
        words = [str(w.lower_) for w in self.en_nlp.tokenizer(sentence)]
        return numpy.mean(numpy.asarray([self.get_word_vector(word, for_background) for word in words]), axis=0)

    def read_background(self, background_file):
        # Read background file and add to indexed_background.
        index = 0
        for sentence in gzip.open(background_file, mode="r"):
            sentence = sentence.decode('utf-8').strip()
            if sentence != '':
                self.indexed_background[index] = sentence
                index += 1

    def fit_lsh(self):
        self.lsh = LSHForest(random_state=12345)
        train_data = [self.encode_sentence(self.indexed_background[i],
                                           True) for i in range(len(self.indexed_background))]
        self.lsh.fit(train_data)

    def print_neighbors(self, sentences_file, outfile, num_neighbors=50):
        sentences = []
        indices = []
        for line in open(sentences_file):
            parts = line.strip().split('\t')
            indices.append(parts[0])
            sentences.append(parts[1])
        test_data = [self.encode_sentence(sentence) for sentence in sentences]
        _, all_neighbor_indices = self.lsh.kneighbors(test_data, n_neighbors=num_neighbors)
        with open(outfile, "w") as outfile:
            for i, sentence_neighbor_indices in zip(indices, all_neighbor_indices):
                print("%s\t%s" % (i, "\t".join([self.indexed_background[j] for j in sentence_neighbor_indices])),
                      file=outfile)
            outfile.close()


def main():
    argparser = argparse.ArgumentParser(description="Build a Locality Sensitive Hash and use it for retrieval.")
    argparser.add_argument("--embeddings_file", type=str, help="Gzipped file containing pretrained embeddings \
                           (required for fitting)")
    argparser.add_argument("--background_corpus", type=str, help="Gzipped sentences file (required for fitting)")
    argparser.add_argument("--questions_file", type=str, help="TSV file with indices in the first column \
                           and question in the second (required for retrieval)")
    argparser.add_argument("--retrieved_output", type=str, help="Location where retrieved sentences will be written \
                           (required for retrieval)")
    argparser.add_argument("--serialization_prefix", type=str, help="Loacation where the lsh will be serialized \
                           (default: lsh/)", default="lsh")
    args = argparser.parse_args()
    bow_lsh = BOWLSH(args.serialization_prefix)
    also_train = False
    if args.embeddings_file is not None and args.background_corpus is not None:
        print("Attempting to fit LSH", file=sys.stderr)
        also_train = True
        print("Reading embeddings", file=sys.stderr)
        bow_lsh.read_embeddings_file(args.embeddings_file)
        print("Reading background", file=sys.stderr)
        bow_lsh.read_background(args.background_corpus)
        print("Fitting LSH", file=sys.stderr)
        bow_lsh.fit_lsh()
        print("Saving model", file=sys.stderr)
        bow_lsh.save_model()
    if args.questions_file is not None and args.retrieved_output is not None:
        print("Attempting to retrieve", file=sys.stderr)
        if not also_train:
            print("Attempting to load fitted LSH", file=sys.stderr)
            bow_lsh.load_model()
        bow_lsh.print_neighbors(args.questions_file, args.retrieved_output)

if __name__== '__main__':
    main()
