import csv
import pickle
from collections import Counter
import more_itertools as mi
import itertools as i
import numpy as np


class Database:
    def __init__(self, connection, name):
        self.name = name
        self.connection = connection
        self.hash_nbits = 20
        self.n_entries = 20
        self.id_nbits = 12
        self.tabla = np.zeros((2 ** self.hash_nbits, self.n_entries), dtype=int)

    def set_name(self, name):
        self.name = name

    def set_path(self, path):
        self.connection.path = path

    def save_footprint(self, id, footprint):
        fp = footprint.T
        for frame_idx in range(len(fp)):
            self.connection.save(id, fp[frame_idx], frame_idx)

    def query(self, footprint):
        frame_count = len(footprint[0])
        matches = sorted(filter(lambda id_frame: id_frame[1] <= frame_count,
                                mi.flatten(
                                    map(lambda frame: self.connection.query_by_footprint(list(frame)), footprint.T))),
                         key=lambda kv: kv[0])
        appearances_count = map(lambda id_list: (id_list[0], mi.ilen(id_list[1])),
                                i.groupby(matches, key=lambda kv: kv[0]))
        return mi.take(5, (sorted(appearances_count, key=lambda key_count: key_count[1], reverse=True)))

    def __add__(self, other):
        return Database(self.connection + other.connection, self.name)

    def flush(self):
        self.connection.flush(self.name)


class Connection:
    def __init__(self, path="."):
        self.path = path
        self.inverted_index = {}
        self.data = {}

    @staticmethod
    def from_disk(path):
        f = open(path, 'rb')
        conn = pickle.load(f)
        f.close()
        return conn

    def save(self, id, footprint, frame_index):
        hash = self.hash_footprint(footprint)
        if hash in self.inverted_index:
            self.inverted_index[hash] += [(id, frame_index)]
        else:
            self.inverted_index[hash] = [(id, frame_index)]

        if id in self.data:
            self.data[id] += [footprint]
        else:
            self.data[id] = [footprint]

    @staticmethod
    def hash_footprint(footprint):
        return sum(footprint)

    def query_by_footprint(self, fp):
        return self.inverted_index.get(self.hash_footprint(fp), [])

    @staticmethod
    def merge_dicts(dict1, dict2):
        d = {}
        d.update(dict1)
        for k, v in dict2.items():
            if k in d:
                d[k] += v
            else:
                d[k] = v
        return d

    def __add__(self, other):
        new_conn = Connection()
        new_conn.data = self.merge_dicts(self.data, other.data)
        new_conn.inverted_index = self.merge_dicts(self.inverted_index, other.inverted_index)
        return new_conn

    def flush(self, db_name):
        filename = self.path + '/' + db_name
        outfile = open(filename, 'wb')
        with outfile:
            print("Writing to disk")
            pickle.dump(self, outfile)
            outfile.close()
