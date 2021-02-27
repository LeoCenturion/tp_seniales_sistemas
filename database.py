import csv
from collections import Counter
import more_itertools as mi
import itertools as i
import numpy as np


class Database:
    def __init__(self, connection):
        self.connection = connection
        self.hash_nbits = 20
        self.n_entries = 20
        self.id_nbits = 12
        self.tabla = np.zeros((2 ** self.hash_nbits, self.n_entries), dtype=int)

    def save_footprint(self, id, footprint):
        fp = footprint.T
        for frame_idx in range(len(fp)):
            self.connection.save(id, fp[frame_idx], frame_idx)

    def query(self, footprint):

        frame_count = len(footprint[0])
        matches = sorted(filter(lambda id_frame: id_frame[1] <= frame_count,
                         mi.flatten(map(lambda frame: self.connection.query_by_footprint(list(frame)), footprint.T))),
                         key=lambda kv: kv[0])
        appearances_count = map(lambda id_list: (id_list[0], mi.ilen(id_list[1])),
                               i.groupby(matches, key=lambda kv: kv[0]))
        return mi.take(5, (sorted(appearances_count, key=lambda key_count: key_count[1], reverse=True)))
