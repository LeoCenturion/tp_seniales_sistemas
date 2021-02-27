import unittest

import numpy as np

from database import Database


class MockConnection:
    def __init__(self):
        self.inverted_index = {}
        self.data = {}

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

    def hash_footprint(self, footprint):
        return sum(footprint)

    def query_by_id(self, id):
        return self.data.get(id)

    def query_by_footprint(self, fp):
        return self.inverted_index.get(self.hash_footprint(fp), [])


class TestDatabase(unittest.TestCase):
    def test_create_database(self):
        connection = MockConnection()
        db = Database(connection)
        self.assertEqual(db.hash_nbits, 20)
        self.assertEqual(db.n_entries, 20)
        self.assertEqual(db.id_nbits, 12)
        self.assertEqual(db.tabla.size, 20 * (2 ** 20))

    def test_save_song_footprint_then_song_was_written(self):
        connection = MockConnection()
        db = Database(connection)
        id = "song"
        h = np.zeros((21, 300))

        db.save_footprint(id, h)

        self.assertEqual(len(connection.data), 1)

        id2 = "song2"
        h = np.ones((21, 300))
        db.save_footprint(id2, h)

        self.assertEqual(len(connection.data), 2)

    def test_save_song_then_can_query_song(self):
        connection = MockConnection()
        db = Database(connection)
        id = "song"
        h = np.zeros((21, 300))
        h[0] = [i for i in range(len(h[0]))]  # ensure no collisions

        db.save_footprint(id, h)

        self.assertEqual(db.query(h), [('song', 300)])  # matches all

        id2 = "song2"
        h2 = np.zeros((21, 300))
        h2[0] = [i for i in range(len(h2[0]))]  # ensure no collisions

        db.save_footprint(id2, h2)

        self.assertEqual(db.query(h2), [('song', 300), ('song2', 300)])  # matches all

    def test_save_song_then_can_query_with_reduced_footprint(self):
        connection = MockConnection()
        db = Database(connection)
        id = "song"
        h = np.zeros((21, 300))
        h[0] = [i for i in range(len(h[0]))]
        reduced_footprint = (h.T[50:100]).T

        db.save_footprint(id, h)

        self.assertEqual(db.query(reduced_footprint), [('song', 1)])
        # because only one matches with the hash value AND the frame length restriction


if __name__ == '__main__':
    unittest.main()
