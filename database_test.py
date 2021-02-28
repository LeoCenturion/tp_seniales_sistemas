import unittest

import numpy as np

from database import Database, Connection


class TestDatabase(unittest.TestCase):
    def test_create_database(self):
        connection = Connection()
        db = Database(connection, "test_db")
        self.assertEqual(db.hash_nbits, 20)
        self.assertEqual(db.n_entries, 20)
        self.assertEqual(db.id_nbits, 12)
        self.assertEqual(db.tabla.size, 20 * (2 ** 20))

    def test_save_song_footprint_then_song_was_written(self):
        connection = Connection()
        db = Database(connection, "test_db")
        id = "song"
        h = np.zeros((21, 300))

        db.save_footprint(id, h)

        self.assertEqual(len(connection.data), 1)

        id2 = "song2"
        h = np.ones((21, 300))
        db.save_footprint(id2, h)

        self.assertEqual(len(connection.data), 2)

    def test_save_song_then_can_query_song(self):
        connection = Connection()
        db = Database(connection, "test_db")
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
        connection = Connection()
        db = Database(connection, "test_db")
        id = "song"
        h = np.zeros((21, 300))
        h[0] = [i for i in range(len(h[0]))]
        reduced_footprint = (h.T[50:100]).T

        db.save_footprint(id, h)

        self.assertEqual(db.query(reduced_footprint), [('song', 1)])
        # because only one matches with the hash value AND the frame length restriction


    def test_merge_databases(self):

        connection = Connection()
        db1 = Database(connection, "test_db")
        id1 = "song"
        h1 = np.zeros((21, 300))
        h1[0] = [i for i in range(len(h1[0]))]

        db1.save_footprint(id, h1)

        db2 = Database(connection, "test_db")
        id2 = "song"
        h2 = np.zeros((21, 300))
        h2[0] = [i for i in range(len(h2[0]))]

        db2.save_footprint(id2, h2)

        db = Database(connection, "test_db")
        db.save_footprint(id1, h1)
        db.save_footprint(id2, h2)

        self.assertDictEqual((db1 + db2).connection.data, db.connection.data)
        self.assertDictEqual((db1 + db2).connection.inverted_index, db.connection.inverted_index)



if __name__ == '__main__':
    unittest.main()
