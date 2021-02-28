import os
import unittest

from database import Connection, Database
from generatedb import generate_db, generate_footprint
import soundfile as sf

class TestDatabaseWithFilesystem(unittest.TestCase):
    path = "/home/leonardo/Documents/seniales/tp/test_songs"

    def test_generate_db_creates_file(self):
        make_connection = Connection
        db_name = "test_db"
        db = generate_db(self.path, db_name, make_connection)
        db.flush()

        f = open(self.path + '/' + db_name)
        with f:
            self.assertIsNotNone(f)
        os.remove(self.path + '/' + db_name)

    def test_read_db_from_disk_and_query_song(self):
        db_name = "test_db_permanent"
        conn = Connection.from_disk(self.path + '/' + db_name)
        db = Database(conn, db_name)
        song = self.path + '/cut_tracks/test_cut_Blur.ogg'
        data, samplerate = sf.read(song)
        footprint = generate_footprint(data, samplerate)

        actual = list(map(lambda result: result[0], db.query(footprint)))
        expected = ["Blur - Song 2-SSbBvKaM6sk.ogg", "Dr. Dre ft. Snoop Dogg, Kurupt, Nate Dogg - The Next Episode.ogg"]
        self.assertEqual(actual, expected)