import csv
import pickle
import random
from random import randint

import more_itertools as mi
import numpy as np
from dec2bin import dec2bin


class DB:
    def __init__(self, hash_nbits, n_entries, ID_nbits, table):
        # Tamanio en memoria para hash de 20 bits: 4MB x N_columnas
        self.hash_nbits = hash_nbits  # Cantidad de bits de los hashes
        self.n_entries = n_entries  # Cantidad de columnas de la tabla
        self.ID_nbits = ID_nbits  # Cantidad de bits para ID numerico de la cancion
        self.table = table
        self.path = "."

    def save_footprint(self, ID, sound_print):
        # Verificamos que HUELLA sea una matriz binaria
        #     if not all(x == 1 or x == 0 for x in sound_print):
        for row in sound_print:
            for elem in row:
                if elem != 0 and elem != 1:
                    print('Error: La matriz HUELLA contiene elementos no binarios')
                    return

        # Verificamos que HUELLA tenga hash_nbits filas
        if not np.shape(sound_print)[0] == self.hash_nbits:
            print('Error: La matriz HUELLA tiene ', np.shape(sound_print)[0], ' filas en lugar de ', self.hash_nbits)
            return

        # Verificamos que ID sea un entero mayor o igual a 1
        if ID < 1 or not isinstance(ID, int):
            print('Error: El identificador ID debe ser un entero mayor o igual a 1')
            return

        # Generamos el elemento VAL a guardar, concatenando los bits del ID con los
        # bits del tiempo de cada frame
        frames_nbits = 32 - self.ID_nbits
        trans_print = np.transpose(sound_print)
        len_print = len(trans_print)

        for i in range(len_print):
            row = trans_print[i][::-1]
            hash_val = ''
            for elem in row:
                hash_val += str(int(elem))
            hash_row = int(hash_val, 2)

            frame = str(dec2bin(i))
            # dejo el largo del frame en 20 bits
            while (len(frame) < self.hash_nbits):
                frame = '0' + frame

            ID_str = str(dec2bin(ID))
            # dejo el largo del id en 12 bits
            while (len(ID_str) < self.ID_nbits):
                ID_str = '0' + ID_str

            #         print(hash_row, int(frame, 2), int(ID_str, 2))
            val = frame + ID_str

            # genero y guardo el dato de tipo entero
            full = True
            for j in range(self.n_entries):
                if self.table[hash_row][j] == 0:
                    self.table[hash_row][j] = int(val, 2)
                    full = False
                    break
            if full:
                self.table[hash_row][randint(0, self.n_entries - 1)] = int(val, 2)

    def query(self, sound_print):
        trans_print = np.transpose(sound_print)
        len_print = len(trans_print)
        hash_rows = []
        for i in range(len_print):
            row = trans_print[i][::-1]
            val = ''
            for elem in row:
                val += str(int(elem))
            hash_rows.append(int(val, 2))
        #     return hash_rows
        # Extraemos los elementos de la tabla que corresponden a los hashes dados
        # val = frame | id
        vals = []
        i = 0
        for row in hash_rows:
            vals.append([])
            j = 0
            while (j < self.n_entries and self.table[row][j] != 0):
                vals[i].append(self.table[row][j])
                j += 1
            i += 1

        if not any(elem for elem in vals):
            # Si los elementos fueron todos nulos, devuelve cero
            ID = [0]
            ID_matches = [0]
            return ID, ID_matches

        # separo de cada elemento de vals el ID numerico y su frame
        ID1 = []
        frames = []
        for row in vals:
            for val in row:
                ID1.append(np.mod(val, 2 ** self.ID_nbits))
                frames.append(np.floor(val / 2 ** self.ID_nbits))
        ID1 = np.array(ID1)
        frames = np.array(frames)

        # Filtramos por tiempo: para cada ID, se cuenta el numero de coincidencias
        # dentro del intervalo de duracion temporal de la huella (i.e. frame_span)
        frame_span = np.shape(sound_print)[1]  # intervalo de duracion de la huella
        IDs = np.array(list(set(np.sort(ID1))))  # ordeno los ids en orden creciente
        ID_matches = np.zeros(np.size(IDs))
        for i in range(len(IDs)):
            # devuelve un array con los i-ésimos elementos de frames en donde haya un match de ids[i] en id1
            #     frame_aux = frames(ID1 == IDs[i])
            frame_aux = []
            for j in range(len(ID1)):
                if ID1[j] == IDs[i]:
                    frame_aux.append(frames[j])
            frame_aux = np.array(frame_aux)
            len_frame = len(frame_aux)
            matches = 0
            for j in range(len_frame):
                # match_aux: numero de matches en intervalo frame_span
                aux1 = []
                aux2 = []
                for k in range(len_frame):
                    aux1.append(1 if frame_aux[k] >= frame_aux[j] else 0)
                    aux2.append(1 if frame_aux[k] <= frame_aux[j] + frame_span else 0)
                aux1 = np.array(aux1)
                aux2 = np.array(aux2)
                match_aux = np.count_nonzero(aux1 & aux2)
                if match_aux > matches:
                    matches = match_aux
            ID_matches[i] = matches

        # Ordeno ID y MATCHES por orden descendente
        #     print('M', ID_matches)
        #     print('I', IDs)
        idx = np.argsort(ID_matches)
        idx = idx[::-1]
        #     print('N', idx)
        ID_matches[::-1].sort()
        #     print(ID_matches)
        _IDs = []
        for index in idx:
            _IDs.append(IDs[index])
        IDs = _IDs
        #     print(IDs)
        # Me quedo con los primeros Nmax elementos
        Nmax = 5
        N = min(Nmax, np.size(IDs))
        ID_matches = ID_matches[:N]
        IDs = IDs[:N]

        return IDs, ID_matches

    @staticmethod
    def merge_tables(table, other):
        new_table = np.zeros(np.shape(table))
        for i in range(len(table)):
            number_of_columns = len(new_table[0])
            merged_list = list(filter(lambda n: n != 0, table[i])) + list(filter(lambda n: n != 0, other[i]))
            if len(merged_list) > number_of_columns:
                new_table[i] = random.sample(np.array(merged_list), number_of_columns)
            else:
                empty_spaces = len(merged_list) - number_of_columns
                new_table[i] = np.array(mi.padded(merged_list, 0.0, empty_spaces))
        return new_table

    def __add__(self, other):
        return DB(self.hash_nbits, self.n_entries, self.ID_nbits, self.merge_tables(self.table, other.table))

    def set_path(self, path):
        self.path = path

    def set_name(self, name):
        self.name = name

    def flush(self):
        np.savetxt(self.path, self.table, delimiter=",")

    @staticmethod
    def from_disk(hash_nbits, n_entries, ID_nbits, path):
        return DB(hash_nbits, n_entries, ID_nbits, np.loadtxt(path, delimiter=","))


def count_close_matches(matches, frame_count, match):
    match_frame = match[1]
    song_id = match[0]

    def is_in_range(other_match_frame):
        return match_frame - frame_count / 2 <= other_match_frame <= match_frame + frame_count / 2

    return mi.ilen(filter(lambda other_match: other_match[0] == song_id and is_in_range(other_match[1]), matches))


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
                                    map(lambda frame: self.connection.query_by_footprint(frame), footprint.T))),
                         key=lambda kv: kv[0])
        grouped = self.group_in_frame_windows(matches, frame_count)
        # appearances_count = map(lambda id_list: (id_list[0], mi.ilen(id_list[1])),
        #                         i.groupby(matches, key=lambda kv: kv[0]))
        return mi.take(5, (sorted(grouped, key=lambda key_count: key_count[1], reverse=True)))

    def __add__(self, other):
        return Database(self.connection + other.connection, self.name)

    def flush(self):
        self.connection.flush(self.name)

    @staticmethod
    def group_in_frame_windows(matches, frame_count):
        """
        Group matches into windows of size frame_count
        :param matches: list of tuples of type [(song_id, frame_number)]
        :param frame_count: integer, number of frames
        :return: list of tuples like [(song_1, 3),  (song_2, 7)]
        """
        count_by_window = map(lambda match: (match[0], count_close_matches(matches, frame_count, match)), matches)
        grouped_by_song = mi.groupby_transform(count_by_window,
                                               keyfunc=lambda song_window_count: song_window_count[0],
                                               valuefunc=lambda song_window_count: song_window_count[1],
                                               reducefunc=max)
        return grouped_by_song


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

    def save(self, id, spectrum, frame_index):
        hash = self.hash_spectrum(spectrum)
        if hash in self.inverted_index:
            self.inverted_index[hash] += [(id, frame_index)]
        else:
            self.inverted_index[hash] = [(id, frame_index)]

    @staticmethod
    def hash_spectrum(spec):
        # bits = len(spec)
        hash_val = ''
        for elem in spec:
            hash_val += str(int(elem))
        return int(hash_val, 2)
        # return sum([(bits-i)**spec[i] for i in range(bits)])
        # hash = hashlib.blake2b(spec.tobytes(), digest_size=20)
        # for dim in spec.shape:
        #     hash.update(dim.to_bytes(4, byteorder='big'))
        # return hash.digest()

    def query_by_footprint(self, fp):
        return self.inverted_index.get(self.hash_spectrum(fp), [])

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
