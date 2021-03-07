import  random
from random import randint

import more_itertools as mi
import numpy as np
from dec2bin import dec2bin
import functools
from os import listdir
from os.path import isfile, join
import multiprocessing as mp
import matplotlib as mplt
import more_itertools as mi
import numpy as np
import pandas as pd
import soundfile as sf
from matplotlib import mlab
from scipy import signal

class Database:
    def __init__(self, hash_nbits, n_entries, ID_nbits, table):
        self.hash_nbits = hash_nbits
        self.n_entries = n_entries
        self.ID_nbits = ID_nbits
        self.table = table
        self.path = "."

    def save_footprint(self, ID, sound_print):
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
        return Database(self.hash_nbits, self.n_entries, self.ID_nbits, self.merge_tables(self.table, other.table))

    def set_path(self, path):
        self.path = path

    def set_name(self, name):
        self.name = name

    def flush(self):
        np.savetxt(self.path, self.table, delimiter=",")

    @staticmethod
    def from_disk(hash_nbits, n_entries, ID_nbits, path):
        return Database(hash_nbits, n_entries, ID_nbits, np.loadtxt(path, delimiter=","))


def stereo2mono(stereo):
    return (stereo[:, 0] + stereo[:, 1]) / 2


def generate_footprint(data, fs):
    # mono = stereo2mono(data)
    spec, freq, t = generate_specgram(fs, data)

    buckets = generate_buckets()

    new_spec = remap_frequencies_for_specgram(spec, freq, buckets)

    return generate_h(new_spec)


def remap_frequencies_for_specgram(spec, freq, buckets):
    return np.array(list(map(lambda s: np.array(remap_frequencies(s, freq, buckets)), spec.T))).T


def generate_h(e):
    h = np.zeros((len(e) - 1, len(e[0])))
    for row_idx in range(1, len(e) - 1):
        for col_idx in range(1, len(e[row_idx]) - 1):
            h[row_idx][col_idx] = int(
                e[row_idx + 1][col_idx] - e[row_idx][col_idx] > e[row_idx + 1][col_idx - 1] - e[row_idx][
                    col_idx - 1])
    return h


def find_new_freq_band(f, bands):
    return list(filter(lambda band: band[0] <= f <= band[1], bands))[0][0]


def remap_frequencies(spectrum, freq, buckets):
    freq_to_energy_list = zip(spectrum ** 2, freq)
    energy_band_new_relation = map(
        lambda energy_freq: (energy_freq[0], find_new_freq_band(energy_freq[1], buckets)), freq_to_energy_list)
    grouped_by_frequencies = mi.groupby_transform(energy_band_new_relation, lambda e_f: e_f[1], lambda e_f: e_f[0])
    frequency_energy_sum = map(lambda kv: (kv[0], sum(kv[1])), grouped_by_frequencies)
    return list(mi.unzip(frequency_energy_sum)[1])[1:-1]


def generate_buckets():
    bands = np.concatenate((np.array([0]), np.geomspace(300, 2000, num=22), np.array([np.inf])), axis=None)

    return list(mi.pairwise(bands))


def generate_specgram(samplerate, mono):
    # df = pd.DataFrame(mono)
    # df['t'] = df.index / samplerate
    # df['avg'] = df[0]
    fs = samplerate
    nyq = fs / 2
    length = filter_order = 87
    f_high = 5512 / 2
    window = "hamming"
    b = signal.firwin(length, cutoff=f_high / nyq, window=window, pass_zero='lowpass')
    num = b
    dec = np.zeros(len(b))
    dec[0] = 1
    decimation = 8
    n = 2 ** 12
    dlti = signal.dlti(num, dec)
    mono_dec = signal.decimate(mono, 8, ftype=dlti, n=filter_order)
    fxx, txx, sxx = signal.spectrogram(mono_dec, fs=fs / decimation, window=window, nperseg=n,
                                       noverlap=np.round(n * .9), mode='magnitude')
    # op_1 = signal.lfilter(b, 1, df['avg'])
    # df['avg_filtered'] = op_1
    # compression_factor = 8
    # compress_sample_df = lambda df, compression_factor: df[df.index % compression_factor == 0]
    # compressed_df = compress_sample_df(df[['t', 'avg_filtered']], compression_factor).reset_index(inplace=False)
    # # compressed_df['avg_filtered_fft'] = np.fft.fft(compressed_df['avg_filtered'])
    # number_of_points = lambda time_mils, samplerate: int((time_mils / 1000) * samplerate)
    # # decimated_signal = compressed_df['avg_filtered']
    # smpl_rate = samplerate / 8
    # spec, freq, t = mlab.specgram(compressed_df['avg_filtered'], Fs=smpl_rate, NFFT=number_of_points(100, smpl_rate),
    #                               window=mplt.mlab.window_hanning)
    # return spec, freq, t
    return sxx, fxx, txx


def get_song_paths(path):
    filenames = [f for f in listdir(path) if isfile(join(path, f))]
    songs = [path + '/' + name for name in filenames if name.endswith(".ogg")]
    return songs


def save_song(db, song):
    data, samplerate = sf.read(song)
    footprint = generate_footprint(data, samplerate)
    song_id = song.split('/')[-1]
    db.save_footprint(song_id, footprint)


def generate_db2(path):
    # Incializo tabla hash
    files = get_song_paths(path)
    hash_nbits = 20  # Cantidad de bits de los hashes
    n_entries = len(files)  # Cantidad de columnas de la tabla; cant canciones
    ID_nbits = 12  # Cantidad de bits para ID numerico de la cancion
    db = Database(hash_nbits, n_entries, ID_nbits, np.zeros((2 ** hash_nbits, n_entries), dtype=np.uint32))

    for k in range(n_entries):
        file_path = files[k]
        # Muestro por consola el archivo que analizamos
        track, fs = sf.read(file_path)
        # Generamos la huella acustica H
        sound_print = generate_footprint(stereo2mono(track), fs)
        #     print(sound_print)
        # Guardamos la huella en la Database. A esta cancion se asigna el
        # identificador numerico k, el cual se guarda en la base de datos
        # La matriz H debera ser una matriz con elementos binarios de
        # hash_nbits filas
        db.save_footprint(k + 1, sound_print)
    db.set_path(path + "/main_db")
    db.flush()


def run_all2():
    files = get_song_paths("./40songs/")
    hash_nbits = 20  # Cantidad de bits de los hashes
    n_entries = len(files)  # Cantidad de columnas de la tabla; cant canciones
    ID_nbits = 12  # Cantidad de bits para ID numerico de la cancion
    # Tamanio en memoria para hash de 20 bits: 4MB x N_columnas
    db = Database.from_disk(hash_nbits, n_entries, ID_nbits, "/home/leonardo/Documents/seniales/tp/40songs/main_db")

    T = [5, 10, 20]
    N = 50
    ids = []
    matches = []
    songs = []
    last = 0
    for time in T:
        for i in range(N):
            #         now = datetime.now()
            #         timestamp = datetime.timestamp(now)
            #         print(timestamp - last)
            #         last = timestamp
            #         print(i)
            print(time, i)
            n_song = random.randint(0, len(files) - 1)
            file = files[n_song]
            track, fs = sf.read(file)
            start = random.randint(0, len(track) - fs * time - 1)
            end = start + fs * time
            sound_print = generate_footprint(stereo2mono(track[start:end]), fs)
            id, match = db.query(sound_print)
            ids.append(id)
            matches.append(match)
            songs.append(n_song + 1)

    count = 0
    for i in range(50):
        if songs[i] == ids[i][0]:
            count += 1
    five_sec = count
    print('5s hits:', five_sec, 'out of 50')
    print(str(five_sec * 100 / 50) + '%')
    count = 0
    for i in range(50, 100):
        if songs[i] == ids[i][0]:
            count += 1
    ten_sec = count
    print('10s hits:', ten_sec, 'out of 50')
    print(str(ten_sec * 100 / 50) + '%')
    count = 0
    for i in range(100, 150):
        if songs[i] == ids[i][0]:
            count += 1
    twenty_sec = count
    print('20s hits:', twenty_sec, 'out of 50')
    print(str(twenty_sec * 100 / 50) + '%')
    total = five_sec + ten_sec + twenty_sec
    first = np.round(total * 100 / 150, 2)
    print('Total hits:', total, 'out of 150')
    print(str(first) + '%')

    def add_noise(track, SNR):
        noise = np.random.randn(1, len(track))
        noise = np.array(noise[0])

        variance = np.var(track)
        alfa = np.sqrt(variance / (10 ** (SNR / 10)))
        norm_noise = noise * alfa ** 2

        return track + norm_noise

    T = [5, 10, 20]
    SNR = [0, 10, 20]
    N = 50
    ids = []
    matches = []
    songs = []
    i = 0
    for time in T:
        ids.append([])
        matches.append([])
        songs.append([])
        for j in range(len(SNR)):
            ids[i].append([])
            matches[i].append([])
            songs[i].append([])
            for k in range(N):
                #             print(k)
                print(time, j, k)
                n_song = random.randint(0, len(files) - 1)
                file = files[n_song]
                track, fs = sf.read(file)
                start = random.randint(0, len(track) - fs * time - 1)
                end = start + fs * time
                track_chunk = stereo2mono(track[start:end])
                track_w_noise = add_noise(track_chunk, SNR[j])

                sound_print = generate_footprint(track_w_noise, fs)
                id, match = db.query(sound_print)

                ids[i][j].append(id)
                matches[i][j].append(match)
                songs[i][j].append(n_song + 1)
        i += 1

    total = 0
    for i in range(len(T)):
        for j in range(len(SNR)):
            count = 0
            for k in range(N):
                if songs[i][j][k] == ids[i][j][k][0]:
                    count += 1
            print('Time:', T[i], 'secs.', 'SNR:', SNR[j])
            print('Hits:', count, 'out of 50')
            print('Porcentage:', str(count * 100 / 50) + '%')
            print('')
            total += count
    print('Hits:', total, 'out of 450')
    print('Porcentage:', str(total * 100 / 450) + '%')


if __name__ == '__main__':
    run_all2()