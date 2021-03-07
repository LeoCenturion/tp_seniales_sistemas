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

from database import Database, Connection, DB


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


def make_db_with_songs(song, connection=Connection):
    print("%s processing \n" % song)
    db = try_get_from_cache(connection, song)
    print("%s processed \n" % song)
    return db


def cache_path_from_song_path(song):
    return "/".join(song.split("/")[:-1] + ["cache"] + [song.split("/")[-1]])


def song_cached(song):
    try:
        f = open(cache_path_from_song_path(song), "rb")
        f.close()
        return True
    except FileNotFoundError:
        return False


def try_get_from_cache(connection, song):
    song_name = song.split("/")[-1].split(".")[0]
    if song_cached(song):
        return Database(Connection.from_disk(cache_path_from_song_path(song)), song_name)
    else:
        db = Database(connection(), song_name)
        save_song(db, song)
        db.set_path(cache_path_from_song_path(song))
        # db.flush()
        return db


def generate_db(path, db_name, make_connection):
    print("collecting songs")

    songs = get_song_paths(path)

    print("Songs found: %i" % len(songs))

    db = make_db(songs, db_name, make_connection)

    save_db(db, db_name, path)
    print("Finished")

    return db


def save_db(db, db_name, path):
    filename = path
    db.set_path(filename)
    db.set_name(db_name)
    db.flush()


def make_db(songs, db_name, make_connection=Connection):
    p = mp.Pool(4)
    dbs = p.map(make_db_with_songs, songs)
    p.close()
    # dbs = list(map(make_db_with_songs, songs))
    print('Merging db\'s')
    db = functools.reduce(lambda db1, db2: db1 + db2, dbs)
    return db


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
    db = DB(hash_nbits, n_entries, ID_nbits, np.zeros((2 ** hash_nbits, n_entries), dtype=np.uint32))

    for k in range(n_entries):
        file_path = files[k]
        # Muestro por consola el archivo que analizamos
        track, fs = sf.read(file_path)
        # Generamos la huella acustica H
        sound_print = generate_footprint(track, fs)
        #     print(sound_print)
        # Guardamos la huella en la DB. A esta cancion se asigna el
        # identificador numerico k, el cual se guarda en la base de datos
        # La matriz H debera ser una matriz con elementos binarios de
        # hash_nbits filas
        db.save_footprint(k + 1, sound_print)
    db.set_path(path + "/main_db")
    db.flush()

if __name__ == '__main__':
    path = "/home/leonardo/Documents/seniales/tp/10songs"
    generate_db2(path)
