import functools
import multiprocessing as mp
import random
from os import listdir
from os.path import isfile, join

import matplotlib as mplt
import more_itertools as mi
import numpy as np
import pandas as pd
import soundfile as sf
from matplotlib import mlab
from scipy import signal

from database import Database, Connection


def generate_footprint(song_array, samplerate):
    df = pd.DataFrame(song_array)
    df['t'] = df.index / samplerate
    df['avg'] = df[[0, 1]].apply(np.average, axis=1)
    df['fft'] = np.fft.fft(df['avg'])

    fs = samplerate
    nyq = fs / 2
    length = 80
    f_high = 5512
    b = signal.firwin(length, cutoff=f_high / nyq, window="hann")
    op_1 = signal.lfilter(b, 1, df['avg'])
    fft = np.fft.fft(op_1)
    freq = np.fft.fftfreq(df['t'].shape[-1], 1 / samplerate)
    df['avg_filtered'] = op_1
    df['avg_filtered_fft'] = fft
    compression_factor = 8
    compress_sample_df = lambda df, compression_factor: df[df.index % compression_factor == 0]

    compressed_df = compress_sample_df(df[['t', 'avg_filtered']], compression_factor).reset_index(inplace=False)
    compressed_df['avg_filtered_fft'] = np.fft.fft(compressed_df['avg_filtered'])

    number_of_points = lambda time_mils, samplerate: int((time_mils / 1000) * samplerate)
    decimated_signal = compressed_df['avg_filtered']
    smpl_rate = samplerate / 8

    spec, freq, t = mlab.specgram(decimated_signal, Fs=smpl_rate, NFFT=number_of_points(50, smpl_rate),
                                  window=mplt.mlab.window_hanning)
    # spec, freq, t, img = p.specgram(decimated_signal, Fs=smpl_rate, NFFT=number_of_points(50, smpl_rate),
    #                                 window=mplt.mlab.window_hanning)
    bands = np.concatenate((np.array([0]), np.geomspace(300, 2000, num=21)[1:], np.array([np.inf])), axis=None)
    buckets = list(mi.pairwise(bands))

    def find_new_freq_band(f, bands):
        return list(filter(lambda band: band[0] <= f <= band[1], bands))[0][0]

    def remap_frequencies(spectrum):
        freq_to_energy_list = zip(spectrum, freq)
        energy_band_new_relation = map(
            lambda energy_freq: (energy_freq[0], find_new_freq_band(energy_freq[1], buckets)), freq_to_energy_list)
        grouped_by_frequencies = mi.groupby_transform(energy_band_new_relation, lambda e_f: e_f[1], lambda e_f: e_f[0])
        frequency_energy_sum = map(lambda kv: (kv[0], sum(kv[1])), grouped_by_frequencies)
        return list(mi.unzip(frequency_energy_sum)[1])

    new_spec = np.array(list(map(lambda s: np.array(remap_frequencies(s)), spec.T))).T

    def generate_h(e):
        h = np.zeros((len(e) - 1, len(e[0])))
        for row_idx in range(1, len(e) - 1):
            for col_idx in range(1, len(e[row_idx]) - 1):
                h[row_idx][col_idx] = int(
                    e[row_idx + 1][col_idx] - e[row_idx][col_idx] > e[row_idx + 1][col_idx - 1] - e[row_idx][
                        col_idx - 1])
        return h

    return generate_h(new_spec)


def make_db_with_songs(song, connection=Connection):
    print("%s processing \n" % song)
    name = random.randint(0, 2 ** 16)
    db = Database(connection(), str(name))
    save_song(db, song)
    print("%s processed \n" % song)
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


if __name__ == '__main__':
    path = "/home/leonardo/Documents/seniales/tp/test_songs"
    generate_db(path, "main_db", Connection)
