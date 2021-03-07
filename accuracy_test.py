import random
from itertools import product, repeat

import soundfile as sf

from database import Connection, Database, DB
from generatedb import generate_footprint, get_song_paths, stereo2mono
import numpy as np
import multiprocessing as mp


def _perform_query(query_db):
    return perform_query(query_db[0], query_db[1])


def clean_song_accuracy_test(songs, db, n=40):
    print("Started clean song accuracy test with %i songs" % n)
    songs_to_test = select_random_songs_with_duplicates(songs, n)
    segment_sizes = [5, 10, 20]
    song_arrays = map(sf.read, songs_to_test)
    queries = product(song_arrays, segment_sizes)
    predictions = run_all_queries(db, queries)
    # l = predictions
    # print(list(l))
    hits = map(lambda pred_query: (check_hit(pred_query), pred_query),
               zip(predictions, product(songs_to_test, segment_sizes)))
    return list(hits)


def run_all_queries(db, queries):
    p = mp.Pool(4)
    predictions = p.map(_perform_query, zip(queries, repeat(db)))
    p.close()
    return predictions


def noisy_song_accuracy_test(songs, db, n=40):
    print("Started clean song accuracy test with %i songs" % n)
    songs_to_test = select_random_songs_with_duplicates(songs, n)
    segment_sizes = [5, 10, 20]
    noise_levels_db = [0, 10, 20]
    songs_info = map(sf.read, songs_to_test)
    song_arrays = map(add_noise_to_song_info,
                      product(songs_info, noise_levels_db))

    queries = product(song_arrays, segment_sizes)
    predictions = run_all_queries(db, queries)
    hits = map(lambda p_song: (check_hit((p_song[0], p_song[1][:2])), p_song),
               zip(predictions, product(songs_to_test, noise_levels_db, segment_sizes)))
    return list(hits)


def add_noise_to_song_info(songinfo_srn):
    song_info, srn_db = songinfo_srn
    song_array, samplerate = song_info
    channel_0 = song_array.transpose()[0]
    channel_1 = song_array.transpose()[0]
    noisy_ch_0 = channel_0 + make_noise(channel_0, srn_db)
    noisy_ch_1 = channel_1 + make_noise(channel_1, srn_db)

    return np.array([noisy_ch_0, noisy_ch_1]).transpose(), samplerate


def select_random_songs_with_duplicates(songs, n=50):
    return [songs[random.randint(0, len(songs) - 1)] for _ in range(n)]


def perform_query(query, db):
    song_info, window_size_seconds = query
    song_array, samplerate = song_info
    window_length = int(window_size_seconds * samplerate)
    song_length = len(song_array)
    song_start = int(random.random() * (song_length - window_length))
    truncated_song = song_array[song_start:song_start + window_length]
    # print("Performing query with  window size %i" % window_size_seconds)
    footprint = generate_footprint(truncated_song, samplerate)
    return db.query(footprint)


def check_hit(prediction_query):
    prediction, query = prediction_query
    song_name, window_size_seconds = query
    song_name = song_name.split("/")[-1]
    songs_predicted = map(lambda name_score: name_score[0], prediction)
    return any(map(lambda pred: song_name == pred, songs_predicted))


def make_noise(signal, target_snr_db):
    # Calculate signal power and convert to dB
    signal_power = signal ** 2
    sig_avg_watts = np.mean(signal_power)
    sig_avg_db = 10 * np.log10(sig_avg_watts)
    # Calculate noise according to [2] then convert to watts
    noise_avg_db = sig_avg_db - target_snr_db
    noise_avg_watts = 10 ** (noise_avg_db / 10)
    # Generate an sample of white noise
    mean_noise = 0
    return np.random.normal(mean_noise, np.sqrt(noise_avg_watts), len(signal_power))


def run_all():
    seed = 2 ** 64 - 1
    random.seed(seed)
    path = "/home/leonardo/Documents/seniales/tp/10songs"
    db_path = path + "/main_db"
    songs = get_song_paths(path)
    db = Database(Connection.from_disk(db_path), "acc_test_db")
    hits = clean_song_accuracy_test(songs, db, 10)
    hit_count = 0
    queries_count = 0
    for is_hit, predictions_query in hits:
        hit_count += is_hit
        queries_count += 1
        predictions, query = predictions_query
        song, window_size = query
        # print(predictions)
        print("%s \t  %i \t %s" % (song.split("/")[-1], window_size, is_hit))
    print(hit_count, queries_count)
    hits = noisy_song_accuracy_test(songs, db, 10)
    hit_count = 0
    queries_count = 0
    for is_hit, predictions_query in hits:
        hit_count += is_hit
        queries_count += 1
        predictions, query = predictions_query
        song, noise_level, window_size = query
        print("%s \t  %i \t  %s \t  %i" % (song.split("/")[-1], window_size, is_hit, noise_level))
    print(hit_count, queries_count)


def run_all2():
    files = get_song_paths("./10songs/")
    hash_nbits = 20  # Cantidad de bits de los hashes
    n_entries = len(files)  # Cantidad de columnas de la tabla; cant canciones
    ID_nbits = 12  # Cantidad de bits para ID numerico de la cancion
    # Tamanio en memoria para hash de 20 bits: 4MB x N_columnas
    db = DB.from_disk(hash_nbits, n_entries, ID_nbits, "/home/leonardo/Documents/seniales/tp/10songs/main_db")

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
