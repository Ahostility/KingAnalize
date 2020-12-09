from ..audio.audio import read_file
import numpy as np


def wav_by_segments(file_path, segments):
    data, sr = read_file(file_path)
    res_wav = data[int(segments[0][0]*sr):int(segments[0][1]*sr)]
    voice_fragments = [[0.0, segments[0][1] - segments[0][0]]]
    last_segment = segments[0][1] - segments[0][0]
    for i in range(1, len(segments)):
        res_wav = np.concatenate([res_wav, data[int(segments[i][0]*sr):int(segments[i][1]*sr)]])
        frag_diff = segments[i][1] - segments[i][0]
        voice_fragments.append([last_segment, last_segment + frag_diff])
        last_segment += frag_diff
    return res_wav, sr, voice_fragments
