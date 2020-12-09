from ..dirs import PEOPLE_DIR, SLEPOK_DIR
from .audio import preprocess_wav
from resemblyzer.voice_encoder import VoiceEncoder
from resemblyzer.hparams import sampling_rate
from .cut_pauses import wav_by_segments
from .segment_another import get_another_fragments
from .kaldi_tools import parse_kaldi_file, creat_output_file, write_file
import os
import numpy as np
import sys


def get_speaker_wavs(segments, wav):
    speaker_wavs = []
    for s in segments:
        speaker_wavs += list(wav[int(s[0] * sampling_rate):int(s[1] * sampling_rate)])
    return np.array(speaker_wavs)


def get_similarity(encoder, cont_embeds, speaker_wav):
    speaker_embeds = encoder.embed_utterance(speaker_wav)
    return cont_embeds @ speaker_embeds


def get_similarity_several(encoder, cont_embeds, speaker_wavs, speaker_names):
    res = dict()
    for i in range(len(speaker_names)):
        res[speaker_names[i]] = get_similarity(encoder, cont_embeds, speaker_wavs[i])
    return res


def get_change_moments(similarity_dict, wav_splits):
    prev_name = ''
    res = []
    res_names = []
    list_names = list(similarity_dict.keys())
    for i in range(len(wav_splits)):
        similarities = [s[i] for s in similarity_dict.values()]
        best = np.argmax(similarities)
        name, similarity = list_names[best], similarities[best]
        if name != prev_name:
            res.append(i)
            res_names.append(name)
        prev_name = name
    diar_frags = list(wav_splits[res[1:]])
    choose = lambda x: x[0] + (x[1] - x[0])/3
    return [wav_splits[0][0]] + list(map(choose, diar_frags)) + [wav_splits[-1][1]], res_names


def get_operator_wavs(operators_dir):
    operator_names = []
    wavs = []
    for slepok_file in os.listdir(operators_dir):
        file_path = os.path.join(operators_dir, slepok_file)
        operator_names.append('_'.join(slepok_file.split('.')[:-1]))
        wav = preprocess_wav(file_path)
        wavs.append(wav)
    return wavs, operator_names


def get_fragment_parts(change_moments, names):
    res = []
    for i in range(len(names)):
        res.append([names[i], change_moments[i], change_moments[i+1]])
    return res


def identify_operator(wav, encoder, cont_embeds):
    #percent = 80
    operators_wavs, operators_names = get_operator_wavs(str(SLEPOK_DIR))
    operators_similarity = get_similarity_several(encoder, cont_embeds, operators_wavs, operators_names)
    operators_similarity_mean = [op_sim.mean() for op_sim in operators_similarity.values()]
    best_id = np.argmax(operators_similarity_mean)
    best_operator_name = operators_names[best_id]
    return operators_wavs[best_id], operators_similarity[best_operator_name], best_operator_name


def diarize_by_wav(wav_fpath, client_wav, operator_wav, operator_name, client_name, encoder, start_end_text):
    wav = preprocess_wav(wav_fpath)
    _, cont_embeds, wav_splits = encoder.embed_utterance(wav, return_partials=True, rate=16)
    client_similarity = get_similarity(encoder, cont_embeds, client_wav)
    operator_similarity = get_similarity(encoder, cont_embeds, operator_wav)

    similarity_dict = {operator_name: operator_similarity, client_name: client_similarity}
    wav_splits_seconds = np.array(list(map(lambda x: [x.start / sampling_rate, x.stop / sampling_rate], wav_splits)))

    # change moments in audio
    change_moments, names = get_change_moments(similarity_dict, wav_splits_seconds)
    fragment_parts = get_fragment_parts(change_moments, names)
    return fragment_parts#unite_time_arr(fragment_parts, start_end_text)


def save_text(arr, name='splits/разбиение.txt'):
    f = open(name, 'w')
    f.write(str(arr).replace('], [', ']\n[').replace('), (', ')\n('))
    f.close()


def pipeline(wav_fpath, cutted_data, cut_sr, voice_fragments, start_end_text, device):
    cutted_wav = preprocess_wav(cutted_data, source_sr=cut_sr)

    encoder = VoiceEncoder(device, verbose=False)
    _, cont_embeds, wav_splits = encoder.embed_utterance(cutted_wav, return_partials=True, rate=16)

    # choose operator
    operator_wav, operator_similarity, operator_name = identify_operator(cutted_wav, encoder, cont_embeds)
    wav_splits_seconds = np.array(list(map(lambda x: [x.start / sampling_rate, x.stop / sampling_rate], wav_splits)))
    # looking for client
    client_fragments = [get_another_fragments(voice_fragments, operator_similarity, wav_splits_seconds)]
    client_wav = get_speaker_wavs(client_fragments, cutted_wav)
    client_name = 'Client'

    # change moments in audio
    diarized_fragments = diarize_by_wav(wav_fpath, client_wav, operator_wav, operator_name, client_name, encoder, start_end_text)
    return diarized_fragments, operator_name #[[name1, start1, end1], [name2, start2, end2]], operator_name


def diarize(wav_fpath, file_kaldi, device):
    start_end_text = parse_kaldi_file(file_kaldi)
    cutted_data, cut_sr, voice_fragments = wav_by_segments(wav_fpath, start_end_text)
    return pipeline(wav_fpath, cutted_data, cut_sr, voice_fragments, start_end_text, device)


def diarize_all(name, gpu=False):
    folder_kaldi = f'{PEOPLE_DIR}/{name}/txt/'
    folder_wav = f'{PEOPLE_DIR}/{name}/wav/'
    device = 'cuda' if gpu else 'cpu'
    for idx, file_name in enumerate(sorted(os.listdir(folder_kaldi))):
        kaldi_fpath = folder_kaldi + file_name
        wav_fpath = folder_wav + file_name.replace('.txt', '.wav')
        markup = diarize(wav_fpath, kaldi_fpath, device)
        result, spk = creat_output_file(kaldi_fpath, markup)
        write_file(result, name, spk, idx)


if __name__ == '__main__':
    diarize_all(sys.argv[1])

''' 
        MAX_SIZE = 3500
        start = 0
        end = MAX_SIZE
        partial_embeds = 0
        if MAX_SIZE > len(mels):
            with torch.no_grad():
                melss = torch.from_numpy(mels[start:]).to(self.device)
                partial_embeds = self(melss).cpu().numpy()
        else:
            while True:
                if end > len(mels):
                    with torch.no_grad():
                        melss = torch.from_numpy(mels[start:]).to(self.device)
                        partial_embeds = np.concatenate((partial_embeds, self(melss).cpu().numpy()), axis=0)
                            break
                    elif start == 0:
                        with torch.no_grad():
                            melss = torch.from_numpy(mels[start:end]).to(self.device)
                            partial_embeds = self(melss).cpu().numpy()
                    else:
                        with torch.no_grad():
                            melss = torch.from_numpy(mels[start:end]).to(self.device)
                            partial_embeds = np.concatenate((partial_embeds, self(melss).cpu().numpy()), axis=0)
                    start = end
                    end += MAX_SIZE
                    torch.cuda.empty_cache()
                '''

# from ..dirs import PEOPLE_DIR, SLEPOK_DIR
# from .audio import preprocess_wav
# from resemblyzer.voice_encoder import VoiceEncoder
# from resemblyzer.hparams import sampling_rate
# from .cut_pauses import wav_by_segments
# from .segment_another import get_another_fragments
# from .kaldi_tools import parse_kaldi_file, creat_output_file, write_file
# import os
# import numpy as np
# import sys
#
#
# def get_speaker_wavs(segments, wav):
#     speaker_wavs = []
#     for s in segments:
#         speaker_wavs += list(wav[int(s[0] * sampling_rate):int(s[1] * sampling_rate)])
#     return np.array(speaker_wavs)
#
#
# def get_similarity(encoder, cont_embeds, speaker_wav):
#     speaker_embeds = encoder.embed_utterance(speaker_wav)
#     return cont_embeds @ speaker_embeds
#
#
# def get_similarity_several(encoder, cont_embeds, speaker_wavs, speaker_names):
#     res = dict()
#     for i in range(len(speaker_names)):
#         res[speaker_names[i]] = get_similarity(encoder, cont_embeds, speaker_wavs[i])
#     return res
#
#
# def get_change_moments(similarity_dict, wav_splits):
#     prev_name = ''
#     res = []
#     res_names = []
#     list_names = list(similarity_dict.keys())
#     for i in range(len(wav_splits)):
#         similarities = [s[i] for s in similarity_dict.values()]
#         best = np.argmax(similarities)
#         name, similarity = list_names[best], similarities[best]
#         if name != prev_name:
#             res.append(i)
#             res_names.append(name)
#         prev_name = name
#     diar_frags = list(wav_splits[res[1:]])
#     choose = lambda x: x[0] + (x[1] - x[0])/3
#     return [wav_splits[0][0]] + list(map(choose, diar_frags)) + [wav_splits[-1][1]], res_names
#
#
# def get_operator_wavs(operators_dir):
#     operator_names = []
#     wavs = []
#     for slepok_file in os.listdir(operators_dir):
#         file_path = os.path.join(operators_dir, slepok_file)
#         operator_names.append('_'.join(slepok_file.split('.')[:-1]))
#         wav = preprocess_wav(file_path)
#         wavs.append(wav)
#     return wavs, operator_names
#
#
# def get_fragment_parts(change_moments, names):
#     res = []
#     for i in range(len(names)):
#         res.append([names[i], change_moments[i], change_moments[i+1]])
#     return res
#
#
# def identify_operator(wav, encoder, cont_embeds):
#     #percent = 80
#     operators_wavs, operators_names = get_operator_wavs(str(SLEPOK_DIR))
#     operators_similarity = get_similarity_several(encoder, cont_embeds, operators_wavs, operators_names)
#     operators_similarity_mean = [op_sim.mean() for op_sim in operators_similarity.values()]
#     best_id = np.argmax(operators_similarity_mean)
#     best_operator_name = operators_names[best_id]
#     return operators_wavs[best_id], operators_similarity[best_operator_name], best_operator_name
#
#
# def diarize_by_wav(wav_fpath, client_wav, operator_wav, operator_name, client_name, encoder, start_end_text):
#     wav = preprocess_wav(wav_fpath)
#     _, cont_embeds, wav_splits = encoder.embed_utterance(wav, return_partials=True, rate=16)
#     client_similarity = get_similarity(encoder, cont_embeds, client_wav)
#     operator_similarity = get_similarity(encoder, cont_embeds, operator_wav)
#
#     similarity_dict = {operator_name: operator_similarity, client_name: client_similarity}
#     wav_splits_seconds = np.array(list(map(lambda x: [x.start / sampling_rate, x.stop / sampling_rate], wav_splits)))
#
#     # change moments in audio
#     change_moments, names = get_change_moments(similarity_dict, wav_splits_seconds)
#     fragment_parts = get_fragment_parts(change_moments, names)
#     return fragment_parts#unite_time_arr(fragment_parts, start_end_text)
#
#
# def save_text(arr, name='splits/разбиение.txt'):
#     f = open(name, 'w')
#     f.write(str(arr).replace('], [', ']\n[').replace('), (', ')\n('))
#     f.close()
#
#
# def pipeline(wav_fpath, cutted_data, cut_sr, voice_fragments, start_end_text, device):
#     cutted_wav = preprocess_wav(cutted_data, source_sr=cut_sr)
#
#     encoder = VoiceEncoder(device)
#     _, cont_embeds, wav_splits = encoder.embed_utterance(cutted_wav, return_partials=True, rate=16)
#
#     # choose operator
#     operator_wav, operator_similarity, operator_name = identify_operator(cutted_wav, encoder, cont_embeds)
#     wav_splits_seconds = np.array(list(map(lambda x: [x.start / sampling_rate, x.stop / sampling_rate], wav_splits)))
#     # looking for client
#     client_fragments = [get_another_fragments(voice_fragments, operator_similarity, wav_splits_seconds)]
#     client_wav = get_speaker_wavs(client_fragments, cutted_wav)
#     client_name = 'Client'
#
#     # change moments in audio
#     diarized_fragments = diarize_by_wav(wav_fpath, client_wav, operator_wav, operator_name, client_name, encoder, start_end_text)
#     return diarized_fragments, operator_name #[[name1, start1, end1], [name2, start2, end2]], operator_name
#
#
# def diarize(wav_fpath, file_kaldi, device):
#     start_end_text = parse_kaldi_file(file_kaldi)
#     cutted_data, cut_sr, voice_fragments = wav_by_segments(wav_fpath, start_end_text[:-1])
#     return pipeline(wav_fpath, cutted_data, cut_sr, voice_fragments, start_end_text, device)
#
#
# def diarize_all(name, gpu):
#     folder_kaldi = f'{PEOPLE_DIR}/{name}/txt/'
#     folder_wav = f'{PEOPLE_DIR}/{name}/wav/'
#     device = 'cuda' if gpu else 'cpu'
#     for idx, file_name in enumerate(sorted(os.listdir(folder_kaldi))):
#         kaldi_fpath = folder_kaldi + file_name
#         wav_fpath = folder_wav + file_name.replace('.txt', '.wav')
#         markup = diarize(wav_fpath, kaldi_fpath, device)
#         result, spk = creat_output_file(kaldi_fpath, markup)
#         write_file(result, name, spk, idx)
#
#
# if __name__ == '__main__':
#     diarize_all(sys.argv[1])
#
# '''
#         MAX_SIZE = 3500
#         start = 0
#         end = MAX_SIZE
#         partial_embeds = 0
#         if MAX_SIZE > len(mels):
#             with torch.no_grad():
#                 melss = torch.from_numpy(mels[start:]).to(self.device)
#                 partial_embeds = self(melss).cpu().numpy()
#         else:
#             while True:
#                 print(start)
#                 if end > len(mels):
#                     with torch.no_grad():
#                         melss = torch.from_numpy(mels[start:]).to(self.device)
#                         partial_embeds = np.concatenate((partial_embeds, self(melss).cpu().numpy()), axis=0)
#                             break
#                     elif start == 0:
#                         with torch.no_grad():
#                             melss = torch.from_numpy(mels[start:end]).to(self.device)
#                             partial_embeds = self(melss).cpu().numpy()
#                     else:
#                         with torch.no_grad():
#                             melss = torch.from_numpy(mels[start:end]).to(self.device)
#                             partial_embeds = np.concatenate((partial_embeds, self(melss).cpu().numpy()), axis=0)
#                     start = end
#                     end += MAX_SIZE
#                     torch.cuda.empty_cache()
#                 '''