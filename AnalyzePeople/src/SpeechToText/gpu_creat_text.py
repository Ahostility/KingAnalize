from ..dirs import MARKUP_TXT
from vosk import Model, KaldiRecognizer, SetLogLevel, GpuInit, GpuInstantiate
import sys
import wave
import json

GpuInit()
def thread_init():
    GpuInstantiate()
thread_init()

SetLogLevel(0)
model = Model("models/kaldi_vosk")


def write_file(data, name):
    with open(f'{MARKUP_TXT}/{name}.txt', 'w', encoding='utf-8') as f:
        for word in data:
            f.write(str([*word.values()])[1:-1] + '\n')


def parse_json(data):
    data = json.loads(data)
    for sample in data['result']:
        del sample['conf']
    return data['result']


def creat_text_gpu(path):
    wf = wave.open(path, "rb")
    rec = KaldiRecognizer(model, wf.getframerate())

    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            pass
        else:
            rec.PartialResult()

    write_file(parse_json(rec.FinalResult()), path.split('/')[-1].replace('.wav', ''))


if __name__ == '__main__':
    creat_text_gpu(sys.argv[1])
