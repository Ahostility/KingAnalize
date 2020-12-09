from ..dirs import FINAL_DIR
from .audio import gluing, read_file, write_file
import os
import sys


def preprocessing_folder(path_folder_wav):
    wav_name = f"{FINAL_DIR}/{path_folder_wav.split('/')[-2]}.wav"
    data = gluing([read_file(f'{path_folder_wav}/{file_name}')[0] for file_name in os.listdir(path_folder_wav)])
    write_file(wav_name, data)


if __name__ == '__main__':
    preprocessing_folder(sys.argv[1])