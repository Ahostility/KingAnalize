## Install
Для запуска проекта необходимо поставить следующие зависимости:
* Установить [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) 
* Собрать [kaldi](https://github.com/kaldi-asr/kaldi) 
* Собрать [vosk-api](https://github.com/alphacep/vosk-api), если планируется работа на gpu то необходимо собирать одноименную ветку
* Установить все зависимости через [pip](https://pip.pypa.io/en/stable/)
```bash
python3.8 -m pip intall -r requirements.txt
```
* Модифицировать библеотеку Resemblyzer, а именно файл voice_encoder.py слудующим образом
```python
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
```


## USING
### Запуск
Запуск для cpu
```bash
./pipeline.sh
```
Запуск для gpu
```bash
./pipeline_gpu.sh
```
Результаты храниться в дериктории data/output/people/, они разбиты по клиентам
* result - текстовые файлы с эмоциональным окрасом
* wav - аудио фрагменты
* txt - текстовые файлы с диалогом оператор клиент

###### Для отладки 
```bash
python3.8 -m src.audio.preprocessing ./path/test/
python3.8 -m src.SpeechToText.creat_text data/output/full_wav/test.wav
python3.8 -m src.SpeechToText.grouping_dialogue test
python3.8 -m src.diarization.diarization test
python3.8 -m src.EmotionsRecognizer.predict_mixed test
```