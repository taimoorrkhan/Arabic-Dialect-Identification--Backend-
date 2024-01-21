import torch
import yaml
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from klaam.external.FastSpeech2.inference import infer_tts, prepare_tts_model
from klaam.models.wav2vec import Wav2Vec2ClassificationModel
from klaam.processors.wav2vec import CustomWav2Vec2Processor
from klaam.utils.utils import load_file_to_data, predict

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SpeechClassification:
    def __init__(self, path=None):
        if path is None:
            
            dir = "arbml/wav2vec2-large-xlsr-dialect-classification"
            print("Printing the directory path: ",dir)
        else:
            print("Printing the directory path: ",dir)

            dir = path
        
        self.model = Wav2Vec2ClassificationModel.from_pretrained(dir).to(DEVICE)
        self.processor = CustomWav2Vec2Processor.from_pretrained(dir)

    def classify(self, wav_file, return_prob=False):
        return predict(load_file_to_data(wav_file), self.model, self.processor, mode="cls", return_prob=return_prob)

