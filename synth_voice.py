#Have to do the import here, otherwise it wont work
import os
import numpy as np
import torch
import json
import sounddevice

import sys
sys.path.append('hifi-gan')
sys.path.append('tacotron2')

#Imports from Tacotron2
from hparams import create_hparams
from model import Tacotron2
from layers import TacotronSTFT
from text import text_to_sequence     

#Imports from hifi-gan
from env import AttrDict
from models import Generator
from meldataset import MAX_WAV_VALUE


class VoiceSynthetizer:
    models_directory = "C:\\Tacotron2_Voice_Models" #Only for windows
    def __init__(self, model_name, sampling_rate=22050):
        
        if not os.path.exists("tacotron2"):
            raise FileNotFoundError("Submodule 'tacotron2' is not installed. Clone the repo 'https://github.com/NVIDIA/tacotron2.git' to install it.")
        if not os.path.exists("hifi-gan"):
            raise FileNotFoundError("Submodule 'hifi-gan' is not installed. Clone the repo 'https://github.com/SortAnon/hifi-gan' to install it.") 
        
        self.sampling_rate=sampling_rate
        
        # Setup Pronounciation Dictionary
        self.pronunciation_dict = {}
        for line in reversed((open('merged.dict.txt', "r").read()).splitlines()):
            self.pronunciation_dict[(line.split(" ",1))[0]] = (line.split(" ",1))[1].strip()
        self.hifigan = VoiceSynthetizer.load_hifigan()

        self.load_Tactron2Model(model_name)
        
        self.model.decoder.max_decoder_steps = 3000 #@param {type:"integer"}
        stop_threshold = 0.324 #@param {type:"number"}
        self.model.decoder.gate_threshold = stop_threshold

        
    @staticmethod
    def available_voices():
        return [voice for voice in os.listdir(VoiceSynthetizer.models_directory) \
                if os.path.isfile(os.path.join(VoiceSynthetizer.models_directory, voice))]
        
    def ARPA(self, text, punctuation=r"!?,.;", EOS_Token=True):
                out = ''
                for word_ in text.split(" "):
                    word=word_; end_chars = ''
                    while any(elem in word for elem in punctuation) and len(word) > 1:
                        if word[-1] in punctuation: end_chars = word[-1] + end_chars; word = word[:-1]
                        else: break
                    try:
                        word_arpa = self.pronunciation_dict[word.upper()]
                        word = "{" + str(word_arpa) + "}"
                    except KeyError: pass
                    out = (out + " " + word + end_chars).strip()
                if EOS_Token and out[-1] != ";": out += ";"
                return out

    @staticmethod
    def load_hifigan():        
        hifigan_pretrained_model = 'hifimodel'
        if not os.path.exists(hifigan_pretrained_model):
            raise Exception("HiFI-GAN model is not found!")
        
        # Load HiFi-GAN
        conf = os.path.join("hifi-gan", "config_v1.json")
        with open(conf) as f:
            json_config = json.loads(f.read())
        h = AttrDict(json_config)
        torch.manual_seed(h.seed)
        hifigan = Generator(h).to(torch.device("cuda"))
        state_dict_g = torch.load(hifigan_pretrained_model, map_location=torch.device("cuda"))
        hifigan.load_state_dict(state_dict_g["generator"])
        hifigan.eval()
        hifigan.remove_weight_norm()
        return hifigan   

    def has_MMI(self, STATE_DICT):
                return any(True for x in STATE_DICT.keys() if "mi." in x)

    def load_Tactron2Model(self, model_name):                 
        tacotron2_pretrained_model = os.path.join(VoiceSynthetizer.models_directory, model_name)
        if not os.path.exists(tacotron2_pretrained_model):
            raise Exception("Voice model not found. Check the location of the model")
            
        # Load Tacotron2 and Config
        hparams = create_hparams()
        hparams.sampling_rate = self.sampling_rate
        hparams.max_decoder_steps = 3000 # Max Duration
        hparams.gate_threshold = 0.25 # Model must be 25% sure the clip is over before ending generation
        self.model = Tacotron2(hparams)
        state_dict = torch.load(tacotron2_pretrained_model)['state_dict']
        if self.has_MMI(state_dict):
            raise Exception("ERROR: This notebook does not currently support MMI models.")
        self.model.load_state_dict(state_dict)
        _ = self.model.cuda().eval().half()
    
    
    def end_to_end_infer(self, text, pronounciation_dictionary=False, show_graphs=False):
        
        
        for i in [x for x in text.split("\n") if len(x)]:
            if not pronounciation_dictionary:
                if i[-1] != ";": i=i+";" 
            else: i = ARPA(i)
            with torch.no_grad(): # save VRAM by not including gradients
                sequence = np.array(text_to_sequence(i, ['english_cleaners']))[None, :]
                sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()
                _, mel_outputs_postnet, _, _ = self.model.inference(sequence)                
                y_g_hat = self.hifigan(mel_outputs_postnet.float())
                audio = y_g_hat.squeeze()
                audio = audio * MAX_WAV_VALUE

                return audio.cpu().numpy().astype("int16")

    def speak(self, line):
        audio= self.end_to_end_infer(line)
        sounddevice.play(audio, samplerate=self.sampling_rate)
        sounddevice.wait()