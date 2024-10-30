import os
import string
import epitran

from g2p_en import G2p
from ipatok import tokenise

from pathlib import PosixPath
from kalpy.utterance import Segment
from kalpy.feat.cmvn import CmvnComputer
from kalpy.fstext.lexicon import HierarchicalCtm
from kalpy.utterance import Utterance as KalpyUtterance

from montreal_forced_aligner.corpus.classes import FileData

from montreal_forced_aligner.online.alignment import align_utterance_online
import os
import json
import librosa
import torchaudio
import torch
from torchaudio.pipelines import MMS_FA as bundle

import common as com
from aligners import get_aligners 
import uroman
from mishkal.tashkeel import TashkeelClass 
import re

logger = com.get_logger(__name__)
ipa2arpa = com.read_file(com.hp.joinpath(com.MAPPINGS_DIR, "ipa_to_arpa.json"), logger)
phoneme_ratios: dict = com.read_file(com.hp.joinpath(com.ASSETS, "phoneme_ratios.json"), logger)

g2p = G2p()
epitran_models = {
    "hindi": epitran.Epitran("hin-Deva"),
    "arabic": epitran.Epitran("ara-Arab"),
    "telugu": epitran.Epitran("tel-Telu"),
    "tamil" : epitran.Epitran("tam-Taml"),
    "bengali": epitran.Epitran("ben-Beng"),
    "marathi": epitran.Epitran("mar-Deva")
}


MFA_WORD_REPLACE_MAP = {
    "<eps>": " "
}
MFA_PHONEME_REPLACE_MAP = {
    "sil": " "
}


# Normalization function
def normalize_uroman(text):
    text = text.lower()
    text = text.replace("’", "'")
    text = re.sub("([^a-z' ])", " ", text)
    text = re.sub(' +', ' ', text)
    return text.strip()

class Mfa_output:
    def __init__(self, utterance_path : str, voice_path : str, **kwargs) -> None:
        self.utterance_path = utterance_path
        self.ur = uroman.Uroman()
        self.voice_path = voice_path
        self.output_dir = com.hp.dirname(self.utterance_path)
        if not os.path.isfile(self.utterance_path) or not os.path.isfile(self.voice_path):
            com.RaiseError(FileNotFoundError, f"output_dir {self.output_dir} does not have either voice.wav and/or utterance.txt files", logger)
        
        self.aligner_output_path = self.output_dir + "/mfa_output.json"
        self.lang = kwargs.get("language", "english").lower()
        self.language = kwargs.get("language", "english")
        self.kwargs: dict = kwargs.get("phoneme_params", {})

        insts = get_aligners(list_=False)
        self.aligner_instance = insts.get(self.language)
        if not self.aligner_instance:
            com.RaiseError(ValueError, f"'{self.language}' language is not yet supported. Languages supported are: {', '.join(list(insts))}", logger)
        
        self.acoustic_model = self.aligner_instance.acoustic_model
        self.lexicon_compiler = self.aligner_instance.lexicon_compiler
        self.tokenizer = self.aligner_instance.tokenizer
        self.inst_conf = self.aligner_instance.conf
    

    def _assign_phoneme_suffix(self, phone: str, phone_index: int, total_phones: int):
        if phone_index==0 and total_phones==1:
            phone = phone+"_S"
        elif phone_index==0:
            phone = phone+"_B"
        elif phone_index==total_phones-1:
            phone = phone+"_E"
        else:
            phone = phone+"_I"
        return phone.upper()
    

    def _check_is_arpa(self, p: str):
        if p[0] in 'AEIOU':
            if not (len(p)==3 or p[:2].isupper()):
                return False
            return True
        if p.isupper():
            return True
        return False
    

    def convert_ipa_arpa(self, p: str):
        arpa = self._check_is_arpa(p)
        if not arpa:
            pi = ipa2arpa.get(p)
            if pi:
                p = pi
            else:
                if p in string.punctuation or not p.strip():
                    p = ""
                else:
                    with open("missing_ipas.txt", "a") as f:
                        f.write(p)
                        f.write("\n")
                    logger.error(f"missing ipa for '{p}', using AA1 for vowel and N for consonants'")
                    p = "AA1" if p[0] not in "AEIOU" else "N"
        return p


    def _oov(self, w_sta, w_end, word, lang: str=None):
        logger.info(f"found {word} which is not in vocabulary")
        t = w_end-w_sta
        lang = lang or self.lang
        if lang=="english":
            ph = g2p(word)
        elif lang in epitran_models:
            epi = epitran_models[lang]
            ph = epi.transliterate(word)
            ph = tokenise(ph)
        #handling the arabic case
        elif lang == "arabic":
            diacritized_text = vocalizer.tashkeel(word)
            romanized_text = self.ur.romanize_string(diacritized_text)
            normalized_arabic_text = normalize_uroman(romanized_arabic_text)
            ph = tokenise(normalized_arabic_text)           
        else:
            com.RaiseError(NotImplementedError, f"Language {lang} is not yet supported", logger)
        
        new_ph = {}
        ph_len = len(ph)
        for i, p in enumerate(ph):
            p = self.convert_ipa_arpa(p)
            if i==0:
                prev = "-"
            else:
                prev = self._assign_phoneme_suffix(ph[i-1], i-1, ph_len)
            
            if i<len(ph)-1:
                nxt = self._assign_phoneme_suffix(ph[i+1], i+1, ph_len)
            else:
                nxt = "-"
            
            ratio = phoneme_ratios.get(self._assign_phoneme_suffix(p, i, ph_len), {}).get(prev, {}).get(nxt)
            if not ratio:
                ratio = 0.3
                for vow in ["A", "E", "I", "O", "U"]:
                    if p.startswith(vow):
                        ratio = 0.5
                        break
            new_ph[p] = ratio
        
        phones = []
        tw = sum(new_ph.values())
        p_sta = w_sta
        for i, (p, w) in enumerate(new_ph.items()):
            if i==len(new_ph)-1:
                p_end = w_end
            else:
                p_end = round(p_sta+(w/tw)*t, 2)
            phones.append([p_sta, p_end, p])
            p_sta = p_end
        logger.info(f"generated the phonemes and timestamps for the word {word}")
        return phones
    
    def _get_word_timestamps(self):
        waveform, sample_rate = librosa.load(self.voice_path, sr=None)
        transcript = com.read_file(self.utterance_path, logger)
        token_spans, num_frames = self.get_token_spans_pytorch(waveform, transcript)
        ratio = waveform.size(1) / num_frames
        phones = []
        word_timestamps = []

        for t_spans, word in zip(token_spans, transcript):
            start_time = t_spans[0].start * ratio / self.sample_rate
            end_time = t_spans[-1].end * ratio / self.sample_rate
            word_timestamps.append({"word": word, "start": start_time, "end": end_time})

            # Call the _oov function to get the phonemes and their timestamps
            ph = self._oov(start_time, end_time, word, lang=self.lang)
            phones.extend(ph)  # Add phonemes to the phones list

        result = {
            "start": 0,
            "end": librosa.get_duration(waveform.numpy(), sr=self.sample_rate),
            "words": word_timestamps,  # List of word timestamps
            "phones": phones  # List of phoneme timestamps
        }
        
        return result
     
    def get_token_spans_pytorch(self, waveform, transcript):
        model = bundle.get_model().to(self.device)
        tokenizer = bundle.get_tokenizer()
        aligner = bundle.get_aligner()
        
        with torch.inference_mode():
            emission, = model(waveform.to(self.device))
            token_spans = aligner(emission[0], tokenizer(transcript))
        
        num_frames = emission.size(1)
        return token_spans, num_frames
        
    
    def mfa_output(self):
        voice_path = PosixPath(self.voice_path)
        utterance_path = PosixPath(self.utterance_path)
        if com.hp.isfile(self.aligner_output_path):
            com.os.remove(self.aligner_output_path)

        alignment_file = PosixPath(self.aligner_output_path)
        file_name = voice_path.stem
        file = FileData.parse_file(file_name, voice_path, utterance_path, "", 0)
        file_ctm = HierarchicalCtm([])
        cmvn_computer = CmvnComputer()

        utterances = []
        for utterance in file.utterances:
            seg = Segment(voice_path, utterance.begin, utterance.end, utterance.channel)
            utt = KalpyUtterance(seg, utterance.text)
            utt.generate_mfccs(self.acoustic_model.mfcc_computer)
            utterances.append(utt)
        cmvn = cmvn_computer.compute_cmvn_from_features([utt.mfccs for utt in utterances])
        self.inst_conf.update(self.kwargs)

        align_options = {
            k: v
            for k, v in self.inst_conf.items()
            if k
            in [
                "beam",
                "retry_beam",
                "acoustic_scale",
                "transition_scale",
                "self_loop_scale",
                "boost_silence",
            ]
        }
        for utt in utterances:
            utt.apply_cmvn(cmvn)
            ctm = align_utterance_online(
                self.acoustic_model,
                utt,
                self.lexicon_compiler,
                tokenizer=self.tokenizer,
                #g2p_model=g2p_model,
                **align_options,
            )
            file_ctm.word_intervals.extend(ctm.word_intervals)

        file_ctm.export_textgrid(
            alignment_file, file_duration=file.wav_info.duration, output_format="json"
        )
        dict_ = com.read_file(self.aligner_output_path, logger)
        com.save_file(self.aligner_output_path.replace("mfa", "original_mfa"), dict_, logger)

        nd = {}
        for k, v in dict_.copy().items():
            if k=="tiers":
                tiers = dict_.pop("tiers")
                nw = []
                for sta, end, w  in tiers["words"]["entries"]:
                    w = MFA_WORD_REPLACE_MAP.get(w, w)
                    nw.append([sta, end, w])

                np = []
                for sta, end, p in tiers["phones"]["entries"]:
                    p = MFA_PHONEME_REPLACE_MAP.get(p, p)
                    np.append([sta, end, p])
                
                nd["words"] = nw
                nd["phones"] = []
                
                for w_sta, w_end, w in nw:
                    for p_sta, p_end, p in np:
                        if len(p)>2 and p[-2:] in ["_B", "_E", "_I", "_S"]:
                            p = p[:-2]
                        
                        if p_sta<w_sta or p_end>w_end:
                            continue
                        elif w_sta==p_sta and w_end==p_end and p.startswith("spn"):
                            out = self._oov(p_sta, p_end, w)
                            if not out:
                                logger.warning("attempting to use english language as a backup")
                                out = self._oov(p_sta, p_end, w, lang="english")
                                #TODO: use epitran backoff to handle different languages
                            if out:
                                nd["phones"].extend(out)
                            else:
                                nd["phones"].append([p_sta, p_end, " "])
                            break
                        else:
                            if p.strip():
                                p = self.convert_ipa_arpa(p)
                            nd["phones"].append([p_sta, p_end, p.upper()])
            else:
                nd[k] = v
        com.save_file(self.aligner_output_path, nd, logger)
        return self.aligner_output_path