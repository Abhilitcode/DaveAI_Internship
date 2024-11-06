import os
import sys
import random
import librosa
from copy import deepcopy

# # Append the current directory (where frames.py is located)
# sys.path.append(os.getcwd())

print("Current working directory:", os.getcwd())

# Explicitly add the path to the synthesis_main directory
project_root = "/mnt/c/Users/HP/Downloads/DaveAI INTERNSHIP/synthesis_main"
if project_root not in sys.path:
    sys.path.append(project_root)

# Add site-packages for the current conda environment to sys.path
conda_site_packages = os.path.join(sys.prefix, "lib", "python3.10", "site-packages")
if conda_site_packages not in sys.path:
    sys.path.append(conda_site_packages)

# Print sys.path to check if the paths are included
print("sys.path:", sys.path)

# Attempt to import common using an absolute path
common_path = "/mnt/c/Users/HP/Downloads/DaveAI INTERNSHIP/synthesis_main/common.py"
if os.path.exists(common_path):
    sys.path.append(os.path.dirname(common_path))  # Add the directory containing common.py
    import common as com
    print("common module imported successfully")
else:
    print(f"{common_path} not found. Please check the file location.")



logger = com.get_logger(__name__)
FRAMES_DEFAULTS = com.read_file(f"{com.ASSETS}/frames_defaults.json", logger)

class Frames:
    def __init__(self, voice_path: str, phonemes_path: str, **kwargs):
        output_dir = com.hp.dirname(voice_path)
        self.audio_duration = librosa.get_duration(path=voice_path)
        self.frame_rate = kwargs.get('frame_rate', com.FRAME_RATE)

        self.phonemes = com.read_file(phonemes_path, logger)
        self.frames_path = output_dir + '/frames.csv'
        try:
            params = deepcopy(kwargs.pop("gesture_params", {}).pop('params', {}) or FRAMES_DEFAULTS[kwargs.pop("default_gestures", "old")])
        except KeyError:
            com.RaiseError(KeyError, f"'default_gestures' must be either 'old', 'male' or 'female'", logger)
    
        context_gestures = params.pop('context_gestures', [])
        self.non_context_gestures = params
        self._talk_head = {}
        self._non_talk_head = {}
        self._talk_anim = []
        self._hand_anim = []
        for gest, gest_vals in self.non_context_gestures.copy().items():
            if not isinstance(gest_vals, dict):
                err = f"received gesture value parameters of type '{type(gest_vals).__name__}' instead of dict with keys weight and frames"
                com.RaiseError(TypeError, err, logger)
            
            if "frames" not in gest_vals or "weight" not in gest_vals:
                err = f"key 'frames' is missing in gesture params"
                com.RaiseError(ValueError, err, logger)
            self.non_context_gestures[gest]['duration'] = round(gest_vals['frames']/self.frame_rate, 2)
            
            if gest.startswith('talk_head'):
                self._talk_head[gest] = gest_vals
            elif "talk" in gest:
                self._talk_anim.append(gest)
                self._non_talk_head[gest] = gest_vals
            elif "hand" in gest:
                self._hand_anim.append(gest)
                self._non_talk_head[gest] = gest_vals
        
        self.context_gestures = []        
        if isinstance(context_gestures, dict):
            for k, v in context_gestures.items():
                self.context_gestures.append([k, v["frames"], v["words"]])        
        if isinstance(self.context_gestures, list):
            for i, gest in enumerate(self.context_gestures):
                gest[1] = round(gest[1]/self.frame_rate, 2)
                self.context_gestures[i] = gest
        logger.info(f"Context gestures are {self.context_gestures}")
        self.frames_list = [["timestamp", "animation"]]        


    def _fit_list(self, time_diff: int, gestures: dict):
        """
        Function to find the gestures that fit within a particular duration.
        """
        fits = []
        for a, b in gestures.copy().items():
            w = b["weight"]
            if not isinstance(w, int):
                err = f"weight of gesture '{a}' should be an integer"
                com.RaiseError(ValueError, err, logger)
        
            if w<1:
                gestures.pop(a)
            elif b["duration"]<=time_diff:
                fits.append(a)

        l = []
        for a in fits:
            l.extend([a,]*int(gestures[a]['weight']))
        return l


    def _generate_combo(self, selected_gestures: list, target_dur: int, gestures: dict):
        g = []
        s = 0
        prev_talk = None
        prev_3 = []
        for a in selected_gestures:
            dur = gestures[a]["duration"]
            if s+dur>target_dur or a in prev_3:
                continue

            if self._hand_anim and self._talk_anim:
                talk = True if "talk" in a else False
                if talk==prev_talk:
                    continue
            
            if s+dur<=target_dur:
                s+=dur
                g.append(a)
                prev_talk = False
                if "talk" in a:
                    prev_talk=True
                prev_3.append(a)
                if len(prev_3)>3:
                    prev_3.pop(0)
    
                w = gestures[a]['weight']-1
                if w==0:
                    gestures.pop(a)
                else:
                    gestures[a]['weight'] = w
        return g, gestures

    def _generate_combo_talk_head(self, selected_gestures: list, target_dur: int, gestures: dict):
        g = []
        s = 0
        prev_3 = []
        for a in selected_gestures:
            dur = gestures[a]["duration"]
            if s+dur>target_dur:
                continue
            
            if s+dur<=target_dur:
                s+=dur
                g.append(a)
                prev_3.append(a)
                if len(prev_3)>3:
                    prev_3.pop(0)
    
                w = gestures[a]['weight']-1
                if w==0:
                    gestures.pop(a)
                else:
                    gestures[a]['weight'] = w
        return g, gestures

    def _non_context_gestures(self, time_ranges: list, talk_head: bool):
        if talk_head:
            gest = self._talk_head.copy()
        else:
            gest = self._non_talk_head.copy()
        
        sorted_times = sorted(time_ranges, key=lambda x: x[1] - x[0])
        new_time_ranges = {tuple(v): v[1]-v[0] for v in sorted_times}
        for k, v in new_time_ranges.items():
            l = self._fit_list(v, gest)
            if not l:
                ge = gest.copy()
                l = self._fit_list(v, ge)
                if not l:
                    continue
                gest = ge

            l = random.sample(l, len(l))
            if talk_head:
                ge, gest = self._generate_combo_talk_head(l, v, gest)
            else:
                ge, gest = self._generate_combo(l, v, gest)
            sta = k[0]
            for g in ge:
                self.frames_list.append([round(sta, 2), g])
                sta+=self.non_context_gestures[g]["duration"]


    def _context_gestures(self):
        non_context_ranges = []
        w_l = []
        st = 0
        for i, l in enumerate(self.phonemes):
            if i==0:
                ge_end = 0

            if ge_end>=self.audio_duration:
                break

            w, sta, end = l[3:6]
            if w not in w_l:
                w_l.append(w)
                for ge, dur, words in self.context_gestures:
                    if w in words:
                        if sta>=ge_end:
                            ge_end = sta+dur
                            if ge_end>self.audio_duration:                                
                                ge_end = self.audio_duration
                                sta = round(ge_end-dur, 2)
                            self.frames_list.append((round(sta, 2), ge))
                            non_context_ranges.append([st, sta])
                            st = ge_end
                        break
                    
            if i==len(self.phonemes)-1:
                if end<self.audio_duration:
                    end = self.audio_duration
                if st<end:
                    non_context_ranges.append([st, end])
        return non_context_ranges

    def frames(self):
        non_context_ranges = self._context_gestures()
        self._non_context_gestures(non_context_ranges, talk_head=False)
        if self._talk_head:
            self._non_context_gestures([[0, self.audio_duration]], talk_head=True)  
        
        fl = sorted(self.frames_list[1:], key=lambda x: x[0])
        frames_list = []
        for i, (t, anim) in enumerate(fl):
            if i!=0:
                if t==prev_t:
                    t = t+1/self.frame_rate
            frames_list.append([t, anim])
            prev_t = t
            
        frames_list.insert(0, self.frames_list[0])
        com.save_file(self.frames_path, frames_list, logger)
        return self.frames_path


if __name__=="__main__":
    import os
    base_dir = "voice_files"  # Updated the correct path
    # base_dir = "static/uploads/3475c3e75ab03cd9a53b1bcc78193832/c358a7ac3ae73ebbaff56728f9da1f11_arabic-male"
    v = os.path.join(base_dir, "voice.wav")
    p = os.path.join(base_dir, "phonemes.csv")
    f = Frames(v, p, default_gestures="male")
    print(f.frames())
