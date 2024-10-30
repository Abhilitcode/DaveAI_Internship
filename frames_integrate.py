import os
import sys
import random
import librosa
from copy import deepcopy

sys.path.append(os.getcwd())
import common as com

logger = com.get_logger(__name__)
FRAMES_DEFAULTS = com.read_file(f"{com.ASSETS}/frames_defaults.json", logger)


class GestureSelector:
    def __init__(self, gestures_dict):
        self.gestures = gestures_dict
        self.organized_gesture = {}
        self.organized_by_groups()

    def organized_by_groups(self):
        for gesture, details in self.gestures.items():
            group = details.get("group", "default")  # Default group if not specified
            if group not in self.organized_gesture:
                self.organized_gesture[group] = {}
            self.organized_gesture[group][gesture] = {
                "frames": details["frames"],
                "weights": details["weights"]
            }

    def gesture_time_range(self, time_range, frame_rate=25):
        selected_gestures = []
        total_frames = time_range * frame_rate  # Convert total time range to frames

        for group, gestures in self.organized_gesture.items():
            group_selected_gestures = []
            remaining_frames = total_frames
            start_time = 0

            # Continuously select gestures until the group's time range is filled
            while remaining_frames > 0:
                gesture_names = list(gestures.keys())
                weights = [details['weights'] for details in gestures.values()]

                # Randomly choose a gesture based on weights
                chosen_gesture = random.choices(gesture_names, weights)[0]
                chosen_details = gestures[chosen_gesture]
                frames = chosen_details['frames']

                # If the gesture fits within the remaining frames, append it
                if frames <= remaining_frames:
                    end_time = start_time + frames / frame_rate
                    group_selected_gestures.append([start_time, chosen_gesture])
                    start_time = end_time
                    remaining_frames -= frames
                else:
                    # If gesture doesn't fit, break the loop
                    break

            selected_gestures.extend(group_selected_gestures)

        return selected_gestures

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
        # this is a source of gestures based on whether 'talk_head' r used.
        gesture_source = self._talk_head if talk_head else self._non_talk_head     
        for time_range in time_ranges:
            duration = time_range[1] - time_range[0]
            
            # list of gestures that fit within the duration
            selected_gestures = self.gesture_selector.gesture_time_range(duration, self.frame_rate)
            start_time = time_range[0]
            
            # now i will Loop through each gesture and its timestamp within the selected gestures.
            for t, gesture in selected_gestures:
                self.frames_list.append([round(start_time + t, 2), gesture])

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
    base_dir = "static/uploads/3475c3e75ab03cd9a53b1bcc78193832/c358a7ac3ae73ebbaff56728f9da1f11_arabic-male"
    v = os.path.join(base_dir, "voice.wav")
    p = os.path.join(base_dir, "phonemes.csv")
    f = Frames(v, p, default_gestures="new_male")
    print(f.frames())