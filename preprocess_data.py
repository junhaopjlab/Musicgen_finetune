import h5py
import numpy as np

from 



"""
data = [
    {audio_id: , audio_input:,  audio_input_sr: , audio_label:, audio_label_sr: }
]
"""

def paths_map(audio_id):
    paths ={
        'mp3': os.path.join(SPLIT_PATH, audio_id +'.mp3'),
        'acc': os.path.join(API_PATH, audio_id+'_acc.wav'),
        'vocal': os.path.join(API_PATH, audio_id+'.wav'),
        'midi': os.path.join(API_PATH, audio_id+'_wav2midi.json'),
        'lyric_txt': os.path.join(SPLIT_PATH, audio_id +'.txt'),
        'lyric_time': os.path.join(API_PATH, audio_id+'_lyrictime.txt'),
        'dtw':os.path.join(DTW_PATH, audio_id+'_align.json'),
        'ace_file': os.path.join(ACE_PATH, audio_id+'.aces'),
        'ace_audio': os.path.join(ACE_PATH, audio_id+'.wav')
    }  
    return paths



def collect_data_hd():

