import numpy as np
import os
import sys

import wave
import copy
import math

from sklearn.preprocessing import label_binarize
from tqdm import tqdm


from utils import *


batch_size = 64
nb_feat = 34
nb_class = 4
nb_epoch = 80

optimizer = 'Adadelta'


code_path = os.path.dirname(os.path.realpath(os.getcwd()))
#emotions_used = np.array(['ang', 'exc', 'neu', 'sad'])
data_path = "../IEMOCAP/"
sessions = ['Session1', 'Session2', 'Session3', 'Session4', 'Session5']
framerate = 16000

def get_mocap_rot(path_to_mocap_rot, filename, start,end):
    f = open(path_to_mocap_rot + filename, 'r').read()
    f = np.array(f.split('\n'))
    mocap_rot = []
    mocap_rot_avg = []
    f = f[2:]
    counter = 0
    for data in f:
        counter+=1
        data2 = data.split(' ')
        if(len(data2)<2):
            continue
        if(float(data2[1])>start and float(data2[1])<end):
            mocap_rot_avg.append(np.array(data2[2:]).astype(np.float))
            
    mocap_rot_avg = np.array_split(np.array(mocap_rot_avg), 200)
    for spl in mocap_rot_avg:
        mocap_rot.append(np.mean(spl, axis=0))
    return np.array(mocap_rot)

def get_mocap_hand(path_to_mocap_hand, filename, start,end):
    f = open(path_to_mocap_hand + filename, 'r').read()
    f = np.array(f.split('\n'))
    mocap_hand = []
    mocap_hand_avg = []
    f = f[2:]
    counter = 0
    for data in f:
        counter+=1
        data2 = data.split(' ')
        if(len(data2)<2):
            continue
        if(float(data2[1])>start and float(data2[1])<end):
            mocap_hand_avg.append(np.array(data2[2:]).astype(np.float))
            
    mocap_hand_avg = np.array_split(np.array(mocap_hand_avg), 200)
    for spl in mocap_hand_avg:
        mocap_hand.append(np.mean(spl, axis=0))
    return np.array(mocap_hand)

def get_mocap_head(path_to_mocap_head, filename, start,end):
    f = open(path_to_mocap_head + filename, 'r').read()
    f = np.array(f.split('\n'))
    mocap_head = []
    mocap_head_avg = []
    f = f[2:]
    counter = 0
    for data in f:
        counter+=1
        data2 = data.split(' ')
        if(len(data2)<2):
            continue
        if(float(data2[1])>start and float(data2[1])<end):
            mocap_head_avg.append(np.array(data2[2:]).astype(np.float))
            
    mocap_head_avg = np.array_split(np.array(mocap_head_avg), 200)
    for spl in mocap_head_avg:
        mocap_head.append(np.mean(spl, axis=0))
    return np.array(mocap_head)



def read_iemocap_mocap():
    data = []
    ids = {}
    for session in tqdm(sessions, "Processing Sessions" ):
        path_to_wav = data_path + session + '/dialog/wav/'
        path_to_emotions = data_path + session + '/dialog/EmoEvaluation/'
        path_to_transcriptions = data_path + session + '/dialog/transcriptions/'
        path_to_avi = data_path + session + '/dialog/avi/DivX/'

        files2 = os.listdir(path_to_wav)

        files = []
        for f in files2:
            if f.endswith(".wav"):
                if f[0] == '.':
                    files.append(f[2:-4])
                else:
                    files.append(f[:-4])
                    

        for f in tqdm(files, desc=f"Processing {session} files", leave=False):
            print(f)
            mocap_f = f
            if (f== 'Ses05M_script01_1b'):
                mocap_f = 'Ses05M_script01_1' 
            
            wav = get_audio(path_to_wav, f + '.wav')
            avi = get_avi(path_to_avi, f + '.avi')
            transcriptions = get_transcriptions(path_to_transcriptions, f + '.txt')
            emotions = get_emotions(path_to_emotions, f + '.txt')
            sample = split_wav(wav, emotions, data_path=data_path)
            avi_sample = split_avi(avi, emotions, data_path=data_path)

            for ie, e in enumerate(emotions):
                id = e['id']
                direction = "right" if id[5] != id[-4] else "left"
                e['audio'] = sample[ie] # contains path to .npy file and metadate
                e['video'] = avi_sample[ie] # contains path to .npy file and metadate
                e['transcription'] = transcriptions[e['id']]
            if e['id'] not in ids:
                data.append(e)
                ids[e['id']] = 1

                        
    sort_key = get_field(data, "id")
    return np.array(data)[np.argsort(sort_key)]
    
data = read_iemocap_mocap()

import pickle
with open(data_path + '/./'+'data_collected.pickle', 'w') as handle:
    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
