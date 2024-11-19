import os
import csv
import wave
import sys
import numpy as np
import pandas as pd
import glob
import cv2
from decord import VideoReader
from decord import cpu
from sklearn.preprocessing import label_binarize

sys.path.append(os.path.dirname(os.path.realpath(__file__)))


########################################################################################################################
#                                                 constants                                                            #
########################################################################################################################


class Constants:
    def __init__(self):
        real_path = os.path.dirname(os.path.realpath(__file__))
        self.available_emotions = np.array(['ang', 'exc', 'neu', 'sad', 'hap', 'fru', 'fea', 'sur', 'dis', 'oth'])
        self.emotion_to_id = {emo: i for i, emo in enumerate(self.available_emotions)}
        self.id_to_emotion = {i: emo for i, emo in enumerate(self.available_emotions)}
        self.path_to_data = real_path + "/../../data/sessions/"
        self.path_to_features = real_path + "/../../data/features/"
        self.sessions = ['Session1', 'Session2', 'Session3', 'Session4', 'Session5']
        self.conf_matrix_prefix = 'iemocap'
        self.framerate = 16000
        self.types = {1: np.int8, 2: np.int16, 4: np.int32}
    
    def __str__(self):
        def display(objects, positions):
            line = ''
            for i in range(len(objects)):
                line += str(objects[i])
                line = line[:positions[i]]
                line += ' ' * (positions[i] - len(line))
            return line
        
        line_length = 100
        ans = '-' * line_length
        members = [attr for attr in dir(self) if not callable(attr) and not attr.startswith("__")]
        for field in members:
            objects = [field, getattr(self, field)]
            positions = [30, 100]
            ans += "\n" + display(objects, positions)
        ans += "\n" + '-' * line_length
        return ans


########################################################################################################################
#                                                 data reading                                                         #
########################################################################################################################


def get_audio(path_to_wav, filename, params=Constants()):
    wav = wave.open(path_to_wav + filename, mode="r")
    (nchannels, sampwidth, framerate, nframes, comptype, compname) = wav.getparams()
    content = wav.readframes(nframes)
    samples = np.fromstring(content, dtype=params.types[sampwidth])
    return (nchannels, sampwidth, framerate, nframes, comptype, compname), samples

def get_avi(path_to_avi, filename):
    full_path = path_to_avi + filename
    with open(full_path, 'rb') as f:
        vr = VideoReader(f, ctx=cpu(0))

    framerate = vr.get_avg_fps()
    frame_count = len(vr)
    (height, width, channel) = vr[0].shape
    return (framerate, frame_count, width, height), vr

def get_transcriptions(path_to_transcriptions, filename, params=Constants()):
    f = open(path_to_transcriptions + filename, 'r').read()
    f = np.array(f.split('\n'))
    transcription = {}
    for i in range(len(f) - 1):
        g = f[i]
        i1 = g.find(': ')
        i0 = g.find(' [')
        ind_id = g[:i0]
        ind_ts = g[i1+2:]
        transcription[ind_id] = ind_ts
    return transcription

def emotion_to_distribution(emotions, params=Constants()):
    size = len(emotions)
    ratio = 1.0 / size
    distribution = [0.0] * len(params.available_emotions)
    for e in emotions:
        distribution[params.emotion_to_id[e]] += ratio
    return distribution

def emotion_to_hard_label(emotions, params=Constants()):
    emotion_count = {emo: 0 for emo in params.available_emotions}

    for emo in emotions:
        if emo in emotion_count:
            emotion_count[emo] += 1
    
    max_count = max(emotion_count.values())

    most_common_emo = [emo for emo, count in emotion_count.items() if count == max_count]

    
    hard_label = []
    for emo in most_common_emo:
        emoh = [0.0]*len(params.available_emotions)
        emoh[params.emotion_to_id[emo]] = 1.0
        hard_label.append(emoh)
        
    return hard_label

        
    

def get_emotions(path_to_emotions, filename, params=Constants()):
    f = open(path_to_emotions + filename, 'r').read()
    f = np.array(f.split('\n'))
    idx = f == ''
    idx_n = np.arange(len(f))[idx]
    emotion = []
    for i in range(len(idx_n) - 2):
        g = f[idx_n[i]+1:idx_n[i+1]]
        head = g[0]
        i0 = head.find(' - ')
        start_time = float(head[head.find('[') + 1:head.find(' - ')])
        end_time = float(head[head.find(' - ') + 3:head.find(']')])
        actor_id = head[head.find(filename[:-4]) + len(filename[:-4]) + 1:
                        head.find(filename[:-4]) + len(filename[:-4]) + 5]
        emo = head[head.find('\t[') - 3:head.find('\t[')]
        vad = head[head.find('\t[') + 1:]

        v = float(vad[1:7])
        a = float(vad[9:15])
        d = float(vad[17:23])
        
        j = 1
        emos = []
        emoh = []
        while g[j][0] == "C":
            head = g[j]
            start_idx = head.find("\t") + 1
            evoluator_emo = []
            idx = head.find(";", start_idx)
            while idx != -1:
                evoluator_emo.append(head[start_idx:idx].strip().lower()[:3])
                start_idx = idx + 1
                idx = head.find(";", start_idx)
            emos.append(evoluator_emo)
            emoh.append(evoluator_emo)
            j += 1
        flattened = [item for sublist in emos for item in sublist]
        for emo in flattened:
            if emo not in params.available_emotions:
                flattened.remove(emo)
        emos = emotion_to_distribution(flattened)
        emoh = emotion_to_hard_label(flattened)        

        emotion.append({'start': start_time,
                        'end': end_time,
                        'id': filename[:-4] + '_' + actor_id,
                        'v': v,
                        'a': a,
                        'd': d,
                        'emotion': emo,
                        'emo_evo': emos,
                        'emo_hard_label': emoh})
    return emotion


def split_wav(wav, emotions, params=Constants(), data_path=None, output_dir=None):
    """
    Splits the wav file into segments and saves as .npy files
    Returns metadata including paths to saved audio segments
    """
    if output_dir is None:
        output_dir = os.path.join(data_path, "processed_audio")
    os.makedirs(output_dir, exist_ok=True)

    (nchannels, sampwidth, framerate, nframes, comptype, compname), samples = wav
    left = samples[0::nchannels]
    right = samples[1::nchannels]

    segments_info = []
    for ie, e in enumerate(emotions):
        start = e['start']
        end = e['end']
        id = e['id']
        
        # Determine which channel to use based on speaker ID
        direction = "right" if id[5] != id[-4] else "left"
        audio_data = right if direction == "right" else left
        
        # Extract segment
        start_idx = int(start * framerate)
        end_idx = int(end * framerate)
        segment = audio_data[start_idx:end_idx]
        
        # Save as .npy file
        npy_path = os.path.join(output_dir, f"{id}.npy")
        np.save(npy_path, segment)
        
        segments_info.append({
            'audio_path': npy_path,
            'n_samples': len(segment),
            'framerate': framerate,
            'sampwidth': sampwidth,
            'direction': direction
        })
        
        # Clear memory
        del segment

    return segments_info

def split_avi(avi, emotions, params=Constants(),batch_size=32,data_path=None,output_dir=None):
    """
    splits the avi file into segments based on the emotion annotations, each frame size is 360x240x3
    """
    if output_dir is None:
        output_dir = os.path.join(data_path, "processed_videoframes")
    os.makedirs(output_dir, exist_ok=True)

    (framerate, frame_count, width, height), vr = avi

    segments_info = []

    for ie, e in enumerate(emotions):
        start = e['start']
        end = e['end']
        id = e['id']
        direction = "right" if id[5] != id[-4] else "left"

        # Set crop dimensions based on direction
        crop_x = 360 if direction == "right" else 0
        crop_y, crop_w, crop_h = 120, 360, 240

        # Calculate frame indices
        start_frame_idx = int(start * framerate)
        end_frame_idx = int(end * framerate)
        start_frame_idx = max(0, min(start_frame_idx, frame_count - 1))
        end_frame_idx = max(0, min(end_frame_idx, frame_count))
        
        if start_frame_idx >= end_frame_idx:
            continue
        
        # Get all frames for this segment
        frame_indices = list(range(start_frame_idx, end_frame_idx))
        frames = vr.get_batch(frame_indices).asnumpy()
        
        # Process frames
        processed_frames = []
        for frame in frames:
            frame = frame[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w, :]
            
            if direction == "left":
                frame = frame.astype(float)
                frame[..., 2] *= 0.77  # Reduce red channel
                frame = np.clip(frame, 0, 255).astype(np.uint8)
            
            processed_frames.append(frame)
        
        processed_frames = np.array(processed_frames)
        
        # Save as .npy file
        npy_path = os.path.join(output_dir, f"{id}.npy")
        np.save(npy_path, processed_frames)
        
        segments_info.append({
            'frames_path': npy_path,
            'n_frames': len(processed_frames),
            'frame_size': (crop_h, crop_w),
            'framerate': framerate
        })
        
        # Clear memory
        del processed_frames
        del frames
    
    del vr
    return segments_info


def read_iemocap_data(params=Constants()):
    data = []
    for session in params.sessions:
        path_to_wav = params.path_to_data + session + '/dialog/wav/'
        path_to_emotions = params.path_to_data + session + '/dialog/EmoEvaluation/'
        path_to_transcriptions = params.path_to_data + session + '/dialog/transcriptions/'

        files = os.listdir(path_to_wav)
        files = [f[:-4] for f in files if f.endswith(".wav")]
        for f in files:           
            wav = get_audio(path_to_wav, f + '.wav')
            transcriptions = get_transcriptions(path_to_transcriptions, f + '.txt')
            emotions = get_emotions(path_to_emotions, f + '.txt')
            sample = split_wav(wav, emotions)

            for ie, e in enumerate(emotions):
                e['signal'] = sample[ie]['left']
                e.pop("left", None)
                e.pop("right", None)
                e['transcription'] = transcriptions[e['id']]
                if e['emotion'] in params.available_emotions:
                    data.append(e)
    sort_key = get_field(data, "id")
    return np.array(data)[np.argsort(sort_key)]


########################################################################################################################
#                                                 features generation                                                  #
########################################################################################################################


def get_features(data, params=Constants()):
    excluded = 0
    overall = 0
    IPy = False
    if "IPython.display" in sys.modules.keys():
        from IPython.display import clear_output
        IPy = True
    for di, d in enumerate(data):
        if di % 100 == 0:
            if IPy:
                clear_output()
            print(di, ' out of ', len(data))
        st_features = calc_feat.calculate_features(d['signal'], params.framerate, None).T
        x = []
        y = []
        for f in st_features:
            overall += 1
            if f[1] > 1.e-4:
                x.append(f)
                y.append(d['emotion'])
            else:
                excluded += 1
        x = np.array(x, dtype=float)
        y = np.array(y)
        save_sample(x, y, params.path_to_features + d['id'] + '.csv')
    return overall, excluded


########################################################################################################################
#                                                 helpers                                                              #
########################################################################################################################


def get_field(data, key):
    return np.array([e[key] for e in data])


def to_categorical(y, params=Constants()):
    return label_binarize(y, params.available_emotions)


def save_sample(x, y, name):
    with open(name, 'w') as csvfile:
        w = csv.writer(csvfile, delimiter=',')
        for i in range(x.shape[0]):
            row = x[i, :].tolist()
            row.append(y[i])
            w.writerow(row)

            
def load_sample(name):
    with open(name, 'r') as csvfile:
        r = csv.reader(csvfile, delimiter=',')
        x = []
        y = []
        for row in r:
            x.append(row[:-1])
            y.append(row[-1])
    return np.array(x, dtype=float), np.array(y)


def get_sample(ids, take_all=False, params=Constants()):
    if take_all:
        files = os.listdir(params.path_to_features)
        ids = np.sort([f[:-4] for f in files if f.endswith(".csv")])
    tx = []
    ty = []
    valid_ids = []
    for i in ids:
        x, y = load_sample(params.path_to_features + i + '.csv')
        if len(x) > 0:
            tx.append(np.array(x, dtype=float))
            ty.append(y[0])
            valid_ids.append(i)
    tx = np.array(tx)
    ty = np.array(ty)
    return tx, ty, np.array(valid_ids)


def pad_sequence_old(x, ts, params=Constants()):
    xp = []
    for i in range(len(x)):
        x0 = np.zeros((ts, x[i].shape[1]), dtype=float)
        if ts > x[i].shape[0]:
            x0[ts - x[i].shape[0]:, :] = x[i]
        else:
            maxe = np.sum(x[i][0:ts, 1])
            x0 = x[i][0:ts, :]
            for j in range(x[i].shape[0] - ts):
                if np.sum(x[i][j:j + ts, 1]) > maxe:
                    x0 = x[i][j:j + ts, :]
                    maxe = np.sum(x[i][j:j + ts, 1])
        xp.append(x0)
    return np.array(xp)


def pad_sequence_into_array(Xs, maxlen=None, truncating='post', padding='post', value=0.):
    """
    Padding sequence (list of numpy arrays) into an numpy array
    :param Xs: list of numpy arrays. The arrays must have the same shape except the first dimension.
    :param maxlen: the allowed maximum of the first dimension of Xs's arrays. Any array longer than maxlen is truncated to maxlen
    :param truncating: = 'pre'/'post', indicating whether the truncation happens at either the beginning or the end of the array (default)
    :param padding: = 'pre'/'post',indicating whether the padding happens at either the beginning or the end of the array (default)
    :param value: scalar, the padding value, default = 0.0
    :return: Xout, the padded sequence (now an augmented array with shape (Narrays, N1stdim, N2nddim, ...)
    :return: mask, the corresponding mask, binary array, with shape (Narray, N1stdim)
    """
    Nsamples = len(Xs)
    if maxlen is None:
        lengths = [s.shape[0] for s in Xs]    # 'sequences' must be list, 's' must be numpy array, len(s) return the first dimension of s
        maxlen = np.max(lengths)

    Xout = np.ones(shape=[Nsamples, maxlen] + list(Xs[0].shape[1:]), dtype=Xs[0].dtype) * np.asarray(value, dtype=Xs[0].dtype)
    Mask = np.zeros(shape=[Nsamples, maxlen], dtype=Xout.dtype)
    for i in range(Nsamples):
        x = Xs[i]
        if truncating == 'pre':
            trunc = x[-maxlen:]
        elif truncating == 'post':
            trunc = x[:maxlen]
        else:
            raise ValueError("Truncating type '%s' not understood" % truncating)
        if padding == 'post':
            Xout[i, :len(trunc)] = trunc
            Mask[i, :len(trunc)] = 1
        elif padding == 'pre':
            Xout[i, -len(trunc):] = trunc
            Mask[i, -len(trunc):] = 1
        else:
            raise ValueError("Padding type '%s' not understood" % padding)
    return Xout, Mask


def convert_gt_from_array_to_list(gt_batch, gt_batch_mask=None):
    """
    Convert groundtruth from ndarray to list
    :param gt_batch: ndarray (B, L)
    :param gt_batch_mask: ndarray (B, L)
    :return: gts <list of size = B>
    """
    B, L = gt_batch.shape
    gt_batch = gt_batch.astype('int')
    gts = []
    for i in range(B):
        if gt_batch_mask is None:
            l = L
        else:
            l = int(gt_batch_mask[i, :].sum())
        gts.append(gt_batch[i, :l].tolist())
    return gts


########################################################################################################################
#                                               metrics                                                                #
########################################################################################################################


def weighted_accuracy(y_true, y_pred):
    return np.sum((np.array(y_pred).ravel() == np.array(y_true).ravel()))*1.0/len(y_true)


def unweighted_accuracy(y_true, y_pred):
    y_true = np.array(y_true).ravel()
    y_pred = np.array(y_pred).ravel()
    classes = np.unique(y_true)
    classes_accuracies = np.zeros(classes.shape[0])
    for num, cls in enumerate(classes):
        classes_accuracies[num] = weighted_accuracy(y_true[y_true == cls], y_pred[y_true == cls])
    return np.mean(classes_accuracies)