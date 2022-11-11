# make sure to run mk_dir.py file to make the folder structure first.add()

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

from librosa.feature import melspectrogram
from librosa.display import specshow
from librosa import power_to_db, get_samplerate, load

def save_png(y:np.ndarray, anno_chunk: pd.DataFrame, sr: int=4000, bi:bool=True):
    """saves chunks from one record to image, file name: cls_pid_chunk_No.

    Args:
        y (np.ndarray): shape (no_chunk, 32000)
        anno_chunk (pd.DataFrame): annotation of no_chunks of one audio record
        sr (int, optional): target sampling rate. Defaults to 4000.
        bi (bool): binary, related to path of image saving. Default->True

    Returns:
        1 when all chunk image stor
    """
    for idx in range(y.shape[0]):
        if bi:
            test_train = anno_chunk['train_test'][idx]
            heal = "health_" + str(anno_chunk['is_healthy'][idx])
            chunk_num = str(anno_chunk.index[idx]+1)

            path = os.path.join(_ws_dir(), "cls_2", test_train, heal)
            
            if not os.path.exists(path): # mkdir if the folder 
                os.makedirs(path)
             
            filename = str(anno_chunk['is_healthy'][idx]) + '_' \
                            +anno_chunk['id'][idx]+ '_'  \
                            + chunk_num
            filepath = os.path.join(path,filename)

            arr = _mel_log(y[idx])
            plt.axis('off')  # no axis
            plt.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])
            specshow(arr, sr=sr,fmax=sr/2) 
            plt.savefig(filepath,  bbox_inches="tight", pad_inches=0)

        else:
            pass # save for multi-classification

    return 1


def _mel_log(vec:np.ndarray, 
            sr: int=4000,
            n_mels: int=50,
            n_fft: int=512, 
            fmax: int=None) -> np.ndarray:

    """FFT mel and with default 512 window size. Mel bins:50; Log scale.

    Args:
        vec (np.ndarray): column vector
        sr (int, optional): sampling rate. Defaults to 4000.
        n_mels (int, optional): number of mel bin. Defaults to 50.
        n_fft (int, optional):FFT win size. Defaults to 512.
        fmax (int, optional): max frequency range. Defaults: tar_sr/2.

    Returns:
        np.ndarray: FFT mel spectrogram, 2D array
    """
    mel = melspectrogram(y=vec, sr=sr, n_fft=n_fft, fmax=fmax, n_mels=n_mels)
    mel_dB = power_to_db(mel, ref=np.max)

    return mel_dB


def _ws_dir(folder:str="Respiratory_Disease_Classification/"):
    # return the absolute data/image directory on local
    ab_dir = os.getcwd()
    return ab_dir.split(folder, 1)[0]+folder+"data/images"