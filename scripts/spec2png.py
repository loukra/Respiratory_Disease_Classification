import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

from librosa.feature import melspectrogram
from librosa.display import specshow
from librosa import power_to_db, get_samplerate, load

def spec2png(y:np.ndarray, 
             anno_chunk: pd.DataFrame,
             sr: int=4000, 
             hop_length: int=512,
             window: str='hann',
             bi:bool=True):
    """saves chunks from one record to image, file name: cls_pid_chunk_No.

    Args:
        y (np.ndarray): shape (no_chunk, 32000)
        anno_chunk (pd.DataFrame): annotation of no_chunks of one audio record
        sr (int, optional): target sampling rate. Defaults to 4000.
        hop_length (int): default=512
        bi (bool): binary classification. Default->True
        window (str,optional): see scipy.signal.get_window. Defaults: "hann"


    Returns:
        1 when all chunk image stored 
    """
    for idx in range(y.shape[0]):
        print("-----spec2png progress: {idx}/{y.shape[0]}-----")
        anno_row = anno_chunk.iloc[idx]
        chunk_num = str(anno_chunk.index[idx]+1)

        filepath = _gen_path(anno_row) + chunk_num # generate the file path, append chunk No.

        arr = _mel_log(y[idx], hop_length=hop_length, window=window) 

        plt.axis('off')  # no axis
        plt.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])
        specshow(arr, sr=sr,fmax=sr/2) 
        plt.savefig(filepath,  bbox_inches="tight", pad_inches=0)

    return 1


def _mel_log(vec:np.ndarray,
            window: str,
            hop_length:int,
            sr: int=4000,
            n_mels: int=50,
            n_fft: int=512, 
            fmax: int=None
            ) -> np.ndarray:

    """FFT mel and with default 512 window size. Mel bins:50; Log scale.

    Args:
        vec (np.ndarray): column vector
        sr (int, optional): sampling rate. Defaults to 4000.
        hop_length (int): default 32
        n_mels (int, optional): number of mel bin. Defaults to 50.
        n_fft (int, optional):FFT win size. Defaults to 512.
        fmax (int, optional): max frequency range. Defaults: tar_sr/2.
        win (str,optional):see scipy.signal.get_window. Defaults: "hann"
    Returns:
        np.ndarray: FFT mel spectrogram, 2D array
    """

    mel = melspectrogram(y=vec, sr=sr, n_fft=n_fft, fmax=fmax,
                        n_mels=n_mels, hop_length=hop_length, window=window)
    mel_dB = power_to_db(mel, ref=np.max)

    return mel_dB


def _gen_path(anno_row: pd.DataFrame,
              folder:str="Respiratory_Disease_Classification/",
              bi:bool=True) -> str:
    """generate filepath to save img file, without chunk No. appended

    Args:
        anno_row: annotation row of the chunk
        folder (str, optional): change if you have different folder name than the GitHub name. Defaults:"Respiratory_Disease_Classification/".
        bi (bool, optional): False if multiclassification. Defaults to True.

    Returns:
        str: path/file str for plt.savefig() method
    """
    ab_dir = os.getcwd()
    img_dir = ab_dir.split(folder, 1)[0]+folder+"data/images"
    if bi:
            test_train = anno_row['train_test']
            heal = "health_" + str(anno_row['is_healthy'])

            # path example: data/images/cls_2/test/health_0
            path = os.path.join(img_dir, "cls_2", test_train, heal)
            
            if not os.path.exists(path): # mkdir if the folder does not exist
                os.makedirs(path)
             
            filename = str(anno_row['is_healthy']) + '_' \
                            +anno_row['id']+ '_'
            filepath = os.path.join(path,filename)
    else:
        pass # for multi-classification
    return filepath
