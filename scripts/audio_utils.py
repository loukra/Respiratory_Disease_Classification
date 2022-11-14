import numpy as np
import pandas as pd
import librosa
import spec2png
import sys
from IPython.display import display

def arr_pad(x, fs, length, mode="pre"):
    """adds zeros before or after a array until it got the desired length

    Args:
        x (_type_): 1-d numpy array
        fs (int): sampling rate of audio signal
        length (_type_): length after padding in seconds
        mode (str, optional): adding zeros before (pre) or after (post) the signal. Defaults to 'pre'.

    Returns:
        _type_: zero-padded 1-d numpy array
    """

    x_len = x.shape[0]  # length of input array
    y_len = fs * length  # length of array after padding
    # if padding is needed
    if y_len - x_len:
        # select mode
        if mode == "pre":
            pad_width = (y_len - x_len, 0)
        elif mode == "post":
            pad_width = (0, y_len - x_len)

        y = np.pad(x, pad_width)

    else:
        y = x
    return y


def arr_split(x, fs, length, annotation, overlap=0.5):
    """splits array into chunks of length fs * length

    Args:
        x (numpy.ndarray): 1d numpy array
        fs (int,float): sample rate of signal
        length (float): length of signal in seconds
        annotation (pandas DataFrame): annotation which have to be extended
        overlap (float, optional): overlap of chunks. 0 means no overlap. Defaults to 0.5.

    Returns:
        numpy.ndarray: 2d array with rows being the signal chunks
    """

    y_len = fs * length  # sample length of chunks
    split_start = np.arange(
        0, x.shape[0], int(y_len * (1 - overlap))
    )  # startings indices for chunks
    split_end = np.arange(
        int(y_len), x.shape[0], int(y_len * (1 - overlap))
    )  # stopping indices

    # if array not empty
    if split_end.size != 0:
        # check whether last stop index is smaller than last index of array
        if split_end[-1] < x.shape[0]:
            split_end = np.append(split_end, x.shape[0])
    else:
        split_end = np.append(split_end, x.shape[0])

    # match array sizes. one starting index pairs with one stopping index
    split_start = split_start[: split_end.shape[0]]

    # create output array
    y = np.zeros((split_end.shape[0], y_len))

    # fill output array with padded arrays
    for idx in range(y.shape[0]):
        y[idx, :] = arr_pad(x[split_start[idx] : split_end[idx]], fs, length)

    extend_annotation = pd.concat([annotation] * y.shape[0], ignore_index=True)

    return y, extend_annotation


def read_wav(filename, tar_sr=4000, verbose=False, console = False):
    """_read_wav_

    Args:
        filename (_type_): filename fetched from annotation.csv filename
        tar_sr (_type_): target sampling rate for output

    Returns:
        vec: time domain vec
        tar_sr: target sampling rate
    """
    if console:
        wav_path = "../data/sounds/" + filename
    else:
        wav_path = "../../data/sounds/" + filename
    ori_sr = librosa.get_samplerate(wav_path)  # save the original sampling rate
    vec, tar_sr = librosa.load(wav_path, sr=tar_sr)
    dur = vec.shape[0] / tar_sr

    if verbose == True:
        print(f"Original sr: {ori_sr}, Target sr: {tar_sr}, duration: {dur} sec")

    return vec, tar_sr


def mel_log(
    vec: np.ndarray,
    sr: int = 4000,
    n_mels: int = 50,
    n_fft: int = 512,
    fmax: int = None,
) -> np.ndarray:
    """_summary_

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


def audio_preprocessing(annotation, console = False, fs=4000, chunk_length=8, overlap=0.5):
    filename = annotation["filename"].values[0]
    y, _ = read_wav(filename, console = console)

    y, extended_annotations = arr_split(y, fs, chunk_length, annotation, overlap)
    return y, extended_annotations


def audio2img(
    annotation_dir="../../data/annotations.csv", console = False
):
    annotations = pd.read_csv(annotation_dir, index_col=0)

    display(annotations)
    X = []
    extended_annotations = pd.DataFrame(
        columns=["id", "diagnosis", "train_test", "filename", "is_healthy"]
    )

    for idx, row in annotations.iterrows():
        df = pd.DataFrame(row).T
        y, extended_annotation = audio_preprocessing(df, console = console)
        [X.append(col) for col in y]
        extended_annotations = pd.concat(
            [extended_annotations, extended_annotation], ignore_index=True
        )

    X = np.array(X)
    spec2png.spec2png(X, extended_annotations, hop_length=512, console = console)


if __name__ == "__main__":
    print(sys.argv[1])
    annot_path = sys.argv[1]
    audio2img(
    annotation_dir=annot_path, console = True)