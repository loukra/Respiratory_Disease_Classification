import numpy as np
from librosa.feature import melspectrogram
from librosa import power_to_db



def array2D_to_rgb(array:np.ndarray, eps=1e-6) -> np.ndarray:
    """array2D_to_rgb

    Args:
        array (np.ndarray): a 2D numpy array, (height, width)
        eps (float, optional): to avoid 0 std error.

    Returns:
        np.ndarray: (height, width, 3) numpy 3D array
    """
    
    array = np.stack((array, array, array), axis=-1)
    # Standardize
    mean = array.mean()
    std = array.std()
    array = (array - mean) / (std + eps)

    # Normalize to [0, 255]
    _min, _max = array.min(), array.max()

    if (_max - _min) > eps:
        V = np.clip(array, _min, _max)
        V = 255 * (V - _min) / (_max - _min)
        V = V.astype(np.uint8)
    else:
        V = np.zeros_like(array, dtype=np.uint8)

    return V # return shape (H,W,3)



def mel_log(vec:np.ndarray, 
            sr: int=4000,
            n_mels: int=50,
            n_fft: int=512, 
            fmax: int=None) -> np.ndarray:
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
    
