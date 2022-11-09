"""read_wav
    function reads wav file
    OUTPUT:
    vector
    sr -> sampling rate
"""

import librosa

def _read_wav_(filename, tar_sr=4000, verbose=False):
    """_read_wav_

    Args:
        filename (_type_): filename fetched from annotation.csv filename
        tar_sr (_type_): target sampling rate for output

    Returns:
        vec: time domain vec
        tar_sr: target sampling rate
    """
    wav_path = "../../data/sounds/"+filename
    ori_sr = librosa.get_samplerate(wav_path) # save the original sampling rate
    vec, tar_sr = librosa.load(wav_path, sr=tar_sr)
    dur = vec.shape[0]/tar_sr

    if verbose == True:
        print(f'Original sr: {ori_sr}, Target sr: {tar_sr}, duration: {dur} sec')
    
    return vec, tar_sr

