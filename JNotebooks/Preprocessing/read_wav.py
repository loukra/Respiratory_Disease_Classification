"""read_wav
    function reads wav file
    OUTPUT:
    vector
    sr -> sampling rate
"""

import librosa

def _read_wav_(filename, tar_sr=4000):
    """_read_wav_

    Args:
        filename (_type_): filename fetched from annotation.csv filename
        tar_sr (_type_): target sampling rate for output

    Returns:
        _type_: _description_
    """
    wav_path = "../../data/sounds/"+filename
    ori_sr = librosa.get_samplerate(wav_path) # save the original sampling rate
    vec, tar_sr = librosa.load(wav_path, sr=tar_sr)
    print(f'Original sr: {ori_sr}, Target sr: {tar_sr}, ')
    return vec, tar_sr




