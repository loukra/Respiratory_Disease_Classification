import numpy as np
import pandas as pd


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

    extend_annotation = pd.concat([annotation] * y.shape[0], ignore_index = True)

    return y, extend_annotation