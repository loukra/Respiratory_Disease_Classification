

def save_png(filename:str, arr:np.ndarray):
    """save_png

    Args:
        filename (str): name of the file
        arr (np.ndarray): vectors of the chunk
    """


    plt.axis('off')  # no axis
    plt.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])
    librosa.display.specshow(mel_dB,
                         sr=tar_sr,
                         fmax=tar_sr/2)                
    plt.savefig(filename,  bbox_inches="tight", pad_inches=0)