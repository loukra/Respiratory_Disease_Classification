
import os
import joblib
from tqdm import tqdm
from PIL import Image
import numpy as np
from skimage.transform import resize


def resize_all(src, pklname, include, width=150, height=None):
    """
    load images from path, resize them and write them as arrays to a dictionary, 
    together with labels and metadata. The dictionary is written to a pickle file 
    named '{pklname}_{width}x{height}px.pkl'.
     
    Parameter
    ---------
    src: str
        path to data
    pklname: str
        path to output file
    width: int
        target width of the image in pixels
    include: set[str]
        set containing str
    """
     
    height = height if height is not None else width
     
    data = dict()
    data['description'] = 'resized ({0}x{1})animal images in rgb'.format(int(width), int(height))
    data['label'] = []
    data['filename'] = []
    data['data'] = []   
     
    pklname = f"{pklname}_{width}x{height}px.pkl"
 
    # read all images in PATH, resize and write to DESTINATION_PATH
    for subdir in os.listdir(src):
        if subdir in include:
            print(subdir)
            current_path = os.path.join(src, subdir)
 
            for file in tqdm(os.listdir(current_path)):
                if file[-3:] in {'jpg', 'png'}:
                    img = Image.open(os.path.join(current_path, file))
                    img_rgb = img.convert('RGB') # convert to RGB

                    im = np.asarray(img_rgb)
                    im = resize(im, (width, height)) #[:,:,::-1]
                    data['label'].append(subdir[-1])
                    data['filename'].append(file)
                    data['data'].append(im)
 
        joblib.dump(data, pklname)