import imageio
from os import listdir
from os.path import isfile, join
import os
import cv2

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def giffify(directory):
    images = []
    im_paths = sorted([directory + "/" + f for f in listdir(directory) if isfile(join(directory, f))])
    print("[INFO] Gathering Images")
    for im in im_paths:
        images.append(cv2.resize(imageio.imread(im), (960,540)))
    print("[INFO] Saving Image Sequence as gif")
    v_path = directory + "/output.gif"
    ensure_dir(v_path)
    imageio.mimsave(v_path, images, duration=1.0)