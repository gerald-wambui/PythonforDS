import numpy as np
from functools import partial
import PIL.Image
import tensorflow as tf
import urllib.request
import os
import zipfile


def main():
    # download google's pretrained model
    url = "https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip"
    data_dir = '../data'
    model_name = os.path.split(url)[-1]
    local_zip_file = os.path.join(data_dir, model_name)

    if not os.path.exists(local_zip_file):
       # download
    model_url = urllib.request.urlopen(url)
    with open(local_zip_file, 'wb') as output:
        output.write(model_url.read())

    with zipfile.ZipFile(local_zip_file, 'r') as zip_ref:
        zip_ref.extractall(data_dir)


    model_fn = 'tensorflow_inception_graph.pb'
    '''
    
    create tensorflow session
    
    Load the model
    
    Initialize the graph
    
    '''
    graph = tf.Graph()
    sess = tf.InteractiveSession(graph=graph)

