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
    with tf.gfile.FastGFile(os.path.join(data_dir, model_fn), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    t_input = tf.placeholder(np.float32, name='input') #input tensor
    imagenet_mean = 117.0
    t_preprocessed = tf.expand_dims(t_input-imagenet_mean, 0)
    tf.import_graph_def(graph_def, {'input': t_preprocessed})

    layers = [op.name for op in graph.get_operations() if op.type=='Conv2D' and 'import/' in op.name]
    feature_nums = [int(graph.get_tensor_by_name(name+':0').get_shape()[-1]) for name in layers]
    print('Number of layers', len(layers))
    print('Total number of feature channels:', sum(feature_nums))

    layer = 'mixed4d_3x3_bottleneck_pre_relu'
    channel = 139

    img0 = PIL.Image.open('pilatus800.jpg')
    img0 = np.float32(img0)

