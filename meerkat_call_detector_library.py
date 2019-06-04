#LIBRARY OF FUNCTIONS, CLASSES, ETC FOR MEERKAT CALL DETECTOR AND CLASSIFIER

#Import stuff

from scipy.io import wavfile
from sklearn.preprocessing import MinMaxScaler, normalize
import matplotlib.pyplot as plt
import numpy as np
import skimage.measure
import scipy.signal as spy
import scipy
import os
import pickle
import glob
from keras.models import Model
from keras.layers import Conv1D, Conv2D, AveragePooling1D, AveragePooling2D, MaxPooling1D, MaxPooling2D, Activation, Input, Add, UpSampling1D, UpSampling2D, Concatenate, BatchNormalization, GlobalAveragePooling2D, Dense
from keras.optimizers import RMSprop
from keras.models import load_model
from keras.callbacks import History
import datetime
import librosa
import csv
import pandas
import re
import keras.layers as layers

#------------------------------ CLASSES -------------------------------
class CallExtractionParams:
  def __init__(self, model_path, wav_path, pckl_path, samprate, t_start, t_end, step = .256, pad = 256, chunk_size = 60, chunk_pad = 1):
    self.model_path = model_path
    self.wav_path = wav_path
    self.samprate = samprate
    self.t_start = t_start
    self.t_end = t_end
    self.step = step
    self.pad = pad
    self.chunk_size = chunk_size
    self.chunk_pad = chunk_pad
    self.pckl_path = pckl_path
    
class CallExtractionOutput:
  def __init__(self, call_extraction_params, times, scores):
    self.call_extraction_params = call_extraction_params
    self.times = times
    self.scores = scores
    
class CallClassificationOutput:
  def __init__(self, call_extraction_params, times, scores, call_times, pred_call_types, pred_call_scores, seg_thresh, seg_boundary_thresh, classification_model):
    self.call_extraction_params = call_extraction_params
    self.times = times
    self.scores = scores
    self.call_times = call_times
    self.pred_call_types = pred_call_types
    self.pred_call_scores = pred_call_scores
    self.seg_thresh = seg_thresh
    self.seg_boundary_thresh = seg_boundary_thresh
    self.classification_model = classification_model
    
class EvaluationOutput:
  def __init__(self, thresh_range,boundary_thresh,precisions,recalls_foc,recalls_non,true_pos_foc,true_pos_non,false_neg_foc,false_neg_non,pckl_path,call_types):
    self.threshes = thresh_range
    self.boundary_thresh = boundary_thresh
    self.precisions = precisions
    self.recalls_foc = recalls_foc
    self.recalls_non = recalls_non
    self.true_pos_foc = true_pos_foc
    self.true_pos_non = true_pos_non
    self.false_neg_foc = false_neg_foc
    self.false_neg_non = false_neg_non
    self.pckl_path = pckl_path
    self.call_types = call_types

#------------------------------ FUNCTIONS ------------------------------

#TRAINING

def generate_sample_call(files, clips_dir, call_type = None, verbose=False):
    
    """Generate a sample call and label from a list of files in a directory. For detection step.
    
    Parameters:
    files (list of strings): list of filenames of audio clips (name only, not path)
    clips_dir (string): directory where clips are stored
    call_type (string): string decribing the call type to select 
        default = None will select a call uniformly at random from the files list
    verbose (bool): whether to print statements about progress
        default = False
        
    Returns:
    X (512 x 128 np matrix float): spectrogram image of call, size currently hard-coded
    y (len 512 np array): vector of 1's and 0's where 1's are placed where calls are. For now dimensions hardcoded to 512, with each "pixel" being 1 ms
    
    """
    

    if(verbose):
        print('generating call noaug')
        
    #hard coded parameters for now
    samprate=8000
    pad_len = 127

    #get indexes to files of type call_type (or sample from all calls if call_type is None)
    if(call_type is not None):
        idxs = np.where([((call_type in s) & ('_aud' in s)) for s in files])[0]
    else:
        idxs = np.where(['_aud' in s for s in files])[0]
    idx = np.random.choice(idxs)

    #get audio file and mask file
    aud_file = files[idx]
    mask_file = aud_file[0:(len(aud_file) - 7)] + 'mask.npy'
    
    if(verbose):
        print(aud_file)

    #read in both files
    aud = np.load(clips_dir + '/' + aud_file)
    mask = np.load(clips_dir + '/' + mask_file)

    len_call = len(aud) - samprate*2
    if(len_call < 4096):
        offset = np.random.randint(samprate - 4096 + len_call,samprate)
    else:
        offset = samprate

    _,_,spec = spy.spectrogram(aud[(offset-pad_len):(offset+4096+pad_len)],fs=samprate,nperseg=255,noverlap=247,window='hanning')
    spec_norm = np.log(spec)
    #spec_norm = spec_norm - np.min(spec_norm) / (np.max(spec_norm) - np.min(spec_norm))
    dim_spec = spec.shape[1]
    dim_wav = aud.shape[0]
    t0_wav = samprate
    tf_wav = dim_wav - samprate
    t0_spec = int(t0_wav * float(dim_spec) / dim_wav)
    tf_spec = int(tf_wav * float(dim_spec) / dim_wav)
    call_length = tf_spec - t0_spec
    pad_length = (4096 - call_length)//2
    X = np.transpose(spec_norm)
    y = mask[(offset):(offset+4096)]
    y = y.reshape((len(y),1))
    y = skimage.measure.block_reduce(y, (8,1), np.max)
    return X, y

def generate_sample_call_augment(files, clips_dir, call_type = None, verbose=False):

    """Generate a sample call and label from a list of files in a directory. Perform basic augemntation by adding noise from another section of the recording (currently drawn from beginning of another clip file, excluding instances where calls are in the selected region). The fraction noise vs call is currently selected uniformly at random from the interval [0,1]. For detection step.
    
    Parameters:
    files (list of strings): list of filenames of audio clips (name only, not path)
    clips_dir (string): directory where clips are stored
    call_type (string): string decribing the call type to select 
        default = None will select a call uniformly at random from the files list
    verbose (bool): whether to print statements about progress
        default = False
        
    Returns:
    X (512 x 128 np matrix float): spectrogram image of call, size currently hard-coded
    y (len 512 np array): vector of 1's and 0's where 1's are placed where calls are. For now dimensions hardcoded to 512, with each "pixel" being 1 ms
    aud_sub (len 4096 + pad_len*2 vector): augmented audio sequence
    aud_sub_noise (len 4096 + pad_len*2 vector): noise used for augmentation
    
    """
    
    if(verbose):
        print('generating call augmented')
    #hard coded parameters for now
    samprate=8000
    pad_len = 127

    #get indexes to files of type call_type (or sample from all calls if call_type is None)
    if(call_type is not None):
        idxs = np.where([((call_type in s) & ('_aud' in s)) for s in files])[0]
    else:
        idxs = np.where(['_aud' in s for s in files])[0]
    idx = np.random.choice(idxs)

    #get audio file and mask file
    aud_file = files[idx]
    mask_file = aud_file[0:(len(aud_file) - 7)] + 'mask.npy'
    
    if(verbose):
        print(aud_file)

    #read in both files
    aud = np.load(clips_dir + '/' + aud_file)
    mask = np.load(clips_dir + '/' + mask_file)

    #get small chunk
    len_call = len(aud) - samprate*2
    if(len_call < 4096):
        offset = np.random.randint(samprate - 4096 + len_call,samprate)
    else:
        offset = samprate
    aud_sub = aud[(offset-pad_len):(offset+4096+pad_len)]

    #Augmentation with noise from another file
    aug_ratio = np.random.random(1)

    #get noise file
    aug_found = False
    while(not(aug_found)):
        idx_noise = np.random.choice(idxs)
        aud_file_noise = files[idx_noise]
        aud_noise = np.load(clips_dir + '/' + aud_file_noise)
        mask_file_noise = aud_file_noise[0:(len(aud_file_noise)-7)] + 'mask.npy'
        mask_noise = np.load(clips_dir + '/' + mask_file_noise)
    if(np.sum(mask_noise[0:len(aud_sub)])==0):
        aug_found = True
    aud_sub_noise = aud_noise[0:len(aud_sub)]

    #add noise
    aud_sub = aud_sub + aud_sub_noise

    _,_,spec = spy.spectrogram(aud_sub,fs=samprate,nperseg=255,noverlap=247,window='hanning')
    spec_norm = np.log(spec)
    dim_spec = spec.shape[1]
    dim_wav = aud.shape[0]
    t0_wav = samprate
    tf_wav = dim_wav - samprate
    t0_spec = int(t0_wav * float(dim_spec) / dim_wav)
    tf_spec = int(tf_wav * float(dim_spec) / dim_wav)
    call_length = tf_spec - t0_spec
    pad_length = (4096 - call_length)//2
    X = np.transpose(spec_norm)
    y = mask[(offset):(offset+4096)]
    y = y.reshape((len(y),1))
    y = skimage.measure.block_reduce(y, (8,1), np.max)
    
    return X, y, aud_sub, aud_sub_noise

def generate_sample_noise(files,clips_dir, verbose=False):
    
    """Generate a sample of noise and label from a list of files in a directory. Noise is currently sampled from the regions nearby regions to calls (i.e. from files that contain 1 sec + call + 1 sec). TODO: Possibly should change this later to sample from other file regions. For detection step.
    
    Parameters:
    files (list of strings): list of filenames of audio clips (name only, not path)
    clips_dir (string): directory where clips are stored
    call_type (string): string decribing the call type to select 
        default = None will select a call uniformly at random from the files list
    verbose (bool): whether to print statements about progress
        default = False
        
    Returns:
    X (512 x 128 np matrix float): spectrogram image of call, size currently hard-coded
    y (len 512 np array): vector of 1's and 0's where 1's are placed where calls are. For now dimensions hardcoded to 512, with each "pixel" being 1 ms
    
    """
    
    if(verbose):
        print('generating noise')
    #hard coded parameters for now
    samprate=8000
    pad_len = 127

    #get indexes to files of type call_type
    idxs = np.where([('_aud' in s) for s in files])[0]
    idx = np.random.choice(idxs)

    #get audio file and mask file
    aud_file = files[idx]
    mask_file = aud_file[0:(len(aud_file) - 7)] + 'mask.npy'
    
    if(verbose):
        print(aud_file)

    #read in both files
    aud = np.load(clips_dir + '/' + aud_file)
    mask = np.load(clips_dir + '/' + mask_file)

    len_call = len(aud) - samprate*2
    offset = 300
    _,_,spec = spy.spectrogram(aud[(offset-pad_len):(offset+4096+pad_len)],fs=samprate,nperseg=255,noverlap=247,window='hanning')
    spec_norm = np.log(spec)
    #spec_norm = spec_norm - np.min(spec_norm) / (np.max(spec_norm) - np.min(spec_norm))
    dim_spec = spec.shape[1]
    dim_wav = aud.shape[0]
    t0_wav = samprate
    tf_wav = dim_wav - samprate
    t0_spec = int(t0_wav * float(dim_spec) / dim_wav)
    tf_spec = int(tf_wav * float(dim_spec) / dim_wav)
    call_length = tf_spec - t0_spec
    pad_length = (4096 - call_length)//2
    X = np.transpose(spec_norm)
    y = mask[(offset):(offset+4096)]
    y = y.reshape((len(y),1))
    y = skimage.measure.block_reduce(y, (8,1), np.max)
    return X, y

def generate_batch(batch_size,clips_dir,augment = False,call_types = ['cc','sn','ld','mov','agg','alarm','soc','hyb','unk','oth'],call_probs = None, p_noise = 0.5, cnn_dim = 2, verbose=False):
    
    """Generate a batch of sample calls and labels from audio clips in a directory. For detection step.
    
    Parameters:
    batch_size (int): how many samples to generate
    clips_dir (string): directory where clips are stored
    augment (bool): whether to augment by adding noise to calls (defaults to False)
    call_types (list of strings): list of call types. Defaults to meerkat call types: ['cc','sn','ld','mov','agg','alarm','soc','hyb','unk','oth']
    call_probs (vector of floats, same length as call_types): sampling probabilities for the different call types
        default = None: in this case call types will be sampled in proportion to their occurrence in clips_dir
    p_noise (float): probability of drawing noise instead of a call 
        default = 0.5
    cnn_dim (int): CNN dimesnion (can be 1 or 2)
        default = 2 for 2d convolutions (this works better)
    verbose (bool): whether the print progress as function runs
        
    Returns:
    X (batch_size x 512 x 128 x 1 np array of floats): batch of call and noise spectrograms (if cnn_dim == 1, last dimension will be removed)
    y (batch_size x 512 x 1 np array): labels (if cnn_dim == 1, last dimension will be removed). vectors of 1's and 0's where 1's are placed where calls are. For now dimensions hardcoded to 512, with each "pixel" being 1 ms
    
    """
    
    files = os.listdir(clips_dir)
    X_list = []
    y_list = []
    
    if(call_probs is not None):
        call_cumprobs = np.cumsum(call_probs)
    for idx in range(batch_size):
        if(np.random.rand(1)<=p_noise):
            X, y = generate_sample_noise(files, clips_dir,verbose=verbose) 
        else:
            if call_probs is not None:
                r = np.random.random(1)
                idx = np.where(call_cumprobs > r)[0][0]
                if(augment):
                    X, y, asub, asubn = generate_sample_call_augment(files,clips_dir,call_types[idx],verbose=verbose) 
                else:
                    X, y = generate_sample_call(files,clips_dir,call_types[idx],verbose=verbose)
            else:
                if(augment):
                    X, y, asub, asubn = generate_sample_call_augment(files,clips_dir,call_type=None,verbose=verbose)
                else:
                    X, y = generate_sample_call(files,clips_dir,call_type=None,verbose=verbose)
        if(cnn_dim == 2):
            X = X.reshape((X.shape[0],X.shape[1],1))
            y = y.reshape((y.shape[0],y.shape[1],1))
        X_list.append(X)
        y_list.append(y)
    X = np.stack(X_list)
    y = np.stack(y_list)
    return (X, y)
 
#Data generators 
def data_generator(clips_dir,batch_size=10,augment=False,call_types = ['cc','sn','ld','mov','agg','alarm','soc','hyb','unk','oth'],call_probs = None,p_noise = 0.5, cnn_dim = 2, verbose=False):
    """Wrapper function to generate data in batches. For detection.
    
    Parameters:
    clips_dir (str): path to directory where clips are stored
    batch_size (int): how many samples per batch
        default = 10
    augment (bool): whether to perform augmentation
        default = False
    call_types (list of strings): list of call types
        default = ['cc','sn','ld','mov','agg','alarm','soc','hyb','unk','oth']
    call_probs (list of floats): probabilities of drawing each call types
        default = None (calls will be drawn at random from the clips in clips_dir)
    p_noise (float): probability of drawing noise
        default = 0.5
    cnn_dim (int): convolution dimension (1 or 2)
        default = 2
    verbose (bool): whether to print progress
        default = False
        
    Returns:
        a data generator from generate_batch
    """
    
    while True:
        yield generate_batch(batch_size,clips_dir,augment,call_types,call_probs,p_noise,cnn_dim,verbose=verbose)

        
 #MODEL CONSTRUCTION

#1d code - working (for creating models)
def conv_pool(inputs, filters, n_convs=3):
    conv = inputs
    for idx in range(n_convs):
        conv = Conv1D(filters, (3), padding='same')(conv)
        conv = Activation('relu')(conv)
    conv = AveragePooling1D()(conv)

    return conv

def conv_upsample(inputs, residual, filters):
    conv = Conv1D(filters, (3), padding='same')(inputs)

    residual = Conv1D(filters, (1))(residual)
    conv = Add()([conv,residual])
    conv = Activation('relu')(conv)
    conv = UpSampling1D()(conv)

    return conv

def construct_unet_model(lr = 2.5e-4):
    """Construct a U-net model with 1D convolutions, for inputs sized 512 x 128
    
    Parameters:
    lr (float): learning rate
        default = 2.5e-4
        
    Returns:
    model (a U-net model using 1D convoluutions, 5 layers)
    """
    #input_layer = Input(batch_shape=(None,None,None))
    input_layer = Input(batch_shape=(None,512,128))
    conv = conv_pool(input_layer, 32)
    res1 = conv
    outputs = []
    
    #could modify to more layers (more than 5)
    for idx in range(5):
        conv = conv_pool(conv, 32*(idx+1))
        outputs.append(conv)

    for idx in range(5):
        conv = conv_upsample(conv, outputs[-(idx+1)], 32*(5-idx))

    conv = conv_upsample(conv, res1,  32)

    #fully connected layer equivalent
    conv = Conv1D(1, (3), padding='same')(conv)
    conv = Activation('sigmoid')(conv)

    model = Model(input_layer, conv)

    #change optimizer to ADAM?
    model.compile(RMSprop(lr=lr), loss='binary_crossentropy')
  
    return model

#2d CNN functions
def conv_pool_2d(inputs, filters, n_convs=3):
    conv = inputs
    for idx in range(n_convs):
        conv = Conv2D(filters, (3,3), padding='same')(conv)
        conv = Activation('relu')(conv)
    conv = AveragePooling2D()(conv)

    return conv

def conv_upsample_2d(inputs, residual, filters):
    conv = Conv2D(filters, (3,3), padding='same')(inputs)

    residual = Conv2D(filters, (1,1))(residual)
    conv = Add()([conv,residual])
    conv = Activation('relu')(conv)
    conv = UpSampling2D()(conv)

    return conv

def construct_unet_model_2d(lr = 2.5e-4):
    """Construct a U-net model with 2D convolutions, for inputs sized 512 x 128
    
    Parameters:
    lr (float): learning rate
        default = 2.5e-4
        
    Returns:
    model (a U-net model using 2D convoluutions, 5 layers)
    """
    
    
    input_layer = Input(batch_shape=(None,512,128,1))
    conv = conv_pool_2d(input_layer, 32)
    res1 = conv
    outputs = []
    for idx in range(5):
        conv = conv_pool_2d(conv, 32*(idx+1))
        outputs.append(conv)

    for idx in range(5):
        conv = conv_upsample_2d(conv, outputs[-(idx+1)], 32*(5-idx))

    conv = conv_upsample_2d(conv, res1,  32)

    # reduce the channels to 1 using a 1x1 2D convolution
    conv = layers.Conv2D(1, (1,1), padding='same')(conv)
    # conv should be shape (512,128,1) at this point
    # reshape to remove the last dimension so we can use 1D convolution
    conv = layers.Reshape((512,128))(conv)
    # reduce the last dimension to 1 using a 1x 1D convolution
    logits = layers.Conv1D(1, 1, padding='same')(conv)
    # logits should be shape (512,1)
    probs = layers.Activation('sigmoid')(logits)
    probs = layers.Reshape((512,1,1))(probs)


    model = Model(input_layer, probs)
    model.compile(RMSprop(lr=lr), loss='binary_crossentropy')
  
    return model

#MODEL TESTING

def generate_cnn_input(wav_data,samprate=8000,pad=256, cnn_dim = 2):

  """Generate a spectrogram of the right size (512 x 128) to put into the neural net
  Parameters:
  wav_data (np 1d array): raw data from a wav file, read in using librosa
  samprate (int): sample rate (Hz)
      default = 8000
  pad (int): padding on either side (in spectrogram generation)
      default = 256
  cnn_dim (int): conv net dimension (1 or 2)
      default = 2
  
  Returns:
  X (np array 512 x 128 x 1): a properly formatted spectrogram image for input into the CNN (note if cnn_dim == 1, last dim will be removed)

  """
  #create spectrogram, normalize, and crop
  _,_,spec = spy.spectrogram(wav_data,fs=samprate,nperseg=255,noverlap=247,window='hanning')
  spec_norm = np.log(spec)
  #spec_norm = spec_norm - np.min(spec_norm) / (np.max(spec_norm) - np.min(spec_norm))
  dim_spec = spec.shape[1]
  dim_wav = wav_data.shape[0]
  t0_wav = pad
  tf_wav = dim_wav - pad
  t0_spec = pad/2/8 + 1
  tf_spec = pad/2/8 + 512 + 1
  X = np.transpose(spec_norm[:,int(t0_spec):int(tf_spec)])

  #add a dimension if doing 2d convolutions (these expect another dimension b/c images usually have 3 channels)
  if(cnn_dim==2):
    X = np.reshape(X,(X.shape[0],X.shape[1],1))

  #return sample
  return X

def extract_scores(model, extraction_params, cnn_dim=2):
  print('-------Running CNN model on new data-------')
  print('model_path: '+ str(extraction_params.model_path))
  print('wav_path: ' + str(extraction_params.wav_path))
  print('samprate: ' + str(extraction_params.samprate))
  print('t_start: ' + str(extraction_params.t_start))
  print('t_end: '+ str(extraction_params.t_end))
  print('')

  #Set up time range and results matrices
  trange = np.arange(extraction_params.t_start,extraction_params.t_end,extraction_params.step)
  results = np.empty((len(trange),512))

  #Generate predictions
  trange = np.arange(extraction_params.t_start,extraction_params.t_end,extraction_params.step) # range of times
  chunk_starts = np.arange(extraction_params.t_start,extraction_params.t_end,extraction_params.chunk_size) #times to start read-in chunks

  print('--------Generating predictions-------')
  print("Start time:")
  print(datetime.datetime.now())

  #Read in audio chunks, make spectrograms, and run model on them, output results
  prev_chunk_idx = -1
  for i in range(len(trange)):
    t0 = trange[i]
    curr_chunk_idx = np.amax(np.where(chunk_starts <= t0))

    #if needed, read in a new chunk from the wav file
    if prev_chunk_idx != curr_chunk_idx:
      prev_chunk_idx = curr_chunk_idx
      curr_wav, sr = librosa.core.load(extraction_params.wav_path,sr=extraction_params.samprate,offset=chunk_starts[curr_chunk_idx]-extraction_params.chunk_pad,duration=extraction_params.chunk_size+2*extraction_params.chunk_pad)

    #get indexes to start and end of sample within the current chunk
    t0_within_chunk = int((extraction_params.chunk_pad+t0)*sr-chunk_starts[curr_chunk_idx]*sr - extraction_params.pad) #place to start within chunk
    tf_within_chunk = int(t0_within_chunk + 4096 + 2*extraction_params.pad)

    #get wav data for the CNN
    wav_data = curr_wav[t0_within_chunk:tf_within_chunk]

    #Generate a properly formatted sample (for now 512 x 128 size)
    X = generate_cnn_input(wav_data,pad=extraction_params.pad,samprate=extraction_params.samprate)

    #Run prediction model and save to results matrix
    result = model.predict(X[None,...])
    if(cnn_dim==1):
        result = result[0,:,0]
    if(cnn_dim==2):
        result = result[0,:,0,0]
    results[i,:] = result

  print("End time:")
  print(datetime.datetime.now())
  print()

  print('-------Generating scores and extracting calls-------')

  #Take results of clips and turn them into a single continuous string of "scores" (0-1)
  tbin = 4096./extraction_params.samprate/512
  times = np.arange(trange[0],trange[len(trange)-1]+4096./extraction_params.samprate,tbin)
  scores = np.zeros(shape=len(times))
  for i in range(len(trange)):
    t_idx = int(i*extraction_params.step*512.*extraction_params.samprate/4096)
    scores[np.arange(t_idx,t_idx+512,1)] = np.maximum(results[i,:],scores[np.arange(t_idx,t_idx+512,1)])

  print('-------Saving output-------')

  #Save to an output file
  f = open(extraction_params.pckl_path, 'wb')
  output = CallExtractionOutput(extraction_params, times, scores)
  pickle.dump(output, f)
  f.close()

  #Print final time
  print('Done')
    
def segment_calls(times,scores,min_thresh,boundary_thresh):
    """Segment scores into calls using a double threshold method.
    
    The function proceeds by finding instances in which the score exceeds one threshold (min_thresh). It then searches forward and backwards from these instances to find the boundaries of calls, defined as the nearest time (forward and back) when the score falls below a second threshold (boundary_thresh)
    
    Parameters:
    times (1d np array of floats): vector holding times associated with scores
    scores (1d np array of floats): vector of scores (0-1)
    min_thresh (float): minimum threshold a region of scores must cross in order to be segmented into a call (upper threshold)
    boundary_thresh (float): threshold that scores must cross in order to segment into a separate call (lower threshold)
    
    Returns:
    calls (n_calls x 2 numpy array): holds times of call beginnings and endings
    """
    #get boundaries so don't try to access out of bounds indexes
    min_idx = 0
    max_idx = len(scores)
    calls = list()
    i = min_idx
    while i < max_idx:
        if scores[i] >= min_thresh:
            fwd = 1
            back = 1
            while ((i+fwd) < max_idx) & (scores[i + fwd] >= boundary_thresh):
                fwd = fwd + 1
            while ((i-back) > min_idx) & (scores[i - back] >= boundary_thresh):
                back = back + 1
            calls.append((times[i-back], times[i+fwd]))
            i = i + fwd
        else:
            i = i + 1

    calls = np.reshape(calls,(len(calls),2))
    return(calls)

def generate_call_labels_audition(calls,times,scores,csv_name,classifs=None):
    """Generate a csv file (formatted to be read into audition) with predicted calls and their scores
    
    Parameters:
    calls (n_calls x 2 numpy array): array of call beginning and end times
    times (1d numpy array): vector of times for the audio file
    scores (1d numpy array): scores between 0 and 1 for each time step (same length as times) 
    csv_name (str): what to name the output csv (full path)
    classifs (1d numpy array of strs or None): predicted classifications
        default = None (in this case don't label with classification predictions)
    
    Returns:
    Nothing (saves output file to csv_name)
    
    """
    with open(csv_name, mode='w') as csv_file:
        writer = csv.writer(csv_file, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['Name','Start','Duration','Time Format','Type','Description'])
        for i in range(calls.shape[0]):
            idx0 = np.where(times == calls[i,0])[0][0]
            idxf = np.where(times == calls[i,1])[0][0]
            if(classifs==None):
                name = str(np.max(scores[idx0:idxf]))
            else:
                name = classifs[i] + ' (' + str(np.max(scores[idx0:idxf])) + ')'
            start = str(datetime.timedelta(seconds=calls[i,0]))
            dur = str(datetime.timedelta(seconds=calls[i,1]-calls[i,0]))
            writer.writerow([name,start,dur,'decimal','Cue',''])
    return
      
#EVALUATION FUNCS

def label_type(label_str):
  """Generate label type from label string
  
  Parameters:
  label_str (str): label name
  
  Returns:
  lab_type (str): label type (one of: 'synch', 'start', 'end', 'call')
  
  """
  if ((re.search('SYNC',label_str,re.IGNORECASE)!=None) | (re.search('BEEP',label_str,re.IGNORECASE)!=None) | (re.search('SYC',label_str,re.IGNORECASE)!=None)):
    lab_type = 'synch'
  elif ((re.search('STAR',label_str,re.IGNORECASE)!=None) | (re.search('SART',label_str,re.IGNORECASE)!=None)):
    lab_type = 'start'
  elif ((re.search('END',label_str,re.IGNORECASE)!=None)):
    lab_type = 'end'
  else:
    lab_type = 'call'
  return lab_type

#Call labels
def call_type_simple(label_str):
    
  """Get simple call type from full label
  Parameters: 
  label_str (str): full name of label
  
  Returns
  lab_type (str) simplified call label (one of 10 types: 'hyb','cc','mov','ld','soc','sn','unk','alarm','agg',oth')
  
  """
  if ((re.search('HYB',label_str,re.IGNORECASE)!=None)):
    lab_type = 'hyb'
  elif ((re.search('LEAD CC',label_str,re.IGNORECASE)!=None)):
    lab_type = 'hyb'
  elif ((re.search('CC',label_str,re.IGNORECASE)!=None) | (re.search('Marker',label_str,re.IGNORECASE)!=None)):
    lab_type = 'cc'
  elif ((re.search('MO',label_str,re.IGNORECASE)!=None)):
    lab_type = 'mov'
  elif ((re.search('LD',label_str,re.IGNORECASE)!=None) | (re.search('LEAD',label_str,re.IGNORECASE)!=None)):
    lab_type = 'ld'
  elif ((re.search('SOC',label_str,re.IGNORECASE)!=None)):
    lab_type = 'soc'
  elif ((re.search('SN',label_str,re.IGNORECASE)!=None)):
    lab_type = 'sn'
  elif ((re.search('UNK',label_str,re.IGNORECASE)!=None)):
    lab_type = 'unk'
  elif ((re.search('ALARM',label_str,re.IGNORECASE)!=None) | (re.search('ALERT',label_str,re.IGNORECASE)!=None)):
    lab_type = 'alarm'
  elif ((re.search('AGG',label_str,re.IGNORECASE)!=None) | (re.search('CHAT',label_str,re.IGNORECASE)!=None) | (re.search('GROWL',label_str,re.IGNORECASE)!=None) | (re.search('AGGRESS',label_str,re.IGNORECASE)!=None)):
    lab_type = 'agg'
  else:
    lab_type = 'oth'
  return lab_type

def caller(label_str):
    
  """Generate caller (focal or nonfocal) from call label
  
  Parameters: 
  label_str (str): label name
  
  Returns:
  lab_type: caller id (one of: 'foc','non','ambig')
  
  """
  if ((re.search('NONFOC',label_str,re.IGNORECASE)!=None)):
    lab_type = 'non'
  elif ((re.search('[*]',label_str,re.IGNORECASE)!=None)):
    lab_type = 'ambig'
  else:
    lab_type = 'foc'
  return lab_type

def precision_recall(groundtruth_calls,predicted_calls,call_types):
    
    """Generate precision and recall from groundtruth data and predicted calls
    
    Parameters:
    groundtruth_calls (pandas array): ground truth labels, generated from get_ground_truth_labels
    predicted_calls (n_calls x 2 np array): beginning and end times of predicted calls
    call_types (list of strings): call types to compute recalls for
    
    Returns:
    recalls_foc (1d array of floats, same len as call_types): recall measure for each call type, focal calls only
    recalls_non (1d array of floats, same len as call_types): recall measure for each call type, nonfoc calls only
    precision (float): overall precision (probability that a detection overlaps with a call of any type)
    tps_foc (1d array of floats, same len as call_types): true positives for each call type, for the focal
    fns_foc (1d array of floats, same len as call_types): false negatives (missed calls) for each call type, for the focal
    tps_non (1d array of floats, same len as call_types): true positives for each call type, for non-focal calls
    fns_non (1d array of floats, same len as call_types): false negatives (missed calls) for each call type, for non-focal calls
    fps (int): false positives (detected calls that are neither focal nor non-focal calls in the groundtruth data)
    """
    
    tps_foc = np.zeros((len(call_types))) #true positives, focal
    fns_foc = np.zeros((len(call_types))) #false negatives, focal
    tps_non = np.zeros((len(call_types))) #true positives, nonfocal
    fns_non = np.zeros((len(call_types))) #false negatives, nonfocal

    #Go through ground truth calls and see which ones overlap predictions
    for i in groundtruth_calls.index.values:
        #get index to the call type (within the call_types list)
        if(groundtruth_calls['call_type'][i] != None):
            call_type_idx = call_types.index(groundtruth_calls['call_type'][i])
        else:
            print(groundtruth_calls['Name'][i])

        foc = False
        if groundtruth_calls['focal'][i] == 'foc':
            foc = True

        #get start and end of the call
        start = groundtruth_calls['start_time'][i]
        end = groundtruth_calls['end_time'][i]

        #check if it overlaps a predicted call
        maxstart = np.maximum(predicted_calls[:,0],start)
        minend = np.minimum(predicted_calls[:,1],end)

        if(sum(maxstart <= minend) > 0):
            if foc:
                tps_foc[call_type_idx] = tps_foc[call_type_idx] + 1
            else:
                tps_non[call_type_idx] = tps_non[call_type_idx] + 1
        else:
            if foc:
                fns_foc[call_type_idx] = fns_foc[call_type_idx] + 1
            else:
                fns_non[call_type_idx] = fns_non[call_type_idx] + 1

    #go through detections and get false positives
    fps = 0
    for i in range(predicted_calls.shape[0]):
        start = predicted_calls[i,0]
        end = predicted_calls[i,1]

        maxstart = np.maximum(groundtruth_calls['start_time'],start)
        minend = np.minimum(groundtruth_calls['end_time'],end)

        if(sum(maxstart <= minend) == 0):
            fps = fps + 1
    
    #compute recall by call type and overall precision
    recalls_foc = np.empty((len(call_types)))
    recalls_non = np.empty((len(call_types)))
    for i in range(len(call_types)):
        if((tps_foc[i] + fns_foc[i]) > 0):
            recalls_foc[i] = float(tps_foc[i]) / (tps_foc[i] + fns_foc[i])
        if((tps_non[i] + fns_non[i]) > 0):
            recalls_non[i] = float(tps_non[i]) / (tps_non[i] + fns_non[i])   

    if(fps + sum(tps_foc + tps_non) > 0):
        precision = 1 - float(fps) / float(fps + sum(tps_foc + tps_non))
    else:
        precision = 0

    output = (recalls_foc, recalls_non, precision, tps_foc, fns_foc, tps_non, fns_non, fps)
    
    return output

def get_ground_truth_labels(wav_name,ground_truth_dir):
    """Generate pandas data frame of ground truth labels associated with an audio file wav_name
    
    Parameters:
    wav_name (str): name of wav file (not full path) for which a corresponding csv file of labels should be found
    ground_truth_dir (str): directory where csv ground truth labels should be stored (Audition format)
    
    Returns:
    labels (pandas data frame): data frame of ground truth labels
    """
    
    ground_truth_path = ground_truth_dir + '/' + wav_name[0:(len(wav_name)-4)] + '.csv'
    ground_truth_path = ground_truth_path.replace('_downsamp','')
    if(os.path.isfile(ground_truth_path)):
        pass
    else:
        ground_truth_path = ground_truth_dir + '/' + wav_name[0:(len(wav_name)-4)] + '.CSV'

    print(ground_truth_path)

    #if ground truth file exists, read it in, otherwise return None
    if(not(os.path.isfile(ground_truth_path))): 
        return None
    else:
        #Read in ground truth labels and convert to seconds
        labels = pandas.read_csv(ground_truth_path,delimiter='\t')
        labels['start_time'] = labels['Start'].map(lambda x: hms_to_seconds(x))
        labels['duration'] = labels['Duration'].map(lambda x: hms_to_seconds(x))
        labels['end_time'] = labels['start_time'] + labels['duration']

        #Add columns for label types
        labels['label_type'] = labels['Name'].apply(label_type)
        labels['call_type'] = labels['Name'].apply(call_type_simple)
        labels['focal'] = labels['Name'].apply(caller)
        
        return labels
    
def get_ground_truth_labels_megan(wav_name, ground_truth_path):
    """Generate pandas data frame of ground truth labels associated with an audio file wav_name. Generate from the focal recordings Megan sent over (not collar data)
    
    Parameters:
    wav_name (str): name of wav file (not full path) for which a corresponding csv file of labels should be found
    ground_truth_dir (str): directory where csv ground truth labels should be stored (Audition format)
    
    Returns:
    labels (pandas data frame): data frame of ground truth labels
    """
    filename = wav_name[0:(len(wav_name)-3)] + 'txt'

    #if ground truth file exists, read it in, otherwise return None
    if(not(os.path.isfile(ground_truth_path))): 
        print('no path to ground truth file found')
        return None
    else:
        #Read in ground truth labels for the current file
        labels = pandas.read_csv(ground_truth_path,delimiter=',')
        labels = labels[labels['filename']==filename]
        labels = labels[(labels['type']=='start') | (labels['type']=='end') | (labels['iscall']==True)]

        labels['label_type'] = labels['type']
        labels['label_type'][labels['iscall']==True] = 'call'
        labels['start_time'] = labels['t0']
        labels['end_time'] = labels['tf']
        labels['duration'] = labels['end_time'] - labels['start_time']
        labels['call_type'] = labels['type']
        labels['focal'] = 'foc'
        
    return labels
    
    
def get_start_end_time_labels(labels, focal_megan = False):
    """Get the start and end times of the labeled period
    
    Parameters:
    labels (pandas data frame): labels from Audition, generated by get_ground_truth_labels
    focal_megan (bool): whether labels are from collar data or from data megan sent 
        default = False
        
    Returns:
    start_time_labels (float): start time of the labeled period
    end_time_labels (float): end time of the labeled period
    """
    #Only include detections and labels where the detection / labeling times overlap
    
    if(focal_megan):
        start_time_labels = np.min(labels['start_time'])
        end_time_labels = np.max(labels['end_time'])
    else:
        start_time_labels = labels[labels['Name'] == 'START']['start_time'].values[0]
        end_time_labels = labels[labels['Name'] == 'END']['start_time'].values[0]
    
    return (start_time_labels, end_time_labels)      

def run_evaluation(pckl_path, thresh_range, save_dir, ground_truth_dir,call_types,boundary_thresh = 0.6, verbose = True, savename = None, ground_truth_path = None, foc_megan = False):
    
    """Run evaluation of detections vs ground truth data
    
    Parameters:
    pckl_path (str): path to the outputted pckl file containing scores from a given file
    thresh_range (1d np array): upper thresholds to use
    save_dir (str): path to directory where results should be saved
    ground_truth_dir (str): path to directory where ground truth labels are stored
    call_types (list of strings): list of call types
    boundary_thresh (float): lower threshold for double threshold method
        default = 0.6
    verbose (bool): whether to print progress
    savename (str or None): name of file to save results in
        default = None (generate a name in standard format automatically)
    ground_truth_path (str or None): path to where ground truth labels are stored
        default = None (this is only needed if foc_megan == True)
     foc_megan (bool): whether running eval on data from megan or collar data
         default = False (collar data)
         
    Returns:
    0, if successful, saves evaluation output in pckl file at pckl_path
    """
    
    if(not(os.path.isfile(pckl_path))):
        print('pckl path does not exist')
        return -1
        
    f = open(pckl_path,'rb')
    output = pickle.load(f,encoding='latin1')
    
    #get name to save file (if unspecified)
    if savename is None:
        pckl_name = os.path.basename(pckl_path)
        savename = save_dir + '/' + pckl_name[0:(len(pckl_name)-5)] + '_eval.pckl'
    
    wav_path = output.call_extraction_params.wav_path
    wav_name = os.path.basename(wav_path)
    
    #if not yet evaluated, run evaluation
    if(os.path.isfile(savename)): 
        print('Evaluation results already saved under name:')
        print(savename)
        return -2
    else:
        
        #Read in ground truth labels and convert to seconds
        if foc_megan:
            labels = get_ground_truth_labels_megan(wav_name=wav_name,ground_truth_path=ground_truth_path)
        else:
            labels = get_ground_truth_labels(wav_name=wav_name,ground_truth_dir=ground_truth_dir)
        
        print(labels)
        
        #if ground truth data is not available, print this and end
        if labels is None:
            print('No ground truth data available for the specified file')
            return -3
            
        else:

            #Only include detections and labels where the detection / labeling times overlap
            [start_time_labels, end_time_labels] = get_start_end_time_labels(labels, focal_megan=foc_megan)
            print(start_time_labels)
            print(end_time_labels)

            t0_predictions = output.times[0]
            tf_predictions = output.times[len(output.times)-1]

            t0_compare = max(start_time_labels,t0_predictions)
            tf_compare = min(end_time_labels,tf_predictions)

            #Remove non-calls
            groundtruth_calls = labels.loc[(labels['label_type']=='call') & (labels['start_time'] >= t0_compare) & (labels['end_time'] < tf_compare)]
            
            print(groundtruth_calls)

            #set up numpy arrays to store results
            recalls_foc = np.zeros((len(thresh_range),len(call_types)))
            recalls_non = np.zeros((len(thresh_range),len(call_types)))
            precisions = np.zeros((len(thresh_range)))
            true_pos_foc = np.zeros((len(thresh_range),len(call_types)))
            false_neg_foc = np.zeros((len(thresh_range),len(call_types)))
            true_pos_non = np.zeros((len(thresh_range),len(call_types)))
            false_neg_non = np.zeros((len(thresh_range),len(call_types)))
            false_pos = np.zeros((len(thresh_range)))

            print('Running evaluation for file: ' + pckl_path)
            for i in range(len(thresh_range)):
                thresh = thresh_range[i]

                #get predicted call times (and remove ones outside labeled time range)
                predicted_calls = segment_calls(output.times,output.scores,min_thresh = thresh, boundary_thresh = boundary_thresh)
                predicted_calls = predicted_calls[(predicted_calls[:,0] > t0_compare) & (predicted_calls[:,1] <= tf_compare),:]

                #Get true positives for each call type, false negatives for each call type, and false positives overall
                out = precision_recall(groundtruth_calls = groundtruth_calls,predicted_calls=predicted_calls,call_types = call_types)

                #store results in matrices
                recalls_foc[i,:] = out[0]
                recalls_non[i,:] = out[1]
                precisions[i] = out[2]
                true_pos_foc[i,:] = out[3]
                false_neg_foc[i,:] = out[4]
                true_pos_non[i,:] = out[5]
                false_neg_non[i,:] = out[6]
                false_pos[i] = out[7]

                if verbose:
                    print(str(i) + '/' + str(len(thresh_range)))
                    print('current threshold: ' + str(thresh))
                    print('current precision: ' + str(precisions[i]))
                    print('current recall:' + str(recalls_foc[i]))
            

            #Store eval output in object
            eval_output = EvaluationOutput(thresh_range,boundary_thresh,precisions,recalls_foc,recalls_non,true_pos_foc,true_pos_non,false_neg_foc,false_neg_non,pckl_path,call_types)
            fid = open(savename, 'wb')
            pickle.dump(eval_output, fid)
            fid.close()
            
            print('Evaluation completed and saved at ' + savename)
            return 0

#FUNCTIONS FOR CLASSIFICATION
#Funcs to generate data
def generate_sample_call_for_classif(files, clips_dir, max_size = 512, call_type = 'cc', call_types = ['cc','sn','ld','mov','agg','alarm','soc','hyb','unk','oth'], verbose=False):

    """Generate call spectrogram and type label. For classification.
    
    Parameters:
    files (list of strings): list of files in clips_dir (audio clips)
    clips_dir (str): directory where clips are stored
    max_size (int): maximum size of spectrogram image
        default = 512
    call_type (str): what type of call should be generated 
        default = 'cc'
    call_types (list of strs): list of all call types (their order determines the integer used as classif label)
        default = ['cc','sn','ld','mov','agg','alarm','soc','hyb','unk','oth']
    verbose (bool): whether to print progress
    
    Returns:
    X (max_size x 128 np array): spectrogram image, with 0 padding after end of call
    y (int): integer label for call type
    """
    
    #hard coded parameters for now
    samprate=8000
    pad_len = 127

    #get indexes to files of type call_type (or sample from all calls if call_type is None)
    if(call_type is not None):
        idxs = np.where([((call_type in s) & ('_aud' in s)) for s in files])[0]
    else:
        idxs = np.where(['_aud' in s for s in files])[0]
    
    len_call= 0
    while(len_call < 240):
        idx = np.random.choice(idxs)

        #get audio file
        aud_file = files[idx]
    
        #read in audio file
        aud = np.load(clips_dir + '/' + aud_file)
    
        #get call length
        len_call = len(aud) - samprate*2
    
    aud = aud[(samprate-pad_len):(samprate+pad_len + 4096)]
    
    #generate spectrogram
    _,_,spec = spy.spectrogram(aud,fs=samprate,nperseg=255,noverlap=247,window='hanning')
    #_,_,spec = spy.spectrogram(aud[(offset-pad_len):(offset+4096+pad_len)],fs=samprate,nperseg=255,noverlap=247,window='hanning')
    spec_norm = np.log(spec)
    
    #pad with 0's where no call is
    spec_norm[:,int(len_call/8000.*1000):spec_norm.shape[1]] = 0

    #generate matrix for input to classifier
    X = np.transpose(spec_norm)
    
    #call type as integer (index in call_types vector)
    y = call_types.index(call_type)
    
    return X, y

def generate_batch_for_classif(batch_size,clips_dir,call_types = ['cc','sn','ld','mov','agg','alarm','soc','hyb','unk','oth'],call_probs = None, verbose=False):
    
    """Generate a batch of calls and labels for classification.
    
    Parameters:
    batch_size (int): size of batch to generate
    clips_dir (str): path to directory where clips are stored
    call_types (list of strings): list of call types to use
        default = ['cc','sn','ld','mov','agg','alarm','soc','hyb','unk','oth']
    call_probs (list of floats or None): list of call probabilities (prob of drawing each type)
        default = None (draw each call type equally often)
        
    Returns:
    X (batch_size x 512 x 128 x 1 np array): stack of spectrograms
    y (np array of ints, len batch_size): call labels (integers) 
    """
    
    #list files
    files = os.listdir(clips_dir)
    
    #create empty lists to hold spectrograms (input) and call labels (output)
    X_list = []
    y_list = []
    
    #probability of drawing each call type can be specified or if not specified is assumed to be uniform across all call types
    if(call_probs is not None):
        call_cumprobs = np.cumsum(call_probs)
    else:
        ncalls = len(call_types)
        call_cumprobs = np.arange(0,1,step=1./ncalls)+1./ncalls #equally spaced bins for drawing probabilities
        
    #generate samples
    for idx in range(batch_size):
        
        #select call type
        r = np.random.random(1)
        idx = np.where(call_cumprobs > r)[0][0]
        call_type = call_types[idx]
        X, y = generate_sample_call_for_classif(files=files, clips_dir=clips_dir, call_type=call_type, call_types=call_types, verbose=verbose)
        
        #reshape for 2d conv
        X = X.reshape((X.shape[0],X.shape[1],1))
        #y = y.reshape((y.shape[0],y.shape[1],1))
        
        #append to list
        X_list.append(X)
        y_list.append(y)
    
    #convert to stack
    X = np.stack(X_list)
    y = np.stack(y_list)
    return (X, y)

#Data generators 
def data_generator_for_classif(clips_dir,batch_size=10,call_types = ['cc','sn','ld','mov','agg','alarm','soc','hyb','unk','oth'],call_probs = [1./7,1./7,1./7,1./7,1./7,1./7,1./7,0,0,0],verbose=False):
    """Data generator for classification task
    
    Parameters:
    clips_dir (str): directory where call clips are stored
    batch_size (int): batch size
        default = 10
    call_types (list of strings): list of call types
        default = ['cc','sn','ld','mov','agg','alarm','soc','hyb','unk','oth']
    call_probs (list of floats): probabilities of drawing each call type, must add to 1
        default = [1./7,1./7,1./7,1./7,1./7,1./7,1./7,0,0,0], i.e. don't train on hyb, unk, or oth calls
    verbose (bool): whether to print progress
    
    Returns:
    a data generator for classification training
    """
    
    while True:
        yield generate_batch_for_classif(batch_size,clips_dir,call_types,call_probs,verbose=verbose)

def classifier_model(n_classes=10):
    
    """Generate a CNN model for classifying calls into types.
    
    Parameters:
        n_classes: number of classes to output
            default = 10
        
    Returns:
        model: a CNN model
    """
    
    inputs = Input(shape=(None, None,1))
    x = inputs
    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = MaxPooling2D()(x)

    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = MaxPooling2D()(x)

    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D()(x)

    # this converts feature maps to 1D feature vectors by taking the channelwise average
    # Also see GlobalMaxPooling2D and Flatten layers, which also transform to 1D vectors
    # GlobalAveragePooling2D has some nice properties in terms of model regularization,
    # which typically makes it the best choice for this transformation
    x = GlobalAveragePooling2D()(x)  
    x = Dense(64, activation='relu')(x)

    class_probabilites = Dense(n_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=class_probabilites)
    
    return(model)

#read in labels from Audition format file and pre-process
def read_labels_for_classif(test_csv):
    
    """Read in labels for classification from a csv
    
    Parameters:
    test_csv (str): path to a csv file containing labels (Audition format)
    
    Returns:
    labels (pandas data frame): data frame containing call labels
    """
    
    labels = pandas.read_csv(test_csv,delimiter='\t')
    labels['t0'] = [hms_to_seconds(x) for x in labels['Start']]
    labels['dur'] = [hms_to_seconds(x) for x in labels['Duration']]
    labels['tf'] = labels['t0'] + labels['dur']
    labels['label_type'] = [label_type(x) for x in labels['Name']]
    labels['call_type'] = [call_type_simple(x) for x in labels['Name']]
    labels['caller'] = [caller(x) for x in labels['Name']]
    foc = labels['caller'] == 'foc'
    call = labels['label_type'] == 'call'
    labels['classify'] = foc & call
    
    return labels

def generate_test_data_for_classif(labels,test_wav,pad_len=127, samprate=8000,win_len=4096,call_types=['cc','sn','ld','mov','agg','alarm','soc','hyb','unk','oth'],verbose=True):
    
    """Generate a stack of test data for the classification of calls
    
    Parameters:
    labels (pandas data frame): data frame containing call ground truth labels (generated from read_labels_for_classif)
    test_wav (str): path to a wav file to draw calls from
    pad_len (int): length of padding for spectrogram generation
        default = 127
    samprate (int): sample rate
        default = 8000
    win_len (int): window length
        default = 4096
    call_types (list of strings): list of call types
        default = ['cc','sn','ld','mov','agg','alarm','soc','hyb','unk','oth']
    verbose (bool): whether to print progress
    
    Returns:
    X (n_calls x 512 x 128 x 1 np array): stack of spectrograms to use as test data
    y (np array of len n_calls): integer labels for the test data call types
    """
    
    #create empty lists to hold spectrograms (input) and call labels (output)
    X_list = []
    y_list = []

    if(verbose):
        print('generating spectrograms for testing')
    
    #generate stack of data
    for i in range(len(labels['Name'])):

        #extract audio
        if(labels['classify'][i] == True):
            t0 = labels['t0'][i]
            dur = labels['dur'][i]
            real = labels['call_type'][i]
            aud, _ = librosa.core.load(path=test_wav,offset=t0-float(pad_len)/samprate,duration=float(win_len+2*pad_len)/samprate,sr=samprate)

            #convert to spectrogram
            _,_,spec = spy.spectrogram(aud,fs=samprate,nperseg=255,noverlap=247,window='hanning')
            spec_norm = np.log(spec)

            #pad with 0's where no call is
            spec_norm[:,int(dur*1000):spec_norm.shape[1]] = 0

            #generate matrix for input to classifier
            X = np.transpose(spec_norm)
            X = X.reshape((X.shape[0],X.shape[1],1))
            X_list.append(X)

            #call type as integer (index in call_types vector)
            y = labels['call_type'][i]
            y = call_types.index(y)
            y_list.append(y)

    #convert to stacks
    X = np.stack(X_list)
    y = np.stack(y_list)
    
    return(X,y)

def evaluate_classif(model,X,y,call_types):
    
    """Evaluation classification accuracy and generate confusion matrix
    
    Parameters:
    model: classification model
    X (n_calls x 512 x 128 x 1 np array): data in the form of spectrograms
    y (np array of len n_calls): numeric labels of call types
    call_types (list of strings): list of call types
    
    Returns:
    confusion (n_calls x n_calls np array): a confusion matrix
    """
    
    out = model.predict(X)
    preds = np.zeros(out.shape[0],dtype='int')
    for i in range(out.shape[0]):
        preds[i] = int(np.argmax(out[i,:]))
    match = preds == y

    confusion = np.zeros((len(call_types),len(call_types)))
    for i in range(confusion.shape[0]):
        for j in range(confusion.shape[1]):
            confusion[i,j] = np.sum((y == i) & (preds == j))
    
    return(confusion)

def plot_confusion_matrix(confusion,call_types,logscale=True,normalize_by_reals = True, normalize_by_preds = False):
    
    """Plot confusion matrix
    
    Parameters:
    confusion (n_calls x n_calls np array of ints): confusion matrix
    call_types (list of strings): call labels associated with matrix indexes
    logscale (bool): whether to use log scale for the colors on the plot
        default = True
    normalize_by_reals (bool): whether to normalize colors so that real call types add to 1
    normalize_by_preds (bool): whether to normalize colors so that predicted call types add to 1
    
    Returns:
    fig: a confusion matrix figure
    """
    
    fig = plt.figure()
    plt.xticks(np.arange(start=0,stop=len(call_types)),call_types,rotation='vertical')
    plt.yticks(np.arange(start=0,stop=len(call_types)),call_types)
    plt.ylabel('real')
    plt.xlabel('predicted')
    to_plot = confusion
    if(normalize_by_reals):
        to_plot = normalize(confusion,axis=1,norm='l1')
    elif(normalize_by_preds):
        to_plot = normalize(confusion,axis=0,norm='l1')
    if(logscale):
        plt.imshow(np.log(to_plot))
    else:
        plt.imshow(to_plot)
    for i in range(confusion.shape[0]):
        for j in range(confusion.shape[1]):
            plt.annotate(text = str(int(confusion[i,j])), xy=(j, i), ha='center',va='center',color='red')
    return(fig)

#MISC FUNCS
def hms_to_seconds(t):
  
  """Convert HMS format times to seconds
  
  Parameters:
  t (str): time in H:M:S format from Audition
  
  Returns:
  sec (float): time in seconds
  """
    
  s = t.split(':')
  if(len(s)==3):
    sec = int(s[0])*3600 + int(s[1])*60 + float(s[2])
  elif(len(s)==2):
    sec = int(s[0])*60 + float(s[1])
  else:
    raise ValueError('Unknown time format')
  return sec
