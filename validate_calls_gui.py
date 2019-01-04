"""
GUI for validating meerkat calls

TODO: 
    Fix row and columns so that output is stored correctly
    Fix bug of first screen going twice (after you press spacebar first time)
    Add capability to label as non-focal?
"""
import pygame
import librosa
import scipy
import numpy as np
import pandas
import math
import os.path
import datetime
import time

from Tkinter import *
import Tkinter, Tkconstants, tkFileDialog
 

#PARAMETERS
get_user_input = False
NROW = 4
NCOL = 4
SAMPRATE = 8000

#FUNCS
#Convert HMS format times to seconds
def hms_to_seconds(t):
    s = t.split(':')
    if(len(s)==3):
        sec = int(s[0])*3600 + int(s[1])*60 + float(s[2])
    elif(len(s)==2):
        sec = int(s[0])*60 + float(s[1])
    else:
        raise ValueError('Unknown time format')
    return sec
  
def get_im_surfaces_and_sounds(aud_file,labels,idxs_curr,pad=.2,wid=2,SAMPRATE=8000):
            
    starts = labels['start_time'][idxs_curr]
    durs = labels['duration'][idxs_curr]

    t0 = np.min(starts) - 1
    tf = np.max(starts+durs) + 1

    #load audio for all current labels
    aud, sr = librosa.core.load(path=aud_file,sr=SAMPRATE,offset=t0,duration=tf-t0)

    #Create an array of arrays to hold spectrogram images
    im_list = []
    im_list2 = []
    im_list3 = []
    im_list4 = []
    im_list5 = []
    
    #Create lists to hold sound objects
    sound_list = []
    n_labels = len(idxs_curr)
    for idx in range(n_labels):
    
        #get start and end times from labels, with padding
        curr_start = labels['start_time'][idxs_curr[idx]]
        curr_end = curr_start + labels['duration'][idxs_curr[idx]]
        idx_start = int(np.round((curr_start - t0 - pad)*SAMPRATE))
        idx_end = int(np.round((curr_end - t0 + pad)*SAMPRATE))
        
        #get audio for current detection
        aud_curr = aud[idx_start:idx_end]
        
        #create spectrogram
        _,_,spec = scipy.signal.spectrogram(aud_curr,fs=sr,nperseg=255,noverlap=230,window='hanning')
        spec_norm = np.log(spec)
        spec_norm = (spec_norm - np.min(spec_norm)) / (np.max(spec_norm) - np.min(spec_norm)) * 255
        spec_norm = spec_norm.astype('uint8')
        spec_norm = np.transpose(spec_norm)
        
        #create images from spectrograms
        spec3 = np.zeros((spec_norm.shape[0],spec_norm.shape[1],3))
        spec3[:,:,0] = spec_norm
        spec3[:,:,1] = spec_norm
        spec3[:,:,2] = spec_norm
        
        spec3_v2 = spec3.copy()
        spec3_v2[:,:,2] = 0
        spec3_v2[:,:,1] = 0
        
        spec3_v3 = spec3.copy()
        spec3_v3[:,:,2] = 0

        spec3_v4 = spec3.copy()
        spec3_v4[:,:,0] = 0
        spec3_v4[:,:,2] = 0

        spec3_v5 = spec3.copy()
        spec3_v5[:,:,0] = 0
        spec3_v5[:,:,1] = 0

        im_scale = spec_norm.shape[0] / (curr_end - curr_start + pad*2)
        call_start = int(pad*im_scale)
        call_end = int(((curr_end - curr_start + 2*pad) - pad)*im_scale)
        spec3[call_start:(call_start+wid),:,:] = 0
        spec3[call_end:(call_end+wid),:,:] = 0
        spec3[call_start:(call_start+wid),:,1] = 255
        spec3[call_end:(call_end+wid),:,1] = 255
        
        spec3_v2[call_start:(call_start+wid),:,:] = 0
        spec3_v2[call_end:(call_end+wid),:,:] = 0
        spec3_v2[call_start:(call_start+wid),:,1] = 255
        spec3_v2[call_end:(call_end+wid),:,1] = 255
        
        spec3_v3[call_start:(call_start+wid),:,:] = 0
        spec3_v3[call_end:(call_end+wid),:,:] = 0
        spec3_v3[call_start:(call_start+wid),:,1] = 255
        spec3_v3[call_end:(call_end+wid),:,1] = 255

        spec3_v4[call_start:(call_start+wid),:,:] = 0
        spec3_v4[call_end:(call_end+wid),:,:] = 0
        spec3_v4[call_start:(call_start+wid),:,0] = 255
        spec3_v4[call_end:(call_end+wid),:,0] = 255


        surf = pygame.Surface(spec_norm.shape)                    
        im = pygame.surfarray.make_surface(spec3)
        im = pygame.transform.scale(im,(200,150))
        im = pygame.transform.flip(im,False,True)
        im_list.append(im)
        
        im2 = pygame.surfarray.make_surface(spec3_v2)
        im2 = pygame.transform.scale(im2,(200,150))
        im2 = pygame.transform.flip(im2,False,True)
        im_list2.append(im2)
        
        im3 = pygame.surfarray.make_surface(spec3_v3)
        im3 = pygame.transform.scale(im3,(200,150))
        im3 = pygame.transform.flip(im3,False,True)
        im_list3.append(im3)

        im4 = pygame.surfarray.make_surface(spec3_v4)
        im4 = pygame.transform.scale(im4,(200,150))
        im4 = pygame.transform.flip(im4,False,True)
        im_list4.append(im4)

        im5 = pygame.surfarray.make_surface(spec3_v5)
        im5 = pygame.transform.scale(im5,(200,150))
        im5 = pygame.transform.flip(im5,False,True)
        im_list5.append(im5)
        
        aud_sound = aud_curr*(2**16-1)
        aud_sound = np.asarray(aud_sound)

        aud_2chan = np.zeros((aud_sound.shape[0],2),dtype=np.int16)
        aud_2chan[:,0] = aud_sound
        aud_2chan[:,1] = aud_sound

        s = pygame.sndarray.make_sound(aud_2chan)
        
        sound_list.append(s)
		
    return(im_list,im_list2,im_list3,im_list4,im_list5,sound_list)
 
# Create a 2 dimensional array. A two dimensional
# array is simply a list of lists.
#grid[i][j] is 0 if a call, 1 if not (clicked)
def init_grid(NROW,NCOL):
    grid = []
    for row in range(NROW):
        # Add an empty array that will hold each cell
        # in this row
        grid.append([])
        for column in range(NCOL):
            grid[row].append(0)  # Append a cell
    return(grid) 
 

#MAIN

#Disable some warnings that were popping up falsely
pandas.options.mode.chained_assignment = None

# Define some colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
 
# This sets the WIDTH and HEIGHT of each grid location
WIDTH = 200
HEIGHT = 150

N_LABS = NROW*NCOL

# This sets the margin between each cell
MARGIN = 5

WINDOW_SIZE = [(WIDTH+MARGIN)*NCOL+MARGIN, (HEIGHT+MARGIN)*NROW+MARGIN]

#AUD_FILE = '/Users/astrandb/Dropbox/meerkats/data/Kalahari2017/HM_2017_3/COLLAR3/AUDIO3/HM_HRT_R07_20170903-20170908/HM_HRT_R07_20170903-20170908_file_6_(2017_09_07-05_44_59)_ASWMUX221092.wav'
#LAB_FILE = '/Users/astrandb/Dropbox/meerkats/meerkats_shared/labeling/autolabels_new_2dconvolution/HM_HRT_R07_20170903-20170908_file_6_(2017_09_07-05_44_59)_ASWMUX221092_label_cnn_20epoch_proportional_aug_2dconv_20181218_1-16200.0_thresh0.99.csv'

#PRINT INSTRUCTIONS
print('')
print('')
print('Welcome to Call It! The meerkat call verification system, developed by Ariana Strandburg-Peshkin (2019)')
print('')
print('NOTE: So far this code has only been tested on Mac OSX 10.12.6, Python 2.7.15.')
print('It is not expected to work in Python 3.')
print('Any questions? Email arianasp@gmail.com')
print('')
print('')
print('INSTRUCTIONS:')
print('')
print('Press "spacebar" to move to next screen')
print('Press "backspace" to move to previous screen')
print('Press "s" to save progress at any time (you should do this periodically!)')
print('')
print('To play a detection, click on it (will only play if un-highlighted).')
print('')
print('To mark a detection as not a call (RED), hold "shift" and click on it')
print('To mark a detection as possibly a call (YELLOW), hold "command" and click on it')
print('To mark a detection as a call with misaligned call boundaries (GREEN), hold "control" and click on it')
print('To mark a detection as containing part of a synch call, and NOT a meerkat call (BLUE), hold "option" and click on it')
print('')
print('To unhihglight a detection, click on it.')
print('')

UID = raw_input('To start, please enter your initials: ')

print('')
print('Got it, ' + UID + '. Now, please select a label file to verify (csv).')
print('')
time.sleep(1)

root = Tk()
LAB_FILE = tkFileDialog.askopenfilename(initialdir = "/",title = "Select label file (wav)",filetypes = (("csv files","*.csv"),("all files","*.*")))
print('Label file selected:')
print(LAB_FILE)
print('')

print('Now please select the corresponding audio file (wav).')
time.sleep(1)

AUD_FILE = tkFileDialog.askopenfilename(initialdir = "/",title = "Select audio file (csv)",filetypes = (("wav files","*.wav"),("all files","*.*")))
print('Audio file selected:')
print (AUD_FILE)
print('')
root.quit()

time.sleep(1)


if get_user_input:

    AUD_FILE = raw_input('Please enter the full path to the audio (.wav) file: ')
    while not(os.path.isfile(AUD_FILE)):
        AUD_FILE = raw_input('File not found. Please enter the full path to the audio (.wav) file: ')

    LAB_FILE = raw_input('Please enter the full path to the label (.csv) file: ')
    while not(os.path.isfile(LAB_FILE)):
        LAB_FILE = raw_input('File not found. Please enter the full path to the label (.csv) file: ')

VER_FILE = LAB_FILE[0:(len(LAB_FILE)-4)] + '_verify_posonly_' + UID + '.csv'
leftoff = 'n'
while os.path.isfile(VER_FILE):
    print('A verification file with name '+ VER_FILE + 'already exists.')
    leftoff = raw_input('Start where you left off? (y/n ) ')
    if leftoff == 'y':
        labels = pandas.read_csv(VER_FILE,delimiter=',')
        if(np.sum(labels['verif']==-1)==0):
            raise Exception('File is completed (no missing verifications). If you would like to overwrite the completed file, you must delete it manually and restart. To create a different file, just enter different initials.')
        first_idx = labels[labels['verif']==-1].index.values.astype(int)[0]
        idxs_curr = range(first_idx,first_idx + N_LABS)
        tot_labels = labels.shape[0]
        #make sure indexes don't go over max
        if(np.max(idxs_curr) >= tot_labels):
            idxs_curr = range(tot_labels-N_LABS,tot_labels)
        results = labels['verif']
        results[idxs_curr] = 0
        print('Starting at index ' + str(np.min(idxs_curr)) + ' of ' + str(tot_labels))
        print('')
        break
    elif leftoff == 'n':
        overw = raw_input('Would you like to overwrite the existing file? (y/n) ')
        if overw == 'y':
            break
        else:
            VER_FILE = raw_input('Please enter an alternative path for the verification file: ')
print('')            
print('Verification file will be saved at: ')
print(VER_FILE)
print('')

if not(leftoff=='y'):
    labels = pandas.read_csv(LAB_FILE,delimiter='\t')
    labels['start_time'] = labels['Start'].map(lambda x: hms_to_seconds(x))
    labels['duration'] = labels['Duration'].map(lambda x: hms_to_seconds(x))
    labels['end_time'] = labels['start_time'] + labels['duration']
    idxs_curr = range(N_LABS)
    tot_labels = labels.shape[0]
    results = np.zeros(tot_labels)
    results[:] = -1
    results[idxs_curr] = 0

tot_labels = labels.shape[0]

ready = 'n'
while not(ready=='y'):
    ready = raw_input('Found ' + str(tot_labels) + ' detections - ready to go? (y/n) ')

print('')
pygame.mixer.pre_init(frequency=8000, size=-16, channels=2, buffer=4096)

# Initialize pygame
pygame.init()

im_list, im_list2, im_list3, im_list4, im_list5, sound_list = get_im_surfaces_and_sounds(aud_file=AUD_FILE,labels=labels,idxs_curr=idxs_curr)
grid = init_grid(NROW,NCOL)

# Set the HEIGHT and WIDTH of the screen
screen = pygame.display.set_mode(WINDOW_SIZE)
 
# Set title of screen
pygame.display.set_caption("Verifying")


# Loop until the user clicks the close button.
done = False
 
# Used to manage how fast the screen updates
clock = pygame.time.Clock()
 
# -------- Main Program Loop -----------
while not done:
    for event in pygame.event.get():  # User did something
        if event.type == pygame.QUIT:  # If user clicked close
            done = True  # Flag that we are done so we exit this loop
        elif event.type == pygame.MOUSEBUTTONDOWN:
            # User clicks the mouse. Get the position
            pos = pygame.mouse.get_pos()
            # Change the x/y screen coordinates to grid coordinates
            column = pos[0] // (WIDTH + MARGIN)
            row = pos[1] // (HEIGHT + MARGIN)

            #get index in current_idxs list
            result_idx = column*NROW+row
            mods = pygame.key.get_mods()
            if mods == 1: #shift = red = not a call
                grid[row][column] = 1
                results[idxs_curr[result_idx]] = 1
            elif mods == 1024: #command = yellow = maybe a call
                grid[row][column] = 2
                results[idxs_curr[result_idx]] = 2
            elif mods == 64: #ctrl = green = misaligned call boundaries
                grid[row][column] = 3
                results[idxs_curr[result_idx]] = 3
            elif mods == 256: #option = blue = synch call
                grid[row][column] = 4
                results[idxs_curr[result_idx]] = 4
            else:
                if grid[row][column]==0:
                    sound_list[result_idx].play()
                    pygame.time.wait(int(1))
                # Set that location to zero
                grid[row][column] = 0
                results[idxs_curr[result_idx]] = 0
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                if not(len(idxs_curr) == 0):
                    if np.max(idxs_curr) < tot_labels:
                        idxs_curr = [x+N_LABS for x in idxs_curr]
                        if(np.max(idxs_curr) >= tot_labels):
                            if(np.min(idxs_curr) == tot_labels):
                                done = True
                                break
                            else:
                                idxs_curr = range(tot_labels-N_LABS,tot_labels)
                        im_list, im_list2, im_list3, im_list4, im_list5, sound_list = get_im_surfaces_and_sounds(aud_file=AUD_FILE,labels=labels,idxs_curr=idxs_curr)
                        grid = init_grid(NROW,NCOL)
                        results[idxs_curr] = 0
                        pygame.display.set_caption("Verifying - " + str(np.round(float(np.min(idxs_curr)) / tot_labels*100)) + '% complete')
            elif event.key == pygame.K_BACKSPACE:
        	if not(np.min(idxs_curr) == 0):
                    idxs_curr = [x-N_LABS for x in idxs_curr]
                im_list, im_list2, im_list3, im_list4, im_list5, sound_list = get_im_surfaces_and_sounds(aud_file=AUD_FILE,labels=labels,idxs_curr=idxs_curr)
                grid = init_grid(NROW,NCOL)
                results[idxs_curr] = 0
            elif event.key == pygame.K_s:
                print('saved file at:')
                print(datetime.datetime.now())
                labels['verif'] = results
                labels.to_csv(VER_FILE,index=False)
            else:
                continue
 
    # Set the screen background
    screen.fill(BLACK)
 
    # Draw the grid
    idx = 0
    for c in range(NCOL):
        for r in range(NROW):
            color = WHITE
            if idx < len(im_list):
                if grid[r][c] == 1:
                    screen.blit(im_list2[idx],((MARGIN + WIDTH)*c + MARGIN,(MARGIN + HEIGHT)*r + MARGIN))
                elif grid[r][c] == 2:
            	    screen.blit(im_list3[idx],((MARGIN + WIDTH)*c + MARGIN,(MARGIN + HEIGHT)*r + MARGIN))
                elif grid[r][c] == 3:
                    screen.blit(im_list4[idx],((MARGIN + WIDTH)*c + MARGIN,(MARGIN + HEIGHT)*r + MARGIN))
                elif grid[r][c] == 4:
                    screen.blit(im_list5[idx],((MARGIN + WIDTH)*c + MARGIN,(MARGIN + HEIGHT)*r + MARGIN))
                else:
                    screen.blit(im_list[idx],((MARGIN + WIDTH)*c + MARGIN,(MARGIN + HEIGHT)*r + MARGIN))
                idx = idx + 1
    
    # Limit to 60 frames per second
    clock.tick(60)
 
    # Go ahead and update the screen with what we've drawn.
    pygame.display.flip()
    
 
# Be IDLE friendly. If you forget this line, the program will 'hang'
# on exit.
pygame.quit()

#Last save
sure = 'n'
while(not(sure=='y')):
    lastsave = raw_input('You have reached the end of the file or quit. Would you like to save? (y/n): ')
    if(lastsave=='n'):
        sure = raw_input('Are you sure you do NOT want to save? (y/n): ')
        if(sure=='y'):
            break
    else:
        sure = 'y'

if(not(lastsave=='n')):
    print('saved file at:')
    print(datetime.datetime.now())
    labels['verif'] = results
    labels.to_csv(VER_FILE,index=False)

print('Exiting.')

