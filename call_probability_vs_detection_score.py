#Measure the probability that a given detection is a call vs the score the detector assigned it

#PARAMS

NBINS = 10
MINBIN = 0.99
SCALE = .7

#Import stuff
import numpy as np
import pandas
import matplotlib.pyplot as plt

INFILE = '/Users/astrandb/Dropbox/meerkats/meerkats_shared/labeling/verify_posonly/HM_HRT_R07_20170903-20170908_file_6_(2017_09_07-05_44_59)_ASWMUX221092_label_cnn_20epoch_proportional_aug_2dconv_20181218_1-16200.0_thresh0.99_verify_posonly_ASP.csv'

labels = pandas.read_csv(INFILE,delimiter=',')

#Get scores and categories as numpy arrays
scores = np.asarray(labels['Name'].apply(pandas.to_numeric))
categories = np.asarray(labels['verif'].apply(pandas.to_numeric))

#Get scores associated with each category

call_scores = scores[np.where(categories == 0)]
noise_scores = scores[np.where(categories == 1)]
unk_scores = scores[np.where(categories == 2)]
misalign_scores = scores[np.where(categories == 3)]
sync_scores = scores[np.where(categories == 4)]

#Print some initial stats
print('Overall precision = ' + str(float(len(call_scores) + len(misalign_scores)) / float(len(noise_scores) + len(call_scores) + len(misalign_scores))))

#Get bins
bins = []
bins.append(MINBIN)
for i in range(NBINS):
    bins.append(bins[i]+ SCALE*(1-bins[i]))
bins.append(1)

bins = np.asarray(bins)
bins = np.flip(bins)

NBINS = len(bins)

probs_upper = np.zeros((NBINS-1))
probs_lower = np.zeros((NBINS-1))
nsamps = np.zeros((NBINS-1))
for i in range(NBINS-1):
    ncalls = sum((call_scores >= bins[i]))
    nnoise = sum((noise_scores >= bins[i]))
    nmisalign = sum((misalign_scores >= bins[i]))
    nsync = sum((sync_scores >= bins[i]))
    nunk = sum((unk_scores >= bins[i]))

    nscores = float(sum((scores >= bins[i])))/len(scores)*100

    if(nscores > 0):
        probs_upper[i] = float(ncalls + nmisalign + nunk) / (ncalls + nnoise + nmisalign + nsync + nunk)
        probs_lower[i] = float(ncalls + nmisalign) / (ncalls + nnoise + nmisalign + nsync + nunk)
    else:
        probs_upper[i] = np.nan
        probs_lower[i] = np.nan

    nsamps[i] = nscores

bins = bins.tolist()
bins = [np.round(x,8) for x in bins]
bins = [str(x) for x in bins]

plt.subplot(211)
plt.plot(range(NBINS-1),probs_upper,range(NBINS-1),probs_lower,marker='.')
plt.grid(True)
plt.xticks(range(NBINS-1),bins)
plt.xlabel('Threshold')
plt.ylabel('Precision')
plt.subplot(212)
plt.plot(range(NBINS-1),nsamps,marker='.')
plt.grid(True)
plt.xlabel('Threshold')
plt.xticks(range(NBINS-1),bins)
plt.ylabel('% of detections')
plt.ylim((0,100))
plt.show()

print(probs_upper)
print(probs_lower)

