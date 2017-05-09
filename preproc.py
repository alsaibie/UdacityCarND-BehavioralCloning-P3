import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.utils import shuffle
import sgsmooth as smooth

############# Configuration ####################
datafolder  = 'data'
modelsave   = 'model8.h5'
toBlur      = True
toNoise     = True
toPlot      = True
toSmooth    = True
################################################

#Import CSV Data
samples = []
with open( datafolder + '/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
#inqazgnore the heading
samples = samples[1:]

def findIndices(lst, a):
    return [i for i, x in enumerate(lst) if x == a]

def clean(samples):
    print("Starting sample size: ", len(samples))
    #try to smooth angle data. Theory it will spread angles a bit away from flat zero.
    if toSmooth:  
        angl = np.array([float(row[3]) for row in samples])
        anglesmooth = smooth.savitzky_golay(angl, 15, 5)
        # return angle data back like nothing happened. 
        for ind, sample in enumerate(samples):
            sample[3] = anglesmooth[ind]

    samples = shuffle(samples)
	#ignore slow speed frames
    num_popped_slow = 0
    num_popped_zeroangle = 0
    n_bins     = 50
    angles = np.array([float(row[3]) for row in samples])
    hist, bins = np.histogram(angles, bins = n_bins)
    avg        = len(samples)/n_bins
    max_count  = max(hist)
    max_index  = findIndices(hist, max_count)
    #num_remove = max_count - 8 * avg
    num_remove = 0
    for sample in samples:
        if float(sample[6]) < 0.5:
            #sample.pop()
            samples.remove(sample)
            num_popped_slow += 1
            continue
        if abs(float(sample[3])) < 0.06 and num_popped_zeroangle < num_remove :
            #sample.pop()
            samples.remove(sample)
            num_popped_zeroangle += 1
    
    print("Samples popped for low speed: ", num_popped_slow)
    print("Samples popped for excess zero angles: ", num_popped_zeroangle)
    return samples

samples = clean(samples)

# Split validation data from training 
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)


def get_angle_hist(samples):
    n_bins = 50
    angles = np.array([float(row[3]) for row in samples])
    hist, bins = hist, bins = np.histogram(angles, n_bins)
    avg = len(samples)/n_bins
    if toPlot:
        plt.hist(angles, bins = n_bins)
        plt.show()
    return hist, bins, avg

		
def preprocess(im):
    # resize
    #im = cv2.resize(im,(320,160), interpolation = cv2.INTER_AREA)
    # convert to YUV
    #im = cv2.cvtColor(im, cv2.COLOR_BGR2YUV)
    return im

def generator(samples, batch_size = 16):
    num_samples = len(samples)
    # Steering angle adjustment for left and right images
    steering_correction = 0.35
    while 1:
        samples = shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                # For windows simulator data use split('\\')           
                #nameleft, namecenter, nameright = './data/IMG/'+batch_sample[1].split('/')[-1], './data/IMG/'+batch_sample[0].split('/')[-1], './data/IMG/'+batch_sample[2].split('/')[-1]
                nameleft, namecenter, nameright = datafolder + '/IMG/'+batch_sample[1].split(
                     '\\')[-1], datafolder + '/IMG/'+batch_sample[0].split(
                     '\\')[-1], datafolder + '/IMG/'+batch_sample[2].split('\\')[-1]
                left_image, center_image, right_image = preprocess(cv2.imread(nameleft)), preprocess(
                    cv2.imread(namecenter)), preprocess(cv2.imread(nameright))
                center_angle = float(batch_sample[3])

                # For each sample, we generate 3 image sets: left, right, and center.
                images.append(center_image)
                angles.append(center_angle)
                # Add left and right images
                left_angle, right_angle = center_angle + steering_correction, center_angle - steering_correction
                images.append(left_image)
                angles.append(left_angle)
                images.append(right_image)
                angles.append(right_angle)

            # Augment images with flipped set
            aug_images, aug_angles = [], []
            for image, angle in zip(images,angles):
                aug_images.append(image)
                aug_angles.append(angle)
                aug_images.append(cv2.flip(image,1))
                aug_angles.append( -1.0 * angle)

            X_train = np.array(aug_images)
            y_train = np.array(aug_angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# Batch size returned will be  x6 due to adding left and right images, then adding flipped ones.  
train_generator = generator(train_samples, batch_size=16)
validation_generator = generator(validation_samples, batch_size=16)
train_sample_len = 6 * len(train_samples)
valid_sample_len = 6 * len(validation_samples)

