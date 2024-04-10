#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Optional libraries, functions, and variables. You can change or remove them.
#
################################################################################


import numpy as np
import os
import sys

from helper_code import *

################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments of the functions.
#
################################################################################

from itertools import chain
import joblib
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from scipy.signal import resample
from scipy.signal import detrend
from scipy.stats import entropy
from typing import Callable, Union
from collections import Counter


List = list
Int = int
Float = float

Numeric = Union[Float, Int]

def mode(data: Union[List, np.ndarray]) -> Numeric:
    # TODO: Fix types
    value, _ = Counter(data).most_common(1)[0]  # returns a list of n most common (i.e., `[(value, count), ...]`)
    return value

def zeroECGSignal(signal: np.ndarray, zeroingMethod: Callable[[np.ndarray], float]=mode) -> np.ndarray:
    zeroPoint = zeroingMethod(signal)

    return signal - zeroPoint


def rotate_image(image):
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
    # Apply edge detection
    edges = cv2.Canny(gray, 100, 200)
        
    # Detect lines using Hough Line Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
        
    # Calculate the angles and filter out vertical lines
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
        if abs(angle) < 45 or abs(angle) > 135:  # Filter out vertical lines
            angles.append(angle)
                
    # Find the dominant horizontal angle
    if angles:
        dominant_angle = max(set(angles), key=angles.count)
    else:
        dominant_angle = 0  # Assume image is already horizontal
            
    # Check if the image is already horizontal
    if abs(dominant_angle) < 5:  # Tolerance of 5 degrees
        rotated_image = image
    else:
        # Calculate the rotation angle to make the dominant horizontal line horizontal
        rotation_angle = dominant_angle
                
        # Rotate the image
        rows, cols, _ = image.shape
        rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), rotation_angle, 1)
        rotated_image = cv2.warpAffine(image, rotation_matrix, (cols, rows))
            
    return rotated_image

        

def grid_lead_removal(image, template_image, grid_threshold):
    
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    _, no_grid = cv2.threshold(gray_image, grid_threshold, 255, cv2.THRESH_BINARY_INV)
    img = no_grid - template_image
    only_signal_img = cv2.bitwise_not(img)
    
    kernel = np.ones((3,3),np.uint8)
    eroded = cv2.erode(only_signal_img,kernel,iterations = 1)
    
    return eroded
    

def divide_leads(image):
    Lead_1 = image[550:850, 50:571]  # Lead 1
    Lead_4 = image[550:855, 570:1066]  # Lead aVR
    Lead_7 = image[550:855, 1065:1555]  # Lead V1
    Lead_10 = image[550:855, 1553:2083]  # Lead V4
    Lead_2 = image[850:1155, 50:571]  # Lead 2
    Lead_5 = image[850:1155, 570:1066]  # Lead aVL
    Lead_8 = image[850:1155, 1065:1555]  # Lead V2
    Lead_11 = image[850:1155, 1553:2083]  # Lead V5
    Lead_3 = image[1150:1455, 50:571]  # Lead 3
    Lead_6 = image[1150:1455, 570:1066]  # Lead aVF
    Lead_9 = image[1150:1455, 1065:1555]  # Lead V3
    Lead_12 = image[1150:1455, 1553:2083]  # Lead V6
    # Lead_13 = image[1400:1600, 100:600]  # Long Lead
    
    ll = [Lead_1, Lead_2, Lead_3, Lead_4,Lead_5, Lead_6, Lead_7,Lead_8,Lead_9,Lead_10,Lead_11,Lead_12]
    
    return ll
    

def extract_ecg_signal(binary_image):
    
    # Initialize list to store signal values
    signal_values = []
    
    # Iterate over each column (x-coordinate)
    for x in range(binary_image.shape[1]):
        # Get the column
        column = binary_image[:, x]
        
        # Find non-zero indices (y-coordinates of ECG signal)
        zero_indices = np.where(column == 0)[0]
        
        # If there are non-zero indices, take median
        if len(zero_indices) > 0:
            median_y = np.median(zero_indices)
        else:
            # If no non-zero indices, set median_y to 0
            median_y = 0
        
        # Append median_y to signal_values
        signal_values.append(median_y)
    
    signal_values = np.array(signal_values)
#     # Normalize signal values to range [0, 1]
    #signal_values = np.array(signal_values) / binary_image.shape[0]

    signal_values = 255 - signal_values

    return signal_values



def scale_ecg_signal(signal_values, fs):

    # Conversion factors
    #pixels_per_200ms = 39.37007874015748
    pixels_per_0_5mV = 39.37007874015748
    
#     time_resolution = 200 / pixels_per_200ms
    
    voltage_resolution = 0.5 / pixels_per_0_5mV 
    
    
#     scaled_time = x_pix * time_resolution
    scaled_voltage = signal_values * voltage_resolution * 1000 # multiply by gain
    
    num_samples = int(2.5 * fs)
    final_signal = resample(scaled_voltage, num_samples)
    
    return final_signal


# Train your digitization model.
def train_digitization_model(data_folder, model_folder, verbose):
    # Find data files.
    if verbose:
        print('Training the digitization model...')
        print('Finding the Challenge data...')

    if verbose:
        print('Done.')
        print()

# Train your dx classification model.
def train_dx_model(data_folder, model_folder, verbose):
    # Find data files.
    if verbose:
        print('Training the dx classification model...')
        print('Finding the Challenge data...')

    records = find_records(data_folder)
    num_records = len(records)

    if num_records == 0:
        raise FileNotFoundError('No data was provided.')

    # Extract the features and labels.
    if verbose:
        print('Extracting features and labels from the data...')

    features = list()
    dxs = list()

    

    for i in range(num_records):
        if verbose:
            width = len(str(num_records))
            print(f'- {i+1:>{width}}/{num_records}: {records[i]}...')

        record = os.path.join(data_folder, records[i])

        # Extract the features from the image, but only if the image has one or more dx classes.
        dx = load_dx(record)
        if dx:
            current_features = extract_features(record)
            # print("hi")
            current_features = get_eeg_features2(current_features)

            features.append(current_features)
            dxs.append(dx)

    if not dxs:
        raise Exception('There are no labels for the data.')

    features = np.vstack(features)
    classes = sorted(set.union(*map(set, dxs)))
    dxs = compute_one_hot_encoding(dxs, classes)

    # Train the model.
    if verbose:
        print('Training the model on the data...')

   
    model = XGBClassifier()

    # Defining the hyperparameters grid for GridSearchCV
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 4, 5],
        'learning_rate': [0.1, 0.01, 0.001]
    }
   
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)

    grid_search.fit(features, dxs)

    model = grid_search.best_estimator_

    # model.fit(features,dxs)

    

    # Create a folder for the model if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)

    # Save the model.
    save_dx_model(model_folder, model, classes)

    if verbose:
        print('Done.')
        print()

# Load your trained digitization model. This function is *required*. You should edit this function to add your code, but do *not*
# change the arguments of this function. If you do not train a digitization model, then you can return None.


def load_digitization_model(model_folder, verbose):

    return 0

# Load your trained dx classification model. This function is *required*. You should edit this function to add your code, but do
# *not* change the arguments of this function. If you do not train a dx classification model, then you can return None.


def load_dx_model(model_folder, verbose):
    filename = os.path.join(model_folder, 'dx_model.sav')
    return joblib.load(filename)

# Run your trained digitization model. This function is *required*. You should edit this function to add your code, but do *not*
# change the arguments of this function.

def run_digitization_model(digitization_model, record, verbose):


    # Extract features.
    signal = extract_features(record)

    signal = signal.T

    return signal

# Run your trained dx classification model. This function is *required*. You should edit this function to add your code, but do
# *not* change the arguments of this function.


def run_dx_model(dx_model, record, signal, verbose):
    model = dx_model['model']
    classes = dx_model['classes']

    # Extract features.
    
    features = extract_features(record)
    features = get_eeg_features2(features)
    features = features.reshape(1, -1)

    # Get model probabilities.
    probabilities = model.predict_proba(features)
 

    probabilities = np.asarray(probabilities, dtype=np.float32)[0]

    # Choose the class(es) with the highest probability as the label(s).
    max_probability = np.nanmax(probabilities)
    labels = [classes[i] for i, probability in enumerate(probabilities) if probability == max_probability]

    return labels

################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################

# Extract features.
def extract_features(record):

    template_image = cv2.imread('lead_template.png', 0)
    signals = []

    image = load_image(record)
    image_np = np.array(image[0])
    rotated_image = rotate_image(image_np)
    clean_signal_image = grid_lead_removal(rotated_image, template_image, 50)
    sub_images = divide_leads(clean_signal_image)
    for sub_image in sub_images:
        signal_extracted = extract_ecg_signal(sub_image)
        zeroed_signal = zeroECGSignal(signal_extracted)
        final_signal = scale_ecg_signal(zeroed_signal, 100)
        signals.append(final_signal)


    return np.array(signals)
  



def zero_crossing_rate(signal):
    KK = []
    for i in range(12):
        crossings = np.where(np.diff(np.sign(signal[i])))[0]
        zcr = len(crossings) / (2 * len(signal[i]))
        KK.append(zcr)
    return np.array(KK)
        
    

def energy(signal):
    KK = []
    for i in range(12):
        f = np.sum(np.square(signal[i]))
        KK.append(f)
    return np.array(KK)



def entropy_feature(signal):
    KK = []
    for i in range(12):
        hist, _ = np.histogram(signal[i], bins=50)
        hist = hist / hist.sum()  # Normalize histogram
        KK.append(entropy(hist))
    return np.array(KK)


def dominant_frequency(signal, fs):
    kk = []
    for i in range(12):
        fft_result = np.fft.fft(signal[i])
        freqs = np.fft.fftfreq(len(fft_result), d=1/fs)
        magnitude = np.abs(fft_result)
        dominant_freq_index = np.argmax(magnitude)
        dominant_freq = freqs[dominant_freq_index]
        kk.append(dominant_freq)
    return np.array(kk)

def spectral_entropy(signal):
    kk = []
    for i in range(12):
        fft_result = np.fft.fft(signal[i])
        magnitudes = np.abs(fft_result)
        normalized_magnitudes = magnitudes / np.sum(magnitudes)
        kk.append(entropy(normalized_magnitudes))
    return np.array(kk)

def power_distribution(signal):
    kk = []
    for i in range(12):
        
        fft_result = np.fft.fft(signal[i])
        power = np.square(np.abs(fft_result))
        normalized_power = power / np.sum(power)
        kk.append(power)
    return np.array(kk)

def get_eeg_features2(data):
    if data is None:
        return float("nan")*np.ones(108)
    
    features = np.hstack(  (zero_crossing_rate(data).ravel() , energy(data).ravel() , entropy_feature(data).ravel()   ,  dominant_frequency(data,100).ravel(), spectral_entropy(data).ravel() ,  np.mean(power_distribution(data),axis = 1).ravel()  )  )
    return features


def save_digitization_model(model_folder, model):
	return 0


# Save your trained dx classification model.

def save_dx_model(model_folder, model, classes):
    d = {'model': model, 'classes': classes}
    filename = os.path.join(model_folder, 'dx_model.sav')
    joblib.dump(d, filename, protocol=0)
