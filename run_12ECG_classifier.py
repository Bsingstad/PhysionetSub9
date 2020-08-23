#!/usr/bin/env python
import numpy as np, os, sys, joblib
import joblib
import tensorflow as tf
from tensorflow import keras
from scipy.io import loadmat
import tensorflow_addons as tfa
from scipy.signal import butter, lfilter, filtfilt
from scipy.signal import find_peaks
from scipy.signal import peak_widths
from scipy.signal import savgol_filter






def create_model():

    inputA = keras.layers.Input(shape=(5000,12))
    inputB = keras.layers.Input(shape=(2,))
    
    
    conv1 = keras.layers.Conv1D(filters=128, kernel_size=8,input_shape=(5000,12), padding='same')(inputA)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.Activation(activation='relu')(conv1)

    conv2 = keras.layers.Conv1D(filters=256, kernel_size=5, padding='same')(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.Activation('relu')(conv2)

    conv3 = keras.layers.Conv1D(128, kernel_size=3,padding='same')(conv2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.Activation('relu')(conv3)

    gap_layer = keras.layers.GlobalAveragePooling1D()(conv3)
    model1 = keras.Model(inputs=inputA, outputs=gap_layer)

    conv1 = keras.layers.Conv1D(filters=256,kernel_size=10,strides=1,padding='same')(inputA)
    conv1 = tfa.layers.InstanceNormalization()(conv1)
    conv1 = keras.layers.PReLU(shared_axes=[1])(conv1)
    conv1 = keras.layers.Dropout(rate=0.2)(conv1)
    conv1 = keras.layers.MaxPooling1D(pool_size=2)(conv1)
    # conv block -2
    conv2 = keras.layers.Conv1D(filters=512,kernel_size=22,strides=1,padding='same')(conv1)
    conv2 = tfa.layers.InstanceNormalization()(conv2)
    conv2 = keras.layers.PReLU(shared_axes=[1])(conv2)
    conv2 = keras.layers.Dropout(rate=0.2)(conv2)
    conv2 = keras.layers.MaxPooling1D(pool_size=2)(conv2)
    # conv block -3
    conv3 = keras.layers.Conv1D(filters=1024,kernel_size=42,strides=1,padding='same')(conv2)
    conv3 = tfa.layers.InstanceNormalization()(conv3)
    conv3 = keras.layers.PReLU(shared_axes=[1])(conv3)
    conv3 = keras.layers.Dropout(rate=0.2)(conv3)
    # split for attention
    attention_data = keras.layers.Lambda(lambda x: x[:,:,:512])(conv3)
    attention_softmax = keras.layers.Lambda(lambda x: x[:,:,512:])(conv3)
    # attention mechanism
    attention_softmax = keras.layers.Softmax()(attention_softmax)
    multiply_layer = keras.layers.Multiply()([attention_softmax,attention_data])
    # last layer
    dense_layer = keras.layers.Dense(units=512,activation='sigmoid')(multiply_layer)
    dense_layer = tfa.layers.InstanceNormalization()(dense_layer)
    # output layer
    flatten_layer = keras.layers.Flatten()(dense_layer)
    model2 = keras.Model(inputs=inputA, outputs=flatten_layer)


    mod3 = keras.layers.Dense(50, activation="relu")(inputB) # 2 -> 100
    mod3 = keras.layers.Dense(2, activation="sigmoid")(mod3) # Added this layer
    model3 = keras.Model(inputs=inputB, outputs=mod3)

    combined = keras.layers.concatenate([model1.output, model2.output, model3.output])
    final_layer = keras.layers.Dense(27, activation="sigmoid")(combined)
    model = keras.models.Model(inputs=[inputA,inputB], outputs=final_layer)

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=tf.keras.optimizers.Adam(), metrics=[tf.keras.metrics.BinaryAccuracy(
            name='accuracy')])
    return model


def run_12ECG_classifier(data,header_data,loaded_model):
    


    threshold = np.array([0.12585957, 0.09031925, 0.09345833, 0.17864081, 0.11545804,
       0.27795241, 0.1596176 , 0.11184793, 0.16626318, 0.24791257,
       0.1930114 , 0.07277747, 0.05153947, 0.06665818, 0.09982059,
       0.00390505, 0.14655532, 0.19118162, 0.17891057, 0.11025203,
       0.15657453, 0.11539103, 0.1691824 , 0.17392144, 0.17765048,
       0.10066959, 0.08176011])


    # Use your classifier here to obtain a label and score for each class.
    model = loaded_model
    padded_signal = keras.preprocessing.sequence.pad_sequences(data, maxlen=5000, truncating='post',padding="post")
    reshaped_signal = padded_signal.reshape(1,5000,12)

    #Rule-based model
    avg_hr = 0
    peaks = 0
    try:
        peaks = DetectRWithPanTompkins(data[1],int(header_data[0].split()[2]))
        
        try:
            peaks = R_correction(data[0], peaks)
        except:
            print("Did not manage to do R_correction")
        
    except:
        print("Did not manage to find any peaks using Pan Tomkins")

          
    try:
        rr_interval, avg_hr = heartrate(peaks,int(header_data[0].split()[2]))
    except:
        print("not able to calculate heart rate")
        rr_interval = 0
        avg_hr = 0

    gender = header_data[14][6:-1]
    age=header_data[13][6:-1]
    if gender == "Male":
        gender = 0
    elif gender == "male":
        gender = 0
    elif gender =="M":
        gender = 0
    elif gender == "Female":
        gender = 1
    elif gender == "female":
        gender = 1
    elif gender == "F":
        gender = 1
    elif gender =="NaN":
        gender = 2

    # Age processing - replace with nicer code later
    if age == "NaN":
        age = -1
    else:
        age = int(age)

    demo_data = np.asarray([age,gender])
    reshaped_demo_data = demo_data.reshape(1,2)

    combined_data = [reshaped_signal,reshaped_demo_data]


    score  = model.predict(combined_data)[0]
    
    binary_prediction = score > threshold
    binary_prediction = binary_prediction * 1

    if avg_hr != 0:     # bare gjør disse endringene dersom vi klarer å beregne puls
        if 60 < avg_hr < 100:
            binary_prediction[16] = 0
            binary_prediction[14] = 0
            binary_prediction[13] = 0
        elif avg_hr < 60 & binary_prediction[15] == 1:
            binary_prediction[13] = 1
        elif avg_hr < 60 & binary_prediction[15] == 0:
            binary_prediction[14] = 1
        elif avg_hr > 100:
            binary_prediction[16] = 1


    classes = ['10370003','111975006','164889003','164890007','164909002','164917005','164934002','164947007','17338001',
        '251146004','270492004','284470004','39732003','426177001','426627000','426783006','427084000','427172004','427393009','445118002','47665007','59118001',
        '59931005','63593006','698252002','713426002','713427006']

    return binary_prediction, score, classes

def load_12ECG_model(model_input):
    model = create_model()
    f_out='model.h5'
    filename = os.path.join(model_input,f_out)
    model.load_weights(filename)

    return model




def DetectRWithPanTompkins (signal, signal_freq):
    '''signal=ECG signal (type=np.array), signal_freq=sample frequenzy'''
    lowcut = 5.0
    highcut = 15.0
    filter_order = 2
    
    nyquist_freq = 0.5 * signal_freq
    low = lowcut / nyquist_freq
    high = highcut / nyquist_freq
    
    b, a = butter(filter_order, [low, high], btype="band")
    y = lfilter(b, a, signal)
    
    diff_y=np.ediff1d(y)
    squared_diff_y=diff_y**2
    
    integrated_squared_diff_y =np.convolve(squared_diff_y,np.ones(5))
    
    normalized = (integrated_squared_diff_y-min(integrated_squared_diff_y))/(max(integrated_squared_diff_y)-min(integrated_squared_diff_y))
    """
    peaks, metadata = find_peaks(integrated_squared_diff_y, 
                                 distance=signal_freq/5 , 
                                 height=(sum(integrated_squared_diff_y)/len(integrated_squared_diff_y))
                                )
    """
    peaks, metadata = find_peaks(normalized, 
                             distance=signal_freq/5 , 
                             #height=500,
                             height=0.5,
                             width=0.5
                            )

    return peaks

def heartrate(r_time, sampfreq):
    
    #qrs = xqrs.qrs_inds from annotateR()
    #sampfreq = sample frequency - can be found with y['fs'] (from getDataFromPhysionet())
    
    HeartRate = []
    TimeBetweenBeat= []
    for index, item in enumerate(r_time,-1):
        HeartRate.append(60/((r_time[index+1]-r_time[index])/sampfreq))
        TimeBetweenBeat.append((r_time[index+1]-r_time[index])/sampfreq)
    del HeartRate[0]
    avgHr = sum(HeartRate)/len(HeartRate)
    TimeBetweenBeat= np.asarray(TimeBetweenBeat)
    TimeBetweenBeat=TimeBetweenBeat * 1000 # sec to ms
    TimeBetweenBeat = TimeBetweenBeat[1:] # remove first element
    return TimeBetweenBeat, avgHr

def R_correction(signal, peaks):
    '''signal = ECG signal, peaks = uncorrected R peaks'''
    peaks_corrected, metadata = find_peaks(signal, distance=min(np.diff(peaks)))            
    return peaks_corrected  
