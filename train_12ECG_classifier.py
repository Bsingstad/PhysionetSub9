#!/usr/bin/env python
import tensorflow as tf
from tensorflow import keras
import wfdb
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
import numpy as np, os, sys, joblib
from scipy.io import loadmat
import tensorflow_addons as tfa





def train_12ECG_classifier(input_directory, output_directory):
    # Load data.
    print('Loading data...')

    gender=[]
    age=[]
    labels=[]
    ecg_filenames=[]
    for ecgfilename in sorted(os.listdir(input_directory)):
        if ecgfilename.endswith(".mat"):
            data, header_data = load_challenge_data(input_directory+"/"+ecgfilename)
            labels.append(header_data[15][5:-1])
            ecg_filenames.append(input_directory + "/" + ecgfilename)
            gender.append(header_data[14][6:-1])
            age.append(header_data[13][6:-1])

    ecg_filenames = np.asarray(ecg_filenames)

    # Gender processing - replace with nicer code later
    gender = np.asarray(gender)
    gender[np.where(gender == "Male")] = 0
    gender[np.where(gender == "male")] = 0
    gender[np.where(gender == "M")] = 0
    gender[np.where(gender == "Female")] = 1
    gender[np.where(gender == "female")] = 1
    gender[np.where(gender == "F")] = 1
    gender[np.where(gender == "NaN")] = 2
    gender = gender.astype(np.int)

    # Age processing - replace with nicer code later
    age = np.asarray(age)
    age[np.where(age == "NaN")] = -1
    age = age.astype(np.int)

    # Load SNOMED codes
    SNOMED_scored=pd.read_csv("SNOMED_mappings_scored.csv", sep=";")
    SNOMED_unscored=pd.read_csv("SNOMED_mappings_unscored.csv", sep=";")

    # Load labels to dataframe
    df_labels = pd.DataFrame(labels)

    # Remove unscored labels
    for i in range(len(SNOMED_unscored.iloc[0:,1])):
        df_labels.replace(to_replace=str(SNOMED_unscored.iloc[i,1]), inplace=True ,value="undefined class", regex=True)

    # Replace overlaping SNOMED codes
    '''
    codes_to_replace=['713427006','284470004','427172004']
    replace_with = ['59118001','63593006','17338001']

    for i in range(len(codes_to_replace)):
        df_labels.replace(to_replace=codes_to_replace[i], inplace=True ,value=replace_with[i], regex=True)
    '''
    # One-Hot encode classes
    one_hot = MultiLabelBinarizer()
    y=one_hot.fit_transform(df_labels[0].str.split(pat=','))
    y= np.delete(y, -1, axis=1)
    classes_for_prediction = one_hot.classes_[0:-1]

    global order_array
    order_array = np.arange(0,y.shape[0],1)

    print(classes_for_prediction)
    print("classes: {}".format(y.shape[1]))

    # Train model.
    print('Training model...')


    model=create_model()
    batchsize = 30
    class_dict= class_dict={0: 63.62172285, 1: 12.54579025, 2: 5.42889102, 3: 60.23758865, 4: 18.14850427, 5: 18.54475983,
    6: 4.03971463, 7: 55.33224756 , 8: 52.26769231, 9: 34.04208417, 10: 7.90828678, 11: 10.91709512, 12: 3.1032152, 13: 7.97886332, 
    14: 66.09727626 , 15: 0.90529738,16: 7.86071263, 17: 100.5147929, 18: 15.11298932, 19: 10.49227918, 20: 43.78092784, 
    21: 7.87528975, 22: 16.91932271, 23: 88.93717277 , 24: 18.81173865, 25: 11.6829436 , 26: 27.75653595 }

    #model.fit_generator(generator=batch_generator(batch_size=batchsize, gen_x=generate_X(input_directory), gen_y=generate_y(y), 
    #gen_z=generate_z(age,gender), ohe_labels = classes_for_prediction),steps_per_epoch=(len(y)/batchsize), epochs=1)
    
    #HUSK Å LEGGE TIL CLASS_DICT
    def scheduler(epoch, lr):
        if epoch < 2:
            lr = 0.001
            return lr
        else:
            return lr * 0.1


    lr_schedule = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=1)

    model.fit(x=batch_generator(batch_size=batchsize, gen_x=generate_X(ecg_filenames), gen_y=generate_y(y), gen_z=generate_z(age, gender), ohe_labels=classes_for_prediction), 
    epochs=7, steps_per_epoch=(len(y)/batchsize), class_weight=class_dict, callbacks=[lr_schedule])

    # Save model.
    print('Saving model...')
    #model.save("model.h5")
    filename = os.path.join(output_directory, 'model.h5')
    model.save_weights(filename)

    #final_model={'model':model, 'imputer':imputer,'classes':classes}

    #filename = os.path.join(output_directory, 'finalized_model.sav')
    #joblib.dump(final_model, filename, protocol=0)

# Load challenge data.
def load_challenge_data(filename):
    x = loadmat(filename)
    data = np.asarray(x['val'], dtype=np.float64)
    new_file = filename.replace('.mat','.hea')
    input_header_file = os.path.join(new_file)
    with open(input_header_file,'r') as f:
        header_data=f.readlines()
    return data, header_data

# Find unique classes.
def get_classes(input_directory, filenames):
    classes = set()
    for filename in filenames:
        with open(filename, 'r') as f:
            for l in f:
                if l.startswith('#Dx'):
                    tmp = l.split(': ')[1].split(',')
                    for c in tmp:
                        classes.add(c.strip())
    return sorted(classes)




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
            name='accuracy', dtype=None, threshold=0.5)])
    return model



def generate_y(y):
    while True:
        for i in order_array:
            y_train = y[i]
            yield y_train

def generate_X(ecg_filenames):
    while True:
        for i in order_array:
            data, header_data = load_challenge_data(ecg_filenames[i])
            X_train_new = keras.preprocessing.sequence.pad_sequences(data, maxlen=5000, truncating='post',padding="post")
            X_train_new = X_train_new.reshape(5000,12)
            yield X_train_new

def generate_z(age, gender):
    while True:
        for i in order_array:
            gen_age = age[i]
            gen_gender = gender[i]
            z_train = [gen_age , gen_gender]
            yield z_train

def batch_generator(batch_size, gen_x,gen_y, gen_z, ohe_labels):
    np.random.shuffle(order_array)
    batch_features = np.zeros((batch_size,5000, 12))
    batch_labels = np.zeros((batch_size,len(ohe_labels)))
    batch_demo_data = np.zeros((batch_size,2))
    while True:
        for i in range(batch_size):

            batch_features[i] = next(gen_x)
            batch_labels[i] = next(gen_y)
            batch_demo_data[i] = next(gen_z)

        X_combined = [batch_features, batch_demo_data]
        yield X_combined, batch_labels