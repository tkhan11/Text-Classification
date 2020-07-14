import string
import numpy as np
import pandas as pd
from numpy import argmax
from numpy import array
import re

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import time
import itertools
import warnings
from sklearn.model_selection import StratifiedKFold

seconds= time.time()
time_start = time.ctime(seconds)   #  The time.ctime() function takes seconds passed since epoch
print("start time:", time_start,"\n")    # as an argument and returns a string representing time.



# Data Acquisition

Train_dataset=pd.read_csv('.\\drugLibTrain_raw.tsv', sep='\t')
Test_dataset=pd.read_csv('.\\drugLibTest_raw.tsv', sep='\t')

#print("Train dataset shape :",Train_dataset.shape, " Test dataset shape: ",Test_dataset.shape,"\n")

# DATA PREPROCESSING Phase


# for Predicting Patient Satisfaction based on rating and all reviews

Prediction_Train_data = pd.DataFrame({'rating':Train_dataset.rating,
                                     'benefits_reviews':Train_dataset.benefitsReview,
                                     'side_effects_reviews':Train_dataset.sideEffectsReview,
                                     'comments':Train_dataset.commentsReview})

Prediction_Test_data = pd.DataFrame({'rating':Test_dataset.rating,
                                     'benefits_reviews':Test_dataset.benefitsReview,
                                     'side_effects_reviews':Test_dataset.sideEffectsReview,
                                     'comments':Test_dataset.commentsReview})


# performing concatanation to join benifits_review, side_effects_review and comments into a Report attribute to predict the overall satisfaction of patients

report=['benefits_reviews','side_effects_reviews','comments']

Prediction_Train_data['report'] = Prediction_Train_data[report].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
Prediction_Test_data['report'] = Prediction_Test_data[report].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)





# Labeling of ratings as Postive, Negative and Neutral for sentiment classification
Prediction_Train_data['Sentiment'] = [ 'Negative' if (x<=4) else 'Neutral' if (4<x<=7) else 'Positive' for x in Prediction_Train_data['rating']]
Prediction_Test_data['Sentiment'] = [ 'Negative' if (x<=4) else 'Neutral' if (4<x<=7) else 'Positive' for x in Prediction_Test_data['rating']]


# Dropping the columns that are not required for the neural network.
Prediction_Train_data.drop(['rating', 'benefits_reviews', 'side_effects_reviews','comments'],axis=1,inplace=True)
Prediction_Test_data.drop(['rating', 'benefits_reviews', 'side_effects_reviews','comments'],axis=1,inplace=True)

print("\nTrain dataset: ",Prediction_Train_data.shape,"Test dataset: ",Prediction_Test_data.shape)



# Text Pre-Processing on Test and Train data

# filtering out all the rows with empty comments.
Prediction_Train_data = Prediction_Train_data[Prediction_Train_data.report.apply(lambda x: x !="")]
Prediction_Test_data = Prediction_Test_data[Prediction_Test_data.report.apply(lambda x: x !="")]


def process_text(report):
    
    # Remove puncuation
    report = report.translate(string.punctuation)
    
    # Convert words to lower case and split them
    report = report.lower().split()
    
    # Remove stop words

    #stop_words = set(stopwords.words("english"))
    #report = [w for w in report if not w in stop_words]
    report = " ".join(report)

    # Clean the text
    report = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", report)
    report = re.sub(r",", " ", report)
    report = re.sub(r"\.", " ", report)
    report = re.sub(r"!", "  ", report)
    report = re.sub(r":", "  ", report)

    # Stemming
    report = report.split()
    stemmer = SnowballStemmer('english')
    stemmed_words = [stemmer.stem(word) for word in report]
    report = " ".join(stemmed_words)

    return report


# Applying process_text function on Train and Test data for cleaning of text
Prediction_Train_data['report'] = Prediction_Train_data['report'].map(lambda x: process_text(x))
Prediction_Test_data['report'] = Prediction_Test_data['report'].map(lambda x: process_text(x))


# Splitting data for Training and testing

Sentiment_train = Prediction_Train_data['Sentiment']
Report_train = Prediction_Train_data['report']

Sentiment_test = Prediction_Test_data['Sentiment']
Report_test = Prediction_Test_data['report']

    
# One-Hot Encoding of Sentiment_Train

Sentiment_train = array(Sentiment_train)

# integer encode
label_encoder = LabelEncoder()
Sentiment_train_integer_encoded = label_encoder.fit_transform(Sentiment_train)

# binary encode
onehot_encoder = OneHotEncoder(sparse=False)
Sentiment_train_integer_encoded = Sentiment_train_integer_encoded.reshape(len(Sentiment_train_integer_encoded), 1)
Sentiment_train_onehot_encoded = onehot_encoder.fit_transform(Sentiment_train_integer_encoded)


# One-Hot Encoding of Sentiment_Test

Sentiment_test = array(Sentiment_test)

# integer encode
label_encoder = LabelEncoder()
Sentiment_test_integer_encoded = label_encoder.fit_transform(Sentiment_test)

# binary encode
onehot_encoder = OneHotEncoder(sparse=False)
Sentiment_test_integer_encoded = Sentiment_test_integer_encoded.reshape(len(Sentiment_test_integer_encoded), 1)
Sentiment_test_onehot_encoded = onehot_encoder.fit_transform(Sentiment_test_integer_encoded)


#print("Sentiment_Train shape after one-hot encoding : ",Sentiment_train_onehot_encoded.shape,"  "
  #    ,"Sentiment_Test shape after one-hot encoding : ",Sentiment_test_onehot_encoded.shape,"\n")

# Tokenize and Create Sequence For Train set

tokenizer = Tokenizer(num_words = 10000)
tokenizer.fit_on_texts(Report_train)

Report_train_sequences = tokenizer.texts_to_sequences(Report_train)
Report_train_padded = pad_sequences(Report_train_sequences, maxlen=100, padding='post', truncating='post')             # maxlen is the size of words in a review here it is 100


# Tokenize and Create Sequence For Test set

Report_test_sequences = tokenizer.texts_to_sequences(Report_test)
Report_test_padded = pad_sequences(Report_test_sequences, maxlen=100, padding='post', truncating='post')


print("Report_Train shape after padding : ",Report_train_padded.shape,"  ","Report_Test shape after padding: ",Report_test_padded.shape)

Sentiment_labels = ['Negative', 'Neutral', 'Positive']              #  0:Negative   1: Neutral  2:Positive


# Defining the LSTM model

model = Sequential()
model.add(Embedding(10000, 100, input_length=100))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(64, activation='sigmoid'))       
model.add(Dense(32, activation='sigmoid'))         
model.add(Dense(3, activation='softmax'))   

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

num_epochs = 10
batch_size = 128

# Train the model
history = model.fit(Report_train_padded, Sentiment_train_onehot_encoded,
                         validation_split=0.1, batch_size = batch_size, epochs= num_epochs)
                        


print("\n","****************MODEL EVALUATION ************************\n")




# Model Evaluation on Test data

test_loss,test_acc = model.evaluate(Report_test_padded, Sentiment_test_onehot_encoded)

print("\n Evaluated model accuracy on test data :",test_acc)

seconds= time.time()
time_stop = time.ctime(seconds)
print("\n","stop time:", time_stop,"\n")




# Predict the values from the Test dataset
Sentiment_pred = model.predict(Report_test_padded)
# Convert predictions classes to one hot vectors 
Sentiment_pred_classes = np.argmax(Sentiment_pred,axis = 1) 
# computing the confusion matrix
confusion_mtx = confusion_matrix(Sentiment_test_integer_encoded, Sentiment_pred_classes) 


#Printing Classification Report

print(classification_report(Sentiment_test_integer_encoded, Sentiment_pred_classes, target_names = Sentiment_labels))

accuracy = accuracy_score(Sentiment_test_integer_encoded, Sentiment_pred_classes)
print('Accuracy: %f' % accuracy)


cohen_score = cohen_kappa_score(Sentiment_test_integer_encoded, Sentiment_pred_classes)
print('Cohen_score: %f' % cohen_score)


# Training and validation curves

# Plot the loss and accuracy curves for training and validation 
fig, ax = plt.subplots(2,1)
ax[0].plot(history.history['loss'], color='b', label="Training loss")
ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['accuracy'], color='b', label="Training accuracy")
ax[1].plot(history.history['val_accuracy'], color='r',label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)

plt.show()



# Defining function for plotting confusion matrix  

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


# plot the confusion matrix
plot_confusion_matrix(confusion_mtx, classes = range(3)) 



