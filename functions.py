import pandas as pd
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Dropout, LSTM, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Flatten

from wordcloud import STOPWORDS

dumblist = ['co', 'amp', '&amp', 'will',' ','-','', 'via', '??' ]

# Function which calculates frequency of words but with removed stopwords.
def word_frex_without_stop(inputText):
    freqDict = {}

    for sentence in inputText:
        sentence = sentence.lower()
        sentence = sentence.split(' ')
        for word in sentence:
            if ((word not in STOPWORDS) & (word not in dumblist) & (not word.startswith( '@amp' )) & (not word.startswith( 'http' )) & (not word.startswith( 'https' )) & (not word.startswith( '@' ))):
                if (word in freqDict):
                    freqDict[word] = freqDict[word]+1
                else:
                    freqDict[word] = 1
    freqDict = pd.DataFrame(sorted(freqDict.items(), key=lambda x: x[1], reverse = True))
    freqDict.columns = ["word", "wordcount"]
    return freqDict

def remove_stop_words(inputText):

    list = []
    array = np.array([])
    for sentence in inputText:
        sentence = sentence.lower()
        sentence = sentence.split(' ')
        for word in sentence:
            if ((word not in STOPWORDS) & (word not in dumblist)  & (not word.startswith( '@amp' )) & (not word.startswith( 'http' )) & (not word.startswith( 'https' )) & (not word.startswith( '@' ))):
                list.append(word)
        list = " ".join(list)
        array = np.append(array,list)
        list = []
    new_series = pd.Series(array)
    return new_series

def sequential_mixed_model( inputShape):

    model = Sequential()
    model.add(Embedding(180, 128, input_length=inputShape))
    model.add(Conv1D(128, 5, activation='relu'))
    model.add(MaxPooling1D(pool_size=4))
    model.add(LSTM(64, dropout=0.4, recurrent_dropout=0.4))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer="Adam", metrics=['binary_accuracy'])
    return model

