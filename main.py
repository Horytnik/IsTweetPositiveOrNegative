from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
import functions
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import plot_confusion_matrix

importedData = pd.read_csv("Data_tweets.csv", sep = ",",header = None)
importedData.columns = ['index','polarity','id', 'date', 'query', 'user', 'text']

print("Neutral messages: ",importedData.loc[importedData['polarity'] ==2])
# As there are no neutral messages I will do binary classification between positive and negative messages 0 is
# negative and 1 is positive


importedData['polarity'].replace(4,1, inplace= True)

positiveTweets = importedData.loc[importedData['polarity'] ==1]
negativeTweets = importedData.loc[importedData['polarity'] ==0]

dictTarRemStop0 = functions.word_frex_without_stop(negativeTweets['text'])
dictTarRemStop1 = functions.word_frex_without_stop(positiveTweets['text'])

plt.figure()
y_pos = np.arange(len(dictTarRemStop0.word.head(50)))
x_pos = dictTarRemStop0.wordcount.head(50)
plt.barh(y_pos, x_pos)
plt.title('Frequency without stopwords when it negative message')
plt.yticks(y_pos, dictTarRemStop0.word.head(50) )
plt.gca().invert_yaxis()
plt.show()

plt.figure()
y_pos = np.arange(len(dictTarRemStop1.word.head(50)))
x_pos = dictTarRemStop1.wordcount.head(50)
plt.barh(y_pos, x_pos)
plt.title('Frequency without stopwords when it is positive message')

plt.yticks(y_pos, dictTarRemStop1.word.head(50) )

plt.gca().invert_yaxis()
plt.show()

clearedData = functions.remove_stop_words(importedData['text'])

Xtrain, Xtest, Ytrain, Ytest = train_test_split(clearedData, importedData['polarity'], test_size= 0.33, shuffle=True, random_state= 42)


#wordVector = TfidfVectorizer(max_features=20000, lowercase=True, analyzer='word', #You can also try 'char'
#                            stop_words= 'english',ngram_range=(1,3),dtype=np.float32)
wordVector = TfidfVectorizer(binary=True)
XtrainVect = wordVector.fit_transform(Xtrain)
XtestVect = wordVector.transform(Xtest)



# Logistic Regression model
logisticRegModel = LogisticRegression(random_state = 1)
logisticRegModel.fit(XtrainVect, Ytrain)

print(logisticRegModel.score(XtestVect, Ytest))

plot_confusion_matrix(logisticRegModel, XtestVect, Ytest)
plt.show()


N = 180
tokenizer = Tokenizer(num_words=N, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(Xtrain)
word_index_train = tokenizer.word_index

X_train_mat = tokenizer.texts_to_sequences(Xtrain)
X_train_mat = pad_sequences(X_train_mat, maxlen=100)



epochs = 10
batch_size = 64
sequentialModel = functions.sequential_mixed_model(X_train_mat.shape[1])
history = sequentialModel.fit(X_train_mat, Ytrain,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_split=0.2)




print(history.history.keys())

# summarize history for accuracy
plt.plot(history.history['binary_accuracy'])
plt.plot(history.history['val_binary_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.ylim(0.5, 1)
# plt.ylim(0, 1)
plt.show()

# summarize history for loss
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
# plt.ylim(0, 0.9)
plt.ylim(0, 1)
plt.show()
print('end')





print("end")
