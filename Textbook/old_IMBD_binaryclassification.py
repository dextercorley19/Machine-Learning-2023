import tensorflow as tf
import numpy as np


#num words means we will only be keeping the top 10000 most frequent words in the training data, 
# rare words to be discarded, allows to work with a data vector of a managable size
(train_data, train_labels), (test_data, test_labels) = tf.keras.datasets.imdb.load_data(
    num_words=5000                          
)      

train_data[0]

test_data[0]

max([max(sequence) for sequence in train_data])

#decoding the reviews back to english words - not manditory
word_index = tf.keras.datasets.imdb.get_word_index() #word index directly maps words to the index
reverse_word_index = dict(
    [(value, key) for (key, value) in word_index.items()] #reverses it and maps integer indices to words
)
decoded_review = ' '.join(
    [reverse_word_index.get(i - 3, '?') for i in train_data[0]] #decodes the review
)

#preparing the data, 
# turning lists into tensors (vectorizing the data) with numpy 

def vectorize_sequences(sequences, dimension=10000): 
    results = np.zeros((len(sequences), dimension)) #creates an all 0 matrix of shape (len(sequences), dimension)
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1. # sets specific indices of results[i] to 1s
        return results
    
x_train = vectorize_sequences(train_data) #vectorized training data
x_test = vectorize_sequences(test_data) #vectorized testing data

#vectorize labels
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

#build model with full stack of dense layers (16) and relu activation
from keras import models
from keras import layers
from keras import optimizers

from keras import losses
from keras import metrics

model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', #using rmsprop and binary crossentropy as the optimizer
              loss = 'binary_crossentropy',
              metrics=['accuracy'])

model.compile(optimizer=tf.keras.optimizers.legacy.RMSprop(learning_rate=0.001), #need to use learning_rate instead of lr
              loss = 'binary_crossentropy',
              metrics=['accuracy'])

model.compile(optimizer=tf.keras.optimizers.legacy.RMSprop(learning_rate=0.001),
              loss=losses.binary_crossentropy,
              metrics=[metrics.binary_accuracy])

#setting aside validation set of data
x_val = x_train[:10000]
partial_x_train = x_train[10000:]

y_val = y_train[:10000]
partial_y_train = y_train[10000:]

#training model
model.compile(optimizer='RMSprop',
              loss='binary_crossentropy',
              metrics=['acc'])

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val))

#history_dict = history.history
#history_dict.keys()

#matplotlib plotting

#training and validation loss
import matplotlib.pyplot as plt


history_dict = history.history
acc = history_dict['acc']
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
val_acc = history_dict['val_acc']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

#training and validation accuracy
plt.clf()
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
