import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding
from old.load_imdb_sequences import *


# Build the model
model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=max_length),
    LSTM(64, return_sequences=False),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

num_epochs = 10
model.fit(train_padded, train_labels, epochs=num_epochs, validation_data=(test_padded, test_labels))

loss, accuracy = model.evaluate(test_padded, test_labels)
print(f'Loss: {loss}, Accuracy: {accuracy}')
