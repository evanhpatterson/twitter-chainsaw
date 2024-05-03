import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import pandas as pd
from wordprop.word_indexer import generate_word_dicts
from wordprop.internal_lexicon import load_internal_lexicon
from wordprop.create_word_matrix import create_word_matrix
from wordprop.create_word_model import create_word_model
from wordprop.synsets_similarity import word_similarity_synsets
from wordprop.glove_similarity import word_similarity_glove_cos, word_similarity_glove_euc
from keras.models import Sequential
from keras.layers import Dense, Reshape, Softmax
from preprocessing import preprocess_data
from random import sample
from calc_direction_accuracy import calc_direction_accuracy
import matplotlib.pyplot as plt
from keras.optimizers import Adam


def fit_model(
        model,
        x_train: list, y_train: np.ndarray,
        val_x: list, val_y: np.ndarray,
        epochs: int, batch_size: int, sample_size: int, max_words: int):
    
    num_samples = len(x_train)
    num_batches = np.ceil(num_samples / batch_size)
    
    training_losses = np.zeros(epochs, dtype=float)
    validation_losses = np.zeros(epochs, dtype=float)

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        epoch_losses = []

        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        x_train = [x_train[i] for i in indices]
        y_train = np.array(y_train)[indices]

        for i in range(0, num_samples, batch_size):
            batch_x = x_train[i:i+batch_size]
            batch_y = y_train[i:i+batch_size]
            
            model_inputs = []
            
            for x in batch_x:
                if sample_size < len(x):
                    model_inputs.append(sample(x, sample_size))
                elif sample_size == len(x):
                    model_inputs.append(x)
                else:
                    padding = [np.zeros(max_words, dtype=int) for _ in range(sample_size - len(x))]
                    padded_x = x + padding
                    model_inputs.append(padded_x)
            
            model_inputs = np.array(model_inputs, dtype=int)
            
            loss = model.train_on_batch(model_inputs, batch_y)
            epoch_losses.append(loss)
        
        val_inputs = []
        
        # validation
        for x in val_x:
            if sample_size < len(x):
                val_inputs.append(sample(x, sample_size))
            elif sample_size == len(x):
                val_inputs.append(x)
            else:
                padding = [np.zeros(max_words, dtype=int) for _ in range(sample_size - len(x))]
                padded_x = x + padding
                val_inputs.append(padded_x)
        
        val_inputs = np.array(val_inputs, dtype=int)
        
        val_loss = model.test_on_batch(val_inputs, val_y)
        
        avg_loss = np.mean(epoch_losses)
        
        training_losses[epoch] = avg_loss
        
        validation_losses[epoch] = val_loss
        
        print(f"Average loss:    {avg_loss:.4f}")
        
        print(f"Validation loss: {val_loss:.4f}")
        
        print()
    
    return training_losses, validation_losses


def train_word_model(train_x, train_y, word_mat):
    '''
        Build and train the initial model that uses words to make predictions.
    '''
    
    word_model = create_word_model(
        word_mat=word_mat,
        num_unique_words=len(unique_words),
        internal_lexicon_size=len(internal_lexicon),
        lstm1_size=128,
        lstm2_size=64,
        extra_words=0)

    model = Sequential()
    model.add(word_model)
    model.add(Dense(6))

    model.compile(optimizer='adam', loss='mse')
    
    fit_model(model=model, x_train=train_x, y_train=train_y, epochs=50, batch_size=32, sample_size=256)

    return model


def test_model(model: Sequential, test_x, test_y, sample_size, max_words, categorical: bool):
    
    model_inputs = []
    
    for x in test_x:
        if sample_size < len(x):
            model_inputs.append(sample(x, sample_size))
        elif sample_size == len(x):
            model_inputs.append(x)
        else:
            padding = [np.zeros(max_words, dtype=int) for _ in range(sample_size - len(x))]
            padded_x = x + padding
            model_inputs.append(padded_x)
    
    model_inputs = np.array(model_inputs, dtype=int)
    
    pred = model.predict(model_inputs)

    if categorical:
        temp = (np.argmax(pred, axis=2) == np.argmax(test_y, axis=2))
        acc = np.sum(temp) / temp.size
    else:
        acc = calc_direction_accuracy(pred=pred, actual=test_y)
    
    return acc


def train_test_split(data_x, data_y, test_size=0.1):
    # Calculate the number of test samples
    num_samples = len(data_x)
    num_test = int(num_samples * test_size)
    
    # Create an index array
    indices = np.arange(num_samples)
    
    # Shuffle indices
    np.random.shuffle(indices)
    
    # Split indices into training and test sets
    test_indices = indices[:num_test]
    train_indices = indices[num_test:]
    
    # Use the indices to create training and test sets
    train_x = [data_x[i] for i in train_indices]
    test_x = [data_x[i] for i in test_indices]
    train_y = data_y[train_indices]
    test_y = data_y[test_indices]
    
    return train_x, train_y, test_x, test_y


def cross_validation_split(data_x, data_y, k=5):
    num_samples = len(data_x)
    
    # Create an index array
    indices = np.arange(num_samples)
    
    # Shuffle indices
    np.random.shuffle(indices)
    
    # Calculate the number of samples per fold
    fold_size = int(num_samples / k)
    
    # Split the indices into k folds
    folds = []
    for i in range(k):
        start = i * fold_size
        end = start + fold_size if i != k - 1 else num_samples
        test_indices = indices[start:end]
        train_indices = np.concatenate([indices[:start], indices[end:]])
        
        train_x = [data_x[j] for j in train_indices]
        test_x = [data_x[j] for j in test_indices]
        train_y = data_y[train_indices]
        test_y = data_y[test_indices]
        
        folds.append((train_x, train_y, test_x, test_y))
    
    return folds


def normalize_matrix(mat, norm_type):
    
    if norm_type == "rows":
        
        row_means = np.mean(mat, axis=1, keepdims=True)

        row_stds = np.std(mat, axis=1, keepdims=True)
        
        for i in range(len(row_stds)):
            if row_stds[i] == 0.0:
                row_stds[i] = 1.0

        return (mat - row_means) / row_stds
    elif norm_type == "cols":
        
        column_means = np.mean(mat, axis=0)

        column_stds = np.std(mat, axis=0)
        
        for i in range(len(column_stds)):
            if column_stds[i] == 0.0:
                column_stds[i] = 1.0

        return (mat - column_means) / column_stds
    else:
        return mat


def plot_losses(training_losses, validation_losses, using_wordprop, epochs):
    epochs = range(1, len(training_losses) + 1)
    
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, training_losses, label='Training Loss', marker='o', linestyle='-')
    plt.plot(epochs, validation_losses, label='Validation Loss', marker='x', linestyle='--')
    
    plt.title('Training and Validation Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    fname = "wordprop-" if using_wordprop else "random-"
    fname += f"{str(epochs)} epochs"
    plt.savefig(fname)
    plt.show()


if __name__=="__main__":
    
    epochs = 100
    
    use_word_mat = True

    categorical = True

    tweet_fpath = "data/preprocessed_tesla_tweets.csv"

    stock_fpath = "data/tesla_stock_price-delta.csv"

    data_x, data_y, input_word_index, unique_words = preprocess_data(tweet_fpath=tweet_fpath, stock_fpath=stock_fpath, max_words=64)

    # get internal lexicon word indices
    internal_lexicon = load_internal_lexicon()
    output_word_index, _ = generate_word_dicts(internal_lexicon)
    
    if use_word_mat:
        # get the word matrix
        word_mat = create_word_matrix(
            input_words=unique_words,
            input_words_dict=input_word_index,
            output_words=internal_lexicon,
            output_words_dict=output_word_index,
            similarity_func=word_similarity_glove_cos).T
        word_mat = normalize_matrix(word_mat, 'cols')
    else:
        word_mat = None

    folds = cross_validation_split(data_x=data_x, data_y=data_y, k=5)
    
    testing_accuracy = []
    
    training_losses_avg = np.zeros(epochs, dtype=float)
    validation_losses_avg = np.zeros(epochs, dtype=float)
    
    for train_x, train_y, test_x, test_y in folds:
        
        word_model = create_word_model(
            word_mat=word_mat,
            num_unique_words=len(unique_words),
            internal_lexicon_size=len(internal_lexicon),
            lstm1_size=128,
            lstm2_size=64,
            extra_words=0)

        model = Sequential()
        model.add(word_model)
        model.add(Dense(6 * 2))
        model.add(Reshape((6, 2)))
        model.add(Softmax())
        
        model.compile(optimizer=Adam(learning_rate=0.00001), loss='categorical_crossentropy')
        
        training_losses, validation_losses = fit_model(
            model=model,
            x_train=train_x, y_train=train_y,
            val_x=test_x, val_y=test_y,
            epochs=epochs, batch_size=32, sample_size=256,
            max_words=64)
        
        training_losses_avg += training_losses
        validation_losses_avg += validation_losses

        acc = test_model(model, test_x, test_y, sample_size=256, max_words=64, categorical=categorical)

        print(f"accuracy for this fold: {acc}")

        testing_accuracy.append(acc)
    
    training_losses_avg /= float(len(folds))
    validation_losses_avg /= float(len(folds))
    
    plot_losses(training_losses_avg, validation_losses_avg, using_wordprop=use_word_mat, epochs=epochs)
    
    avg_accuracy = np.mean(np.array(testing_accuracy, dtype=float))
    print(f"accuracy: {avg_accuracy}")
