from keras.models import Model, Sequential
from keras.layers import Dense, LSTM, Embedding, Input, Reshape

embedding_dim = 16

def build_encoder(vocab_size, max_length, latent_dim):
    encoder = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=max_length),
        LSTM(64, return_sequences=False),
        Dense(latent_dim, activation='sigmoid')
    ], name='encoder')
    return encoder


def build_decoder(vocab_size, max_length, latent_dim):
    inputs = Input(shape=(latent_dim,))
    x = Dense(128, activation='relu')(inputs)
    x = Dense(vocab_size * max_length, activation='relu')(x)
    x = Reshape((max_length, vocab_size))(x)
    decoder_model = Model(inputs, x, name='decoder')
    return decoder_model


def create_autoencoder(vocab_size, max_length, latent_dim):
    encoder = build_encoder(vocab_size=vocab_size, max_length=max_length, latent_dim=latent_dim)
    decoder = build_decoder(vocab_size=vocab_size, max_length=max_length, latent_dim=latent_dim)

    autoencoder = Sequential()
    autoencoder.add(encoder)
    autoencoder.add(decoder)

    autoencoder.compile(optimizer='adam', loss='categorical_crossentropy')

    return autoencoder, encoder, decoder
