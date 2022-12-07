
# for variational RNN, we use SAME dropout mask(0.5) at each timestep including recurrent layers.
# applying DIFFERENT mask on input & output layers deteriorates it's performance(Naive dropout).
# embedding dropout MUST be applied AFTER conversion of words in embeddings
# embedding dropout drops word types from word-sequences/word-embeddings
# with NO embedding dropout, model overfits and needs early stopping
# learning rate DECAY also performs best
 
import numpy as np
from keras.models import Input, Model
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional, concatenate
from keras.layers import Conv1D, Flatten,GlobalMaxPooling1D
from keras_contrib.layers import CRF


def create_model(max_len, n_words, n_tags, pos_tags ,\
					max_len_chars, n_chars, _word2idx=None):
	# input and embedding for words
	word_in = Input(shape=(max_len,))

	if _word2idx != None:
		embeddings_index = prepare_glove_embeddings()
		embedding_matrix = leverage_embeddings(_word2idx, embeddings_index)
		emb_word = Embedding(input_dim=n_words + 2, output_dim=100,weights=[embedding_matrix],
	                     input_length=max_len, mask_zero=True)(word_in)
	else:
		emb_word = Embedding(input_dim=n_words + 2, output_dim=100,
	                     input_length=max_len, mask_zero=True, name="emb_word")(word_in)
	# input and character-embeddings
	char_in = Input(shape=(max_len, max_len_chars))
	emb_char = TimeDistributed(Embedding(input_dim=n_chars + 2, output_dim=50,
	                           input_length=max_len_chars, name="emb_char"))(char_in)
	conv_1d = TimeDistributed(Conv1D(filters=50, kernel_size=3,padding="valid", activation="relu", name="Conv1D"))(emb_char)
	conv_1d = TimeDistributed(Dropout(0.4))(conv_1d)
	maxpool1d = TimeDistributed(GlobalMaxPooling1D())(conv_1d)
	char_enc = TimeDistributed(Flatten())(maxpool1d)
	word_embeddings = concatenate([emb_word, char_enc])
	main_lstm = Bidirectional(LSTM(units=100, return_sequences=True,
	                               recurrent_dropout=0.6))(word_embeddings)
	main_lstm = Dropout(0.5)(main_lstm)
	z = Dense(100, activation='tanh')(main_lstm)
	crf = CRF(n_tags + 1, sparse_target=False)
	loss = crf.loss_function
	pred = crf(z)

	model = Model(inputs=[word_in, char_in], outputs=pred)
	model.compile(optimizer = "adam", loss = loss, metrics = [crf.accuracy])

	return model


def prepare_glove_embeddings(path = "./NER/data/glove.6B.100d.txt"):
	""" loads Google's glove 100-D word-embeddings for training """
	print("started reading glove ...")
	embeddings_index = {}
	with open(path) as f:
		for line in f:
			values = line.split()
			word = values[0]
			coefs = np.asarray(values[1:], dtype='float32')
			embeddings_index[word] = coefs
			# f.close()
	print('Found %s word vectors.' % len(embeddings_index))
	return embeddings_index


def leverage_embeddings(_word2idx, embeddings_index):
	""" prepare word-embeddings for training """
	embedding_matrix = np.zeros((len(_word2idx), 100))
	for word, i in _word2idx.items():
	    embedding_vector = embeddings_index.get(word)
	    if embedding_vector is not None:
	        # words not found in embedding index will be all-zeros.
	        embedding_matrix[i] = embedding_vector

	return embedding_matrix
