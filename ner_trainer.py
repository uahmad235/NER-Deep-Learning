
from src.model import create_model
from src.tagger import predict
from src.utils import load_model_params, plot_history, save_model_params
import numpy as np


NER_MODEL_OUTPUT_PATH = "./NER/saved/My_Custom_Model3.h5"
NER_MAPPING_PATH = "./NER/saved/model_params.pkl"

class Trainer(object):

	def __init__(self, sequence, preprocessor):
		self.sObj = sequence
		self.preprocessor = preprocessor

	def build(self):
		""" build necessary components of NER system for training, this includes
		generating sequence of words and creating model with required params """

		refined_sentences = self.preprocessor.get_only_sentences_without_pos_nerTags()
		self.sObj.generate_dicts_from_vocab()
		self.sObj.generate_input_sequence(refined_sentences)
		print("printing refined_sentences now..")
		print(refined_sentences[:10])
		self.sObj.generate_output_sequence()

		self.model = create_model(self.sObj.max_seq_len, self.sObj.n_words, self.sObj.n_tags,\
									None, self.sObj.max_len_chars, self.sObj.n_chars,\
													 _word2idx=self.sObj.word2idx)

	def fit(self, _batch_size = 32, _epochs = 5):
		""" trains the model on training data """
		# generate sequence of preprocessed words
		self.history = self.model.fit([self.sObj.X_tr, np.array(self.sObj.X_char_tr)],\
								np.array(self.sObj.y_tr),batch_size=_batch_size,\
								 epochs= _epochs, validation_split=0.1, verbose=1)
		self._save_model()
		
	def _save_model(self):
		""" save model parameters after training for reuse """
		self.model.save(NER_MODEL_OUTPUT_PATH)

		model_params = {
			"max_len_chars": self.sObj.max_len_chars,
			"n_words": self.sObj.n_words,
			"max_seq_len": self.sObj.max_seq_len,
			"n_tags": self.sObj.n_tags,
			"n_chars": self.sObj.n_chars
		} 
		save_model_params(NER_MAPPING_PATH, model_params)

	@staticmethod
	def load_model_parameters():
		""" load model-specific params for model regeneration """
		model_params = load_model_params(NER_MAPPING_PATH)
		return model_params

	def evaluate(self):
		""" evaluate model on the basis of accuracy and make predictions """
		score = self.model.evaluate([self.sObj.X_te, np.array(self.sObj.X_char_te)],\
									np.array(self.sObj.y_te), verbose=1)
		print("%s: %.2f%%" % (self.model.metrics_names[1], score[1]*100))

		plot_history(self.history)
		predict(self.model, self.sObj.X_te, self.sObj.y_te,\
				self.sObj.X_char_te, self.sObj.idx2word, self.sObj.idx2tag )
