
import pandas as pd
import pickle
import matplotlib.pyplot as plt


def save_load_word_idx(filename, word2idx = None, load= False, save= False):
	if load:
		with open(filename,"rb") as f:
			data = pickle.load(f)
		return data
	elif save:
		with open(filename,"wb") as f:
			pickle.dump(word2idx,f)


def sort_word2idx(word2idx):
	""" return sorted words """
	return sorted(word2idx.items(), key=operator.itemgetter(1))


def save_model_params(file_name, dict_to_save):

	with open(file_name,"wb") as f:
		pickle.dump(dict_to_save,f)


def load_model_params(file_name):
	with open(file_name,"rb") as f:
		data = pickle.load(f)
	return data


def plot_history(history):
	""" plot model history on graph """
	hist = pd.DataFrame(history.history)
	plt.figure(figsize=(12,12))
	plt.plot(hist["acc"])
	plt.plot(hist["val_acc"])
	plt.legend()
	plt.show()


def load_dataset(path_to_dataset = None):

	if not path_to_dataset:
		raise Exception("Must provide path to a csv file to train..")
		return

	data = pd.read_csv(path_to_dataset, encoding="latin1") 

	data = data.fillna(method="ffill")

	# prints last 10 rows of data
	print(data.tail(10))

	#extract words from dataset
	words = list(set(data["Word"].values))

    # words.append("UNK")
	# words.append("ENDPAD")
    
	# extract tags from words
	tags = list(set(data["Tag"].values))
	pos = list(set(data["POS"].values))

	return data , words, tags, pos
