
import numpy as np
from seqeval.metrics.sequence_labeling import get_entities
from seqeval.metrics import f1_score, classification_report



def convert_to_vocab(arr1, idx2tag):
	""" converts list of predicted indices into text e.g., PER,ORG,GEO etc """
	arr1_vocab, arr2_vocab = [], []
	for lst in arr1:
		dummy = []
		for i in lst:
			if idx2tag[i] != "ENDPAD":
				dummy.append(idx2tag[i])

		arr1_vocab.append(dummy)
	return arr1_vocab



def calculate_precision_recall(my_pred, my_true, idx2tag):
	""" generates the whole report of accuracy metrics """
	pred_vocab = convert_to_vocab(my_pred, idx2tag)
	true_vocab = convert_to_vocab(my_true, idx2tag)
	
	score = f1_score(true_vocab, pred_vocab)
	print(' - f1: {:04.2f}'.format(score * 100))
	print(classification_report(true_vocab, pred_vocab))


def predict(model, X_te, y_te, X_char_te, idx2word, idx2tag, sentences ):
	""" predict values based on trained model """

	y_pred = model.predict([X_te, np.array(X_char_te)])
	my_pred = np.argmax(y_pred, axis = -1)
	complete_pred = np.hstack(my_pred)

	p_arr = np.argmax(y_pred, axis = -1) 

	tags_arr = []

	if idx2word is not None and idx2tag is not None:
		for sent, pred_arr in zip(sentences, p_arr):# second_arg:true
			tag_list = []
			for w , pred in zip(sent, pred_arr):
				tt = idx2tag[pred]
				tag_list.append(tt)

			tags_arr.append(tag_list)

	return p_arr, tags_arr

def _is_already_tagged(res, text):
	""" check for dulplicate entities """
	# entities = res['entities']
	for ent in res:
		if text == ent["text"]:
			return True

	return False


def build_custom_response(self, words, tags, prob = None):

    res = []

    chunks = get_entities(tags)

    for chunk_type, chunk_start, chunk_end in chunks:
	    chunk_end += 1
	    txt = ' '.join(words[chunk_start: chunk_end])
	    
	    if not _is_already_tagged(res, txt):
		    entity = {
		            'text': txt,
		            'entity_type': chunk_type,
		            'score': float(np.average(prob[chunk_start: chunk_end]))
		        }
		    res.append(entity)

    return res



def _build_response(self, words, tags, prob = None):
    # words = self.tokenizer(sent)
    res = {
        'words': words,
        'entities': [

        ]
    }
    chunks = get_entities(tags)

    for chunk_type, chunk_start, chunk_end in chunks:
        chunk_end += 1
        entity = {
            'text': ' '.join(words[chunk_start: chunk_end]),
            'type': chunk_type,
            'score': float(np.average(prob[chunk_start: chunk_end])),
            'beginOffset': chunk_start,
            'endOffset': chunk_end
        }
        res['entities'].append(entity)

    return res