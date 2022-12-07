from nltk.tokenize import WordPunctTokenizer
import nltk
import re


class Preprocessor(object):
    
    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, p, t) for w, p, t in zip(s["Word"].values.tolist(),
                                                           s["POS"].values.tolist(),
                                                           s["Tag"].values.tolist())]
        self.grouped = self.data.groupby("Sentence #").apply(agg_func)
        self.sentences = [s for s in self.grouped]

    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None
    
    def get_only_sentences_without_pos_nerTags(self):

        return [[token[0] for token in sent] for sent in self.sentences ] 


def get_tokens_from_sentences(sentences):
    tokens = [token for sentence in sentences for token in sentence]
    return tokens


def normalize_to_sentences(text):
    re_url = re.compile(r"((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\
                        .([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*",
                        re.MULTILINE|re.UNICODE)
    # replace ips
    re_ip = re.compile(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}")

    # replace URLs
    text = re_url.sub("URL", text)
    # replace IPs
    text = re_ip.sub("IPADDRESS", text)
    # setup tokenizer
    sent_text = []
    for sentence in nltk.tokenize.sent_tokenize(text):
        # for sentence in sentences:
        sent_text.append(nltk.tokenize.word_tokenize(sentence))
    return sent_text


def word2features(sent, i):

    word = sent[i][0]
    postag = sent[i][1]

    # feature of curernt word
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],
    }

    # features for prev word, if not first word of sentence
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS'] = True

    # features for next word, if not last word of sentence
    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
        })
    # if word is last in a sentence, add  "End Of Sentence" feature
    else:
        features['EOS'] = True

    return features


def sent2features(sent):
	""" returns a "list of features" for "list of words" in sentence """
	return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
	""" returns list of label against specific token """
	return [label for token, postag, label in sent]
