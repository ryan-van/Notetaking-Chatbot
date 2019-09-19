import numpy as np
import sklearn
import random
import nltk
from nltk.corpus import stopwords, wordnet as wn
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

class Conversation:
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    take_words = ['take', 'remind', 'copy', 'keep', 'say', 'note', 'record']
    remove_words = ['remove', 'delete', 'erase', 'omit', 'discard', 'separate']
    placement = ['first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'tenth']
    last_words = ['last', 'final', 'recent', 'new', 'newest']
    retrieve_words = ['retrieve', 'repeat', 'get', 'be', 'fetch', 'recover', 'was']
    total_words = ['many', 'number', 'amount']

    note_responses = ['Sure, what would you like me to take note of?', "What is your note?", "OK, what would you like me to say?", "What do you want to record?"]
    remove_responses = ["Okay, I've removed that", "Removed.", "Okay, it's gone"]
    retrieve_responses = ["Your last note was: ", "This was your last note: "]

    notes = []
    record_note = False

    def __init__(self):
        '''
        The init function: Here, you should load any
        PyTorch models / word vectors / etc. that you
        have previously trained.
        '''
        pass

    def respond(self, sentence):
        '''
        This is the only method you are required to support
        for the Conversation class. This method should accept
        some input from the user, and return the output
        from the chatbot.
        '''
        if (not self.record_note):
            filtered = self.filter(sentence)
            lemmatized = self.lemmatize(filtered)
            temp = self.categorize(lemmatized)
            return self.get_response(temp.cat())
        else:
            self.notes.append(sentence)
            self.record_note = False
            return self.get_response('record')

    def filter(self, s):
        tokens = word_tokenize(s)
        return [t.lower() for t in tokens]

    def lemmatize(self, s):
        tagged = nltk.pos_tag(s)
        translated = []
        for pair in tagged:
            token, tag = pair
            if tag.startswith('J'):
                translated.append((token, wn.ADJ))
            elif tag.startswith('V'):
                translated.append((token, wn.VERB))
            elif tag.startswith('N'):
                translated.append((token, wn.NOUN))
            elif tag.startswith('R'):
                translated.append((token, wn.ADV))
            else:
                translated.append((token, None))
        lemmatized = []
        for pair in translated:
            token, tag = pair
            if tag:
                lemmatized.append(self.lemmatizer.lemmatize(token, pos=tag))
            else:
                lemmatized.append(self.lemmatizer.lemmatize(token))
        return lemmatized

    def categorize(self, lst):
        ca = ''
        id = ''
        pl = ''
        for i in lst:
            if i in self.take_words:
                ca = 'take'
                break
            elif i in self.remove_words:
                ca = 'remove'
                break
            elif i in self.retrieve_words:
                ca = 'retrieve'
                break
            elif i in self.total_words:
                ca = 'total'
                break;
            elif i in self.placement:
                pl = i
            elif i in self.last_words:
                id = i
        if ca == '':
            ca = self.analyze(lst)
        return Node(ca, id, pl)

    # perform sentiment analysis
    def analyze(self, lst):
        # remove stop words
        lst = [i for i in lst if i not in self.stop_words]
        flag = ''
        #convert words in list to wordnet variables
        try:
            temp = []
            for i in lst:
                max_take = max([wn.synsets(word, pos='v')[0].lch_similarity(wn.synsets(i, pos='v')[0]) for word in self.take_words])
                max_remove = max([wn.synsets(word, pos='v')[0].lch_similarity(wn.synsets(i, pos='v')[0]) for word in self.remove_words])
                max_retrieve = max([wn.synsets(word, pos='v')[0].lch_similarity(wn.synsets(i, pos='v')[0]) for word in self.retrieve_words])
            categories = ['take', 'remove', 'retrieve']
            maxes = [max_take, max_remove, max_retrieve]
            print(max(maxes))
            if (max(maxes) >= 2):
                flag = categories[np.argmax(maxes)]
        except:
            pass
        return flag


    def get_response(self, flag):
        if flag == "take":
            self.record_note = True
            return random.choice(self.note_responses)
        elif flag == "remove":
            self.notes.pop()
            return random.choice(self.remove_responses)
        elif flag == "retrieve":
            return random.choice(self.retrieve_responses) + self.notes[len(self.notes) - 1]
        elif flag == 'record':
            return "Okay, I've recorded that."
        elif flag == 'total':
            return "You have " + str(len(self.notes)) + " notes."
        else:
            return "Sorry, I don't recognize what you said."

class Node:
    def __init__(self, category, identifier=None, placement=None):
        self.category = category
        self.identifier = identifier
        self.placement = placement
    def cat(self):
        return self.category
    def iden(self):
        return self.identifier
    def place(self):
        return self.placement
