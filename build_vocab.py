import nltk
import pickle
import argparse
from collections import Counter
import codecs
#vocab = Vocabulary()
class Vocabulary(object):
    """Simple vocabulary wrapper."""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):  #each word call one time in function build_vocab
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):   #not call
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):    #not call
        return len(self.word2idx)

def build_vocab(json_, threshold):
    """Build a simple vocabulary wrapper."""
    fvocab = codecs.open(json_, 'r', encoding = 'utf-8')
    fvocabLines = fvocab.readlines()
    #coco = COCO(json)
    counter = Counter()  
    #ids = coco.anns.keys()
    for i in range(len(fvocabLines)):
        caption = fvocabLines[i][:-1].lower()   #every time read a caption sentense into caption
        tokens = caption.split('  ')   #convert the caption into lower format and split it into list
        counter.update(tokens)  #update every word with times it appeared
        #print(caption)   #Snowboarder cuts his way down a ski slope.  
        #print(str(tokens))   #['snowboarder', 'cuts', 'his', 'way', 'down', 'a', 'ski', 'slope', '.']
        #print(str(counter)+'\n')  #Counter({'a': 2, '.': 2, 'console': 1, 'slope': 1, 'play': 1, 'his': 1, 'cuts': 1, 'the': 1, 'men': 1, 'way': 1, 'appear': 1, 'to': 1, 'wii': 1, 'ski': 1, 'two': 1, 'snowboarder': 1, 'down': 1, 'via': 1, 'game': 1})
        if (i+1) % 1000 == 0:    #41 line and 42 line's exiting doesn't matter
            print("[{}/{}] Tokenized the captions.".format(i+1, len(fvocabLines)))
    #for word,cnt in counter.items():    #dict_items([('a', 2), ('.', 2), ('console', 1), ('slope', 1), ('play', 1), ('his', 1)])--->['a', '.', 'console', 'slope', 'play', 'his']
    #    print(word)              #console
    #    print(str(cnt)+'\n')     # 1
    # If the word frequency is less than 'threshold', then the word is discarded.
    words = [word for word, cnt in counter.items() if cnt >= threshold]  #return a list(named words) including each word which times its appearing bigger then threshold

    # Create a vocab wrapper and add some special tokens.
    vocab = Vocabulary()
    vocab.add_word('<pad>')   #0
    vocab.add_word('<start>') #1
    vocab.add_word('<end>')   #2
    vocab.add_word('<unk>')   #3

    # Add the words to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)
    return vocab    #return a Vocabulary object including word2idx and idx2word

def main(args):
    vocab = build_vocab(json_=args.caption_path, threshold=args.threshold)  #a Vocabulary object including word2idx and idx2word
    print(vocab.word2idx)
    vocab_path = args.vocab_path
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    print("Total vocabulary size: {}".format(len(vocab)))
    print("Saved the vocabulary wrapper to '{}'".format(vocab_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--caption_path', type=str, 
                        default='./data/annotations/train_phrase.txt', 
                        help='path for train annotation file')
    parser.add_argument('--vocab_path', type=str, default='./data/vocab.pkl', 
                        help='path for saving vocabulary wrapper')
    parser.add_argument('--threshold', type=int, default=0, 
                        help='minimum word count threshold')
    args = parser.parse_args()
    main(args)