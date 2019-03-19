import torch
import numpy as np 
import argparse
import pickle 
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from build_vocab import Vocabulary
from data_loader import get_loader
from build_vocab import Vocabulary
from model import EncoderGGNN, DecoderRNN,Propogator
from torch.nn.utils.rnn import pack_padded_sequence
import codecs
from BLEUcalcu import BLEU, ngrams, count_occurences, modified_precision, brevity_penalty, bleu_score
from tensorboardX import SummaryWriter

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
writer = SummaryWriter('./testVisual319')

def main(args):

    with open(args.vocab_path, 'rb') as f: #in build_vocab function,pickle.dump(f),then here load equals a dict vocab
        vocab = pickle.load(f)
   
    # Load vocabulary wrapper
    vocab_image = vocab
    # Build models
    encoder = EncoderGGNN(len(vocab_image),256).to(device)  # eval mode (batchnorm uses moving mean/variance)
    decoder = DecoderRNN(args.embed_size, args.hidden_size, len(vocab), args.num_layers).to(device)
    data_loader = get_loader(args.image_dir, args.caption_path, args.relationship_path, vocab, vocab_image,
                             args.batch_size, shuffle=True, num_workers=args.num_workers, ids=438430)
    criterion = nn.CrossEntropyLoss()

    
    # Test the models
    total_step = len(data_loader)  # numbers of batchs
    print('There is total {} batch in test data set\n'.format(total_step))
    for epoch in range(1, args.num_epochs):
        encoder.load_state_dict(torch.load('./models/encoder-{}.ckpt'.format(epoch)))
        decoder.load_state_dict(torch.load('./models/decoder-{}.ckpt'.format(epoch)))
        with torch.no_grad():
            encoder.eval()
            decoder.eval()
            f_samp = codecs.open('./candidate.txt', 'w', encoding = 'utf-8')
            f_ref = codecs.open('./reference.txt', 'w', encoding = 'utf-8')
            sum_loss = 0.0
            for i, (images, lengths_images, captions, lengths, adjmatrixs) in enumerate(data_loader):
                images = images.to(device)
                captions = captions.to(device)
                adjmatrixs = adjmatrixs.to(device)
                targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
                features = encoder(images, adjmatrixs, lengths_images)
                outputs = decoder(features, captions, lengths)
                loss = criterion(outputs, targets)
                sum_loss += loss
                sampled_ids = decoder.sample(features)
                #print(sampled_ids.shape)  # torch.Size([512, 20])observe the shape[0] whether is the batch
                #print(type(sampled_ids))  # <class 'torch.Tensor'>
                #print(captions.shape)     # torch.Size([512, 12])
                #print(type(captions))     # <class 'torch.Tensor'>  
                #samp_sentences = []
                #ref_sentences = []
                #print('Traslation')
                for j in range(len(sampled_ids)):   # len(sampled_ids) is  batch_size
                    sampled_id = sampled_ids[j]
                    sampled_id = sampled_id.cpu().numpy()
                    sampled_caption = []
                    for word_id in sampled_id:   # word_id is a np.int64 scalar
                        word = vocab.idx2word[word_id]
                        sampled_caption.append(word)
                        if word == '<end>':
                            break
                    sentence = ' '.join(sampled_caption)
                    f_samp.write(sentence + ' . \n')
                    #samp_sentences.append(sentence)

                    caption_len = lengths[j]
                    caption = captions[j].cpu().numpy()
                    ref_caption = []
                    for l in range(caption_len):
                        word_id = caption[l]
                        word = vocab.idx2word[word_id]
                        ref_caption.append(word)
                    reference = ' '.join(ref_caption)
                    f_ref.write(reference + ' . \n')
                    #ref_sentences.append(reference)
                    #print the generated caption
        f_samp.close()
        f_ref.close()  
        score, pn = BLEU('./candidate.txt', './reference.txt')
        sum_loss /= total_step
        print('loss is {:.6f}\tBLEU is {:.8f}\n'.format(sum_loss, score))
        writer.add_scalar('test/loss', sum_loss, epoch)
        writer.add_scalar('test/BLEU', score, epoch)
        writer.add_scalar('test/n-grams', {'bleu-1' : pn[0], 'bleu-2' : pn[1], 'bleu-3' : pn[2], 'bleu-4' : pn[3]}, epoch)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab_path', type=str, default='./data/vocab.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--vocab_image_path', type=str, default='./data/vocab.pkl')
    parser.add_argument('--image_dir', type=str, default='./data/annotations/test_object.txt', help='directory for resized images')
    parser.add_argument('--caption_path', type=str, default='./data/annotations/test_phrase.txt', help='path for train annotation json file')
    parser.add_argument('--relationship_path',type=str,default='./data/annotations/test_rela.txt',help='path for adjacency matrix')
    parser.add_argument('--num_epochs', type=int, default= 425)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--embed_size', type=int , default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_workers', type=int, default=4)  #the numbers of processes
    parser.add_argument('--num_layers', type=int , default=1, help='number of layers in lstm')
    args = parser.parse_args()
    print(args)
    main(args)