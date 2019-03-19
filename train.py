import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from data_loader import get_loader 
from build_vocab import Vocabulary
from model import DecoderRNN,EncoderGGNN,Propogator
from torch.nn.utils.rnn import pack_padded_sequence
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR


writer = SummaryWriter('./trainTensorBoard319')

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(args):   #all args is default) as main() function indicating
    # Create model directory
    if not os.path.exists(args.model_path): # models/
        os.makedirs(args.model_path)
    
    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f: #in build_vocab function,pickle.dump(f),then here load equals a dict vocab
        vocab = pickle.load(f)
    # with open(args.vocab_image_path, 'rb') as fi: #in build_vocab function,pickle.dump(f),then here load equals a dict vocab
    #    vocab_image = pickle.load(fi)
    vocab_image = vocab
    # print(vocab.word2idx)
    # Build data loader
    data_loader = get_loader(args.image_dir, args.caption_path, args.relationship_path, vocab, vocab_image, 
                             args.batch_size, shuffle=True, num_workers=args.num_workers, ids=500000) 
    
    # Build the models
    encoder = EncoderGGNN(len(vocab_image), 256).to(device)
    decoder = DecoderRNN(args.embed_size, args.hidden_size, len(vocab), args.num_layers).to(device)
    # encoder = nn.DataParallel(encoder)
    # decoder = nn.DataParallel(decoder)
    if args.resume == True:
        encoder.load_state_dict(torch.load(args.encoder_path))
        decoder.load_state_dict(torch.load(args.decoder_path))
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    # params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
    params = list(decoder.parameters()) + list(encoder.parameters())
    optimizer = torch.optim.Adam(params, lr=args.learning_rate, weight_decay = 1e-5)
    #scheduler = ReduceLROnPlateau(optimizer, 'min', factor = 0.5, patience = 100, verbose = True)
    scheduler =  StepLR(optimizer, step_size = 100, gamma = 0.7)
    # Train the models
    total_step = len(data_loader)  # numbers of tuples (images,captions,lengths)  # representation of the number of batch of all of the train_data_set
    print('There is total {} batch in test data set\n'.format(total_step))
    for epoch in range(0, args.num_epochs):
        scheduler.step()
        encoder.train()
        decoder.train()
        sum_loss = 0.0
        for i, (images, lengths_images, captions, lengths, adjmatrixs) in enumerate(data_loader):
            #print('One Batch')
            # Set mini-batch dataset
            images = images.to(device) 
            adjmatrixs = adjmatrixs.to(device)
            #print(type(images))      
            #print(images.shape)    # torch.Size([128, 3])
            #print(lengths_images)  # all is 3 
            captions = captions.to(device)   #[128,max_length] torch.Size([128, 23]),torch.Size([128, 20])....
            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
            #print(images)
            #print(images.cpu().detach().numpy())   
            #print(images.cpu().detach().numpy()[0])
            #print(images.cpu().detach().numpy().shape)   #(128,3,85)
            #print(len(images.cpu().detach().numpy()))  #128
            #print(images.shape)   # torch.Size([128, 3, 85])
            #print(captions.shape) # torch.Size([128, 19])       
            #print("targets shape is {}".format(targets.shape))  # targets shape is torch.Size([3668])
            # Forward, backward and optimize
        
            features = encoder(images, adjmatrixs, lengths_images)  
            #print(features.shape)   #torch.Size([128, 256])
            outputs = decoder(features, captions, lengths)   # lengths are original truth length of captions
            #print(features.shape)  #torch.Size([128, 256])
            #print(outputs)
            #print(outputs.shape)   #torch.Size([931, 7553])
            loss = criterion(outputs, targets)     # crossentroyloss，和single-label classification 计算一样
            decoder.zero_grad()
            encoder.zero_grad()
            loss.backward()
            optimizer.step()
            sum_loss += loss.item()
        sum_loss /= total_step
        #scheduler.step(sum_loss)
        writer.add_scalar('train/loss', sum_loss, epoch)
        writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], epoch)
        # Print log info 
        print('Epoch [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                .format(epoch, args.num_epochs, sum_loss, np.exp(sum_loss))) 
        # Save the model checkpoints
        torch.save(decoder.state_dict(), os.path.join(args.model_path, 'decoder-{}.ckpt'.format(epoch+1)))
        torch.save(encoder.state_dict(), os.path.join(args.model_path, 'encoder-{}.ckpt'.format(epoch+1)))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./models/' , help='path for saving trained models')
    parser.add_argument('--vocab_path', type=str, default='./data/vocab.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--vocab_image_path', type=str, default='data/vocab.pkl')
    parser.add_argument('--image_dir', type=str, default='./data/annotations/train_object.txt', help='directory for resized images')
    parser.add_argument('--caption_path', type=str, default='./data/annotations/train_phrase.txt', help='path for train annotation json file')
    parser.add_argument('--relationship_path',type=str,default='./data/annotations/train_rela.txt',help='path for adjacency matrix')
    
    # Model parameters
    parser.add_argument('--encoder_path', type=str, default='./models/encoder-312.ckpt', help='path for trained encoder')
    parser.add_argument('--decoder_path', type=str, default='./models/decoder-312.ckpt', help='path for trained decoder')
    parser.add_argument('--embed_size', type=int , default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1, help='number of layers in lstm')
    parser.add_argument('--resume', default=False, help = 'whether to resume model')
    parser.add_argument('--num_epochs', type=int, default=5000)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--num_workers', type=int, default=4)  #the numbers of processes
    parser.add_argument('--learning_rate', type=float, default=0.001)
    args = parser.parse_args()
    print(args)  #all args is same(default) as above
    main(args)
