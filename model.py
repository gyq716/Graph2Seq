import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Propogator(nn.Module):
    def __init__(self, node_dim):    
        super(Propogator, self).__init__()
        self.node_dim = node_dim
        self.reset_gate = nn.Sequential(
            nn.Linear(node_dim * 2, node_dim), 
            nn.Sigmoid()
        )
        self.update_gate = nn.Sequential(
            nn.Linear(node_dim * 2, node_dim),  
            nn.Sigmoid()
        )
        self.tansform = nn.Sequential(
            nn.Linear(node_dim * 2, node_dim), 
            nn.Tanh()
        )

    def forward(self, node_representation, adjmatrixs):   # ICLR2016 fomulas implementation
        a = torch.bmm(adjmatrixs, node_representation)
        joined_input1 = torch.cat((a, node_representation), 2)
        z = self.update_gate(joined_input1)
        r = self.reset_gate(joined_input1)
        joined_input2 = torch.cat((a, r * node_representation), 2)   
        h_hat = self.tansform(joined_input2)      
        output = (1 - z) * node_representation + z * h_hat
        return output

class EncoderGGNN(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(EncoderGGNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size, padding_idx = 0)   
        self.state_dim = 256
        self.n_steps = 5
        self.propogator = Propogator(self.state_dim)
        self.out1 = nn.Sequential(
            nn.Linear(self.state_dim + self.state_dim, self.state_dim),
            nn.Tanh()
        )
        self.out2 = nn.Sequential(    # this is new adding for graph-level outputs
            nn.Linear(self.state_dim + self.state_dim, self.state_dim),
            nn.Sigmoid()
        )
        self._initialization()
    # lstm 基于 RNNBse已经初始化， embed也已经随机初始化， 一般Linear，Conv需要初始化
    def _initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, images, adjmatrixs, lengths): 
        lengths = torch.Tensor(lengths).reshape(-1, 1).to(device)
        embeddings = self.embed(images).to(device)
        node_representation = embeddings
        init_node_representation = node_representation
        for i_step in range(self.n_steps):    # time_step updating
            node_representation = self.propogator(node_representation, adjmatrixs)   
        gate_inputs = torch.cat((node_representation, init_node_representation), 2)
        gate_outputs = self.out1(gate_inputs)
        features = torch.sum(gate_outputs, 1)    
        features = features / lengths

        """
        # graph-level models with soft attention
        gate_outputs1 = self.out1(gate_inputs)
        gate_outputs2 = self.out2(gate_inputs)
        gate_outputs = gate_outputs1 * gate_outputs2
        features = torch.sum(gate_outputs, 1)    # average pooling
        features = features / lengths
        """
        return features


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, max_seq_length=20):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size, padding_idx = 0)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_seg_length = max_seq_length
        
    #outputs = decoder(features, captions, lengths)
    def forward(self, features, captions, lengths):
        """Decode image feature vectors and generates captions."""
        #print(captions)
        #print(captions.shape)   # torch.Size([512, 13])
        embeddings = self.embed(captions)
        #print(embeddings.shape) # torch.Size([512, 13, 256])
        #print(features.shape)   # torch.Size([512, 256])
        #print(features.unsqueeze(1).shape)  # torch.Size([512, 1, 256])
        #print(torch.cat((features.unsqueeze(1), embeddings), 1).shape) # torch.Size([512, 14, 256])
        #print("lengths shape is {}".format(np.array(lengths).shape))   # lengths shape is (512,)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        #lengths = [length+1 for length in lengths]
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)  # <class 'torch.nn.utils.rnn.PackedSequence'>
        #print(np.sum(lengths))   # 4200   整个batch中所有句子中真实单词的个数
        #print(np.max(lengths))   # 16
        #print("packed[0] shape is {}".format(packed[0].shape))   # torch.Size([4200, 256]) 想法是正确的
        #print("packed[1] shape is {}".format(packed[1].shape))   # packed[1] shape is torch.Size([16])
        #print(type(packed))
        #print("packed shape is {}".format(packed.size()))
        hiddens, _ = self.lstm(packed)
        #print(type(hiddens))     # # <class 'torch.nn.utils.rnn.PackedSequence'>
        #print("hiddens[0] shape is {}".format(hiddens[0].shape)) # hiddens[0] shape is torch.Size([3668, 512])
        #print("hiddens[1] shape is {}".format(hiddens[1].shape)) # hiddens[1] shape is torch.Size([13])
        outputs = self.linear(hiddens[0])
        #print("outputs shape is {}".format(outputs.shape))       # outputs shape is torch.Size([3668, 9214])
        return outputs
    
    def sample(self, features, states=None):
        """Generate captions for given image features using greedy search."""
        sampled_ids = []   # 用于存储多个torch.size(128)的predicted，每一个predicted对应一个这个batch的预测句子的每个word，多个predicted对应依次的多个word
        inputs = features.unsqueeze(1)   # torch.Size([512, 1, 256])  [batch_size, time_steps, embeddingVector_size]
        """
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.embed = nn.Embedding(vocab_size, embed_size)
        """
        for i in range(self.max_seg_length):                     # max_seg_length = 20
            # 核心是输入的inputs的time_step是1
            hiddens, states = self.lstm(inputs, states)          # hiddens: (batch_size, time_steps, hidden_size)   torch.Size([512, 1, 512])代表所有time_step的隐层输出
            # len of states is 2
            # states[0]=hn shape is torch.Size([1, 512, 512])    第一个512代表batch_size,代表最后一个time_step的hn
            # states[1]=Cn shape is torch.Size([1, 512, 512])    第一个512代表batch_size,代表最后一个time_step的cn
            outputs = self.linear(hiddens.squeeze(1))            # outputs:  (batch_size, vocab_size)
            _, predicted = outputs.max(1)                        # predicted: (batch_size) is the position corresponding the max
            sampled_ids.append(predicted)   # the unit of predicted arrays is the label of the vocab_size 
            inputs = self.embed(predicted)                       # inputs: (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)                         # inputs: (batch_size, 1, embed_size)
        sampled_ids = torch.stack(sampled_ids, 1)                # sampled_ids: (batch_size, max_seq_length)
        # torch.size(128,20)  每个句子有20个word，每个word是digit，对应vocab里面word的position
        return sampled_ids
