import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Propogator(nn.Module):
    """
    Gated Propogator for GGNN
    Using LSTM gating mechanism
    """
    def __init__(self, node_edge_dim):     # 原来还有个第三个参数：n_proposals
        super(Propogator, self).__init__()

        #self.n_node = n_node
        #self.n_edge_types = n_edge_types

        self.node_edge_dim = node_edge_dim
        #self.n_proposals = n_proposals    # 复件1新加，理解为graph中结点个数,为了看node_representation，暂时先注释

        """
        self.reset_gate = nn.Sequential(
            nn.Linear(node_edge_dim * 2, node_edge_dim),
            nn.Sigmoid()
        )
        """
        self.reset_gate = nn.Sequential(
            nn.Linear(node_edge_dim, node_edge_dim), # 这里暂时还没改，相比于node_edge*2->node_edge
            nn.Sigmoid()
        )
        self.update_gate = nn.Sequential(
            nn.Linear(node_edge_dim, node_edge_dim), # 这里暂时还没改，相比于node_edge*2->node_edge 
            nn.Sigmoid()
        )
        self.tansform = nn.Sequential(
            nn.Linear(node_edge_dim, node_edge_dim), # 这里暂时还没改，相比于node_edge*2->node_edge
            nn.Tanh()
        )

    def forward(self, node_representation):
        #A_in = A[:, :, :self.n_node*self.n_edge_types]
        #A_out = A[:, :, self.n_node*self.n_edge_types:]

        #a_in = torch.bmm(A_in, state_in)
        #a_out = torch.bmm(A_out, state_out)
        #print(node_representation.shape)   # torch.Size([13, 256])
        #print(edge_representation.shape)   # torch.Size([13, 256])
        a = node_representation
        #a = node_representation     # 原来的是上面一行

        r = self.reset_gate(a)
        #print(type(r))                     # <class 'torch.Tensor'>
        #print(r.shape)                     # torch.Size([13, 256])
        z = self.update_gate(a)
        joined_input = r * node_representation   # r乘代表对应矩阵元素相乘
        #joined_input = r * node_representation   # 原来的是上一句
        h_hat = self.tansform(joined_input)      

        output = (1 - z) * node_representation + z * h_hat

        return output


class EncoderGGNN(nn.Module):
    """
    Gated Graph Sequence Neural Networks (GGNN)
    Mode: SelectNode
    Implementation based on https://arxiv.org/abs/1511.05493
    """
    def __init__(self, vocab_size, embed_size):
        super(EncoderGGNN, self).__init__()
        
        self.embed = nn.Embedding(vocab_size,embed_size)   # 这是EncoderGGNN加的
        self.state_dim = 256
        self.n_node = 256
        self.n_steps = 24
        #self.linear = nn.Linear(256,85)   # 这个线性变换之前是用来将3个结点时给关系结点考虑另外两个结点时所有的
        #print('3')
        """
        self.matrix_fix = nn.Sequential(     # 原来的里面有，但这个用不到
            nn.Linear(26, self.state_dim),
            nn.ReLU())
        """
        #self.node_fix = nn.Sequential(
            #nn.Linear(6, self.state_dim),    # 这个原来的有，没用
            #nn.ReLU())
        """
        self.matrix_transform = nn.Sequential(
            nn.Linear(self.state_dim, 1),     # 这个原来有，用来将adjmatrix每个位置的向量转换成scalar
            nn.ReLU())
        """
        #self.representation_transform = nn.Sequential(
            #nn.Linear(self.state_dim, self.n_node),
            #nn.ReLU(True))
        #print('1')
        # Propogation Model
        self.propogator = Propogator(self.state_dim)
        #print('2')
        # Output Model
        self.out1 = nn.Sequential(
            nn.Linear(self.state_dim + self.n_node, self.state_dim),
            nn.Tanh(),
            nn.Linear(self.state_dim, 256))

        #self.out2 = nn.Sequential(
            #nn.Linear(self.state_dim, 21),
            #nn.ReLU())

        self._initialization()

    def _initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, images, adjmatrixs, lengths):  #把trian中的image传进来，然后利用embed处理成node——repres
        lengths = torch.Tensor(lengths).reshape(-1, 1).to(device)
        embeddings = self.embed(images).to(device)
        #print("images shape is {}".format(images.shape))          # images shape is torch.Size([512, 5])
        #print("embedding shape is {}".format(embeddings.shape))   # embedding shape is torch.Size([512, 5, 256])
        #print("adjmatrixs shape is {}".format(adjmatrixs.shape))  # adjmatrixs shape is torch.Size([512, 5, 5])
        node_representation = embeddings
        adjacency_matrix = adjmatrixs
        #print("node_representation shape is {}".format(node_representation.shape))   # torch.Size([5, 256])
        
        init_node_representation = node_representation
        for i_step in range(self.n_steps):    # 更新结点的步骤
            in_states = []
            out_states = []
            #edge_matrix = adjacency_matrix.cuda()
            #edge_matrix = adjacency_matrix.view(-1, 26)
            #edge_matrix = self.matrix_fix(adjacency_matrix.view(-1, 26)) # 原来有
            #print(node_representation.shape)
            #node_representation = self.node_fix(node_representation)        
            #edge_representation = self.matrix_transform(edge_matrix)   # 原来有
            #print(edge_representation.view(n_proposals, n_proposals, 1).squeeze(2).shape)
            #print(edge_representation.shape)
            # 下面的原来有

            node_representation = torch.bmm(adjacency_matrix, node_representation)
            #print(adjacency_matrix.shape, node_representation.shape, node_representation.shape)
            #print(node_representation.shape)
            node_representation = self.propogator(node_representation)   
            #print(node_representation)
        #join_state = torch.cat((prop_state, annotation), 2)

        #output = self.out(join_state)
        #output = output.sum(2)
        gate_inputs = torch.cat((node_representation, init_node_representation.type_as(node_representation)), 2)
        #print("gate_inputs shape is {}".format(gate_inputs.shape))     # gate_inputs shape is torch.Size([128, 5, 512])
        #gate_output = F.softmax(self.out1(gate_input.cuda()), 1)
        gate_outputs = self.out1(gate_inputs)
        #print("gate_outputs shape is {}".format(gate_outputs.shape))   # gate_outputs shape is torch.Size([128, 5, 256])
        
        gate_outputs = torch.sum(gate_outputs, 1)    # average pooling
        gate_outputs = gate_outputs / lengths
        #print("gate_outputs shape is {}\n".format(gate_outputs.shape))   # gate_outputs shape is gate_outputs shape is torch.Size([128, 256])
        return gate_outputs

#decoder = DecoderRNN(args.embed_size, args.hidden_size, len(vocab), args.num_layers).to(device)
#            D of word_embedding_vectors, D of lstm_hidden_states
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, max_seq_length=20):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
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
