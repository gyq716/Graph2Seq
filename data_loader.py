import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import numpy as np
import nltk
from build_vocab import Vocabulary
import codecs

# This file is the most important.


class MyDataset(data.Dataset):

    def __init__(self, root, caption_path, relationship_path, vocab, vocab_image, ids):
        """Set the path for images, captions and vocabulary wrapper.
        
        Args:
            root: image directory.
            json: coco annotation file path.
            vocab: vocabulary wrapper.
            transform: image transformer.
        """
        self.caption_path = codecs.open(caption_path, 'r', encoding = 'utf-8')
        self.root = codecs.open(root, 'r', encoding = 'utf-8')
        self.relationship_path = codecs.open(relationship_path, 'r', encoding = 'utf-8')
        self.fcaptionLines = self.caption_path.readlines()
        self.fimageLines = self.root.readlines()
        self.frelationshipLines = self.relationship_path.readlines()
        self.vocab = vocab
        self.vocab_image = vocab
        self.ids = ids
    def __getitem__(self, index):  #Returns one data pair (image and caption).
        vocab = self.vocab
        vocab_image = self.vocab

        caption_sentence = self.fcaptionLines[index][:-1].lower()
        image_graph = self.fimageLines[index][:-1].lower()
        relationship = self.frelationshipLines[index][:-3].lower()
        
        
        #print(image.shape)   # torch.Size([3, 224, 224]) 
        # Convert caption (string) to word ids.
        tokens = caption_sentence.split('  ')
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)

        tokensImage = image_graph.split('  ')
        i = 0
        dic = {}
        for obj_ in tokensImage:
            dic[obj_] = i
            i += 1
        rel = relationship.split('  ')
        length = len(dic)
        matrix = np.zeros((length,length),dtype = int)
        for m in range(0,length):  # 对角矩阵，保留本结点表征
            matrix[m][m] = 1
        for r in rel:
            rs = r.split(',')
            if len(rs) == 1:     # 只有单独object，无att和relationship
                if rs[0] in dic.keys():
                    loc = dic[rs[0]]
                    if loc >= length:   # 处理数据异常时
                        pass
                    else:
                        matrix[loc][loc] = 1
                else:
                    pass
            if len(rs) == 2:
                if rs[0] in dic.keys() and rs[1] in dic.keys():
                    loc1 = dic[rs[0]]    # 有attribute或者relationship
                    loc2 = dic[rs[1]]    # 处理成对称矩阵（无向）
                    if loc1 >= length or loc2 >= length:  # 处理数据异常时
                        pass
                    else:
                        matrix[loc1][loc2] = 1
                        matrix[loc2][loc1] = 1
                else:
                    pass
        #print(type(matrix))        # <class 'numpy.ndarray'>
        #matrix = np.array(matrix)
        caption2 = []
        caption2.extend([vocab_image(token) for token in tokensImage])
        image = torch.Tensor(caption2)


        #print(index) 
        #print(image)                       # tensor([  760.,   834.,   366.,  3728.])
        #print(image.shape)                 # torch.Size([3]/[4]/[2])
        #print(caption_sentence)            # window  on  the  building
        #print(caption)                     # [1, 7604, 1856, 6287, 4304, 2]
        #print(torch.Tensor(caption).shape) # torch.Size([6])
        #print(matrix)     
        #print(matrix.shape)                # torch.Size([3, 3]) 和image(graph)的结点个数一样
        #print('\n')
        return image, target, matrix

    def __len__(self):
        return self.ids


def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).
    
    We should build custom collate_fn rather than using default collate_fn, 
    because merging caption (including padding) is not supported in default.

    Args:
        data: list of tuple (image, caption). 
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions, matrixs = zip(*data)     
    #print(len(images))        # tuple，128
    #print(images[0])          # the number of nodes of graph tensor([ 2843.,  1602.,  8507.,  4304.])
    #print(images[0].shape)    # torch.Size([3])    
    #print(len(captions))      # tuple，128
    #print(len(matrixs))       # tuple，128
    #print(matrixs[0])
    #print(matrixs[0].shape)   # torch.Size([3,3])
    #print(type(matrixs[0]))    # <class 'torch.Tensor'>
    #print(captions[0].shape)  # torch.Size([11])
    # Merge images (from tuple of 3D tensor to 4D tensor).
    lengths_image = [len(image) for image in images]
    targets_image = torch.zeros(len(images), max(lengths_image)).long()
    length_pad = max(lengths_image)    # use
    #print(type(targets_image))   # <class 'torch.Tensor'>
    #print(lengths_image)
    #print(length_pad)     # 10
    # convert matrixs from tuple to np.array and use np.pad to fill the matrixs
    matrixs_pad = []
    for j, img in enumerate(images):
        end_img = lengths_image[j]
        targets_image[j, :end_img] = img[:end_img]
        #matrixs[j] = np.array(matrixs[j])   # convert matrixs which type is torch.Tensor to np.array,then use np.pad to pad the matrixs conveted. 
        #print(matrixs[j].shape)
        newmatrix = np.pad(matrixs[j],((0,length_pad-len(matrixs[j])),(0,length_pad-len(matrixs[j]))),'constant',constant_values=(0,0))
        matrixs_pad.append(newmatrix)
    matrixs_pad = torch.Tensor(matrixs_pad)
    #print(images.shape)    # torch.Size([128, 255])
    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]    
    #print(targets.shape)   # torch.Size([128, 12])
    #print(lengths)   #[12, 12, 12, 11, 11, 11, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
    #print(np.array(lengths).shape) #(128,)   
    #print(targets_image.shape)    # torch.Size([128, 10])
    #print(targets_image[0])      
    #print(targets_image[0].shape)    # torch.Size([16])
    #print(targets[0].shape)          # torch.Size([11])
    #print('\n')
    return targets_image, lengths_image, targets, lengths, matrixs_pad


#data_loader = get_loader(args.image_dir, args.caption_path, vocab, transform, args.batch_size, shuffle=True, num_workers=args.num_workers) 
#                           resized2014      caption.json     pickle  compose      128                                      2
def get_loader(root, caption_path, relationship_path, vocab, vocab_image, batch_size, shuffle, num_workers, ids):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    # caption dataset
    mydataset = MyDataset(root=root, #image_dir
                       caption_path=caption_path, #caption_path
                       relationship_path=relationship_path,
                       vocab=vocab,
                       vocab_image = vocab,
                       ids = ids)  #vocab
    
    # Data loader for dataset
    # This will return (images, captions, lengths) for each iteration.
    # images: a tensor of shape (batch_size, 3, 224, 224).
    # captions: a tensor of shape (batch_size, padded_length).
    # lengths: a list indicating valid length for each caption. length is (batch_size).
    data_loader = torch.utils.data.DataLoader(dataset=mydataset, 
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader
