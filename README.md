# Graph2Seq
GGNN-LSTM,using scene graph to generate captions

## 本代码是历时多半年的毕设研究。包含了图神经网络到序列学习的任务，目的是输入一个graph，输出对应的描述caption。

## 本项目基于encoder-decoder,graph network相当于encoder, lstm作decoder，模块包含图神经网络、LSTM、attention以及整个数据预处理、数据生成、evaluation的各个细节。

放出来的目的是为了让对图神经网络和encoder-decoder based attention感兴趣的人更快的了解这方面的知识。

## 打印版论文.pdf 是我的本科毕设论文，其详细介绍了每个模块的实现细节和图解表示，相信你如果能认真阅读的话，以上每个实现必将了如指掌。
框架草图演示了更多的细节。
![image](https://github.com/nwpuhq/Graph2Seq/blob/master/%E6%A1%86%E6%9E%B6%E8%8D%89%E5%9B%BE.png)


## 运行环境 python3.5+pytorch0.4.1

# 数据文件解释  data/annotation：
  train_object.txt或者test_object.txt 是graph中的node,代表图中的节点，其中每一行是一个graph的object，relationship和attribute都当作graph的node，比如clock  by  sidewalk就代表代表当前graph共有3个节点；
  train_rela.txt或者test_rela.txt 是graph的关系relationship，每一行是一个graph中的所有的relationship，比如sidewalk,by  by,clock代表sidewalk和by之间有连线，by和clock之间有连线；
  train_phrase.txt或者test_phrase.txt是ground-truth的caption,用于train时decoder的输入和test时evaluation；
  上述三个文件每行互相对应，即相同行代表同一个graph的信息
  
# 代码文件解释：
  train.py 直接运行
  model.py 包括了graph network（相当于encoder network），attention network和lstm(相当于decoder)network
  build_vocab.py是建立字典库，生成data/vocab.pkl，我已经给了vocab.pkl，你可以不运行bulid_vocab.py
  data_loader.py 是把生成的graph和对应的caption组成pair送给网络进行train和test
  process_Scenegraph.py 是处理生成graph，我已经给data/annotation,你不必运行
  
  
