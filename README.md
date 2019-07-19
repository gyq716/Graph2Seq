# Graph2Seq  
GGNN-LSTM,using scene graph to generate captions  

## 本代码是历时多半年的毕设研究。包含了图神经网络到序列学习的任务，目的是输入一个graph，输出对应的描述caption。  
___
## 本项目基于encoder-decoder,`graph neural network`相当于encoder, lstm作decoder  
   模块包含 `图神经网络`、`LSTM`、`attention` 以及 整个`数据预处理`、`数据生成`、`evaluation`的各个细节。  
___
放出来的目的是为了让对*graph neural network（图神经网络）*和*encoder-decoder based attention*感兴趣的人更快的了解这方面的知识。  

## 打印版论文.pdf 是我的本科毕设论文，其详细介绍了每个模块的实现细节和图解表示，相信你如果能认真阅读的话，以上每个实现必将了如指掌。  
框架草图演示了更多的细节。
![image](https://github.com/nwpuhq/Graph2Seq/blob/master/%E6%A1%86%E6%9E%B6%E8%8D%89%E5%9B%BE.png)


## 运行环境   
  python3.5+pytorch0.4.1    

# 数据文件解释  data/annotation：     

  1. train_object.txt或者test_object.txt 是graph中的node,代表图中的节点，其中每一行是一个graph的object，relationship和attribute都当作graph的node，比如clock  by  sidewalk就代表代表当前graph共有3个节点；    
      
  2. train_rela.txt或者test_rela.txt 是graph的关系relationship，每一行是一个graph中的所有的relationship，比如sidewalk,by  by,clock代表sidewalk和by之间有连线，by和clock之间有连线；    
      
  3. train_phrase.txt或者test_phrase.txt是ground-truth的caption,用于train时decoder的输入和test时evaluation；
      
  上述三个文件每行互相对应，即相同行代表同一个graph的信息。   
  
# 代码文件解释：
    
  1. train.py 运行直接进行训练；    
  2. sample_batch.py 是运用测试;  
  3. model.py 包括了graph network（相当于encoder network），attention network和lstm(相当于decoder)network；    
  4. build_vocab.py是建立字典库，生成data/vocab.pkl，我已经给了vocab.pkl，你可以不运行bulid_vocab.py；    
  5. data_loader.py 是把生成的graph和对应的caption组成pair送给网络进行train和test；   
  6. process_Scenegraph.py 是处理生成graph，我已经给data/annotation,你不必运行；    
  7. word_split 也是对数据的预处理；
  8. 其他文件都是我在实现过程中做的一些to do list，还有实现过程中的一些增删改还有debug记录。    
  
  **data/annotation中所给的文件已经是完成了上述与数据预处理相关的任何操作，如果你只是想了解 graph neural network、  attention机制结合lstm，那么无需关心数据的预处理工作~~ **
     
     
  **如果你正在做graph的相关研究，本人的github仓库上还有graph convolutional network，graph attention network和gated graph nerual network的单独仓库，如果你有任何问题或者对图神经网络感兴趣但无从下手，欢迎在issues中提问！我看到必立刻回答解释!**
