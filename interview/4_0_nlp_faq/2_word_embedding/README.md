1. word2vec负采样过程
    - 词汇表的大小为V,将一段长度为1的线段分成V份，每份对应词汇表中的一个词。每个词对应的线段长度是不一样的，高频词对应的线段长，低频词对应的线段短。每个词w的线段长度由下式决定：`len(w)=count(w)/∑u∈vocabcount(u)`；在word2vec中，分子和分母都取了3/4次幂如下：`len(w)=count(w)3/4∑u∈vocabcount(u)3/4`。在采样前，将这段长度为1的线段划分成M等份，这里M>>V，以保证每个词对应的线段都会划分成对应的小块。而M份中的每一份都会落在某一个词对应的线段上。在采样的时候，只需要从M个位置中采样出neg个位置就行，此时采样到的每一个位置对应到的线段所属的词就是我们的负例词。在word2vec中，M取值默认为10^8。
    
    
    
### Filtered References
1. [word2vec原理(三) 基于Negative Sampling的模型](https://www.cnblogs.com/pinard/p/7249903.html)