pyLDA
=====

python实现的LDA算法

模型参数在lda.py文件的前几行修改，参数分别为：
'''
alpha = 0.1
beta = 0.1
K = 10  //主题个数
iter_num = 50  //迭代次数
top_words = 20  //每个主题显示的词的个数

wordmapfile  = './model/wordmap.txt'  //wordmap文件存储位置
trnfile = "./model/test.dat"          //训练文件
modelfile_suffix = "./model/final"    //模型文件的存储位置以及前缀
'''

输入文件的要求：
每行为一篇文档，分词后用空格隔开。

运行命令：

'''
python lda.py
'''
