# Supervised-Treebank-Conversion
for paper, Dependency Treebank Conversion and Exploitation Based on Full-Tree LSTM

## Requirements
```txt
python 3.6.5
pytorch 0.4.1
cython  0.28.2
```

## Run
* cd exp
* config.txt: 设置各种配置
  -  设置数据文件，pretrained embedding (giga-100)地址
  -  指定loss类型（crfloss, cross entropy loss)
* run.sh：
   - 设置train or test
   - 将exe指向src中的main.py
   - 没有字典先执行创建字典的那条命令-》创建字典并保存初始化参数
   （run.sh: $exe --is_dictionary_exist 0 --is_train 1 --is_test 0 > log.create-dict 2>&1）
   - 再有了字典和初始化模型的情况下，可以进行模型训练了
   (run.sh: $exe --is_dictionary_exist 1 --random_seed 1540422239 --is_train 1 --is_test 0 > log.train 2>&1)
