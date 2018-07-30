# neural networks implement by numpy

implemented model:

- multiply layer perceptron
- alexnet
- resnet

### dependencies:

- python3(3.5 or above)
- numpy(1.15.0 or above)
- pillow(5.2.0 or above)
- (optional) matplotlib: visualize results(TODO)
- (optional) graphviz: visualize network architecture(TODO)

### setup env

```
python3 -m venv env
source env/bin/activate
pip3 install -r requirements.txt
```

### repo architecture

```
.
├── requirements.txt
├── demo.py                                 # inference
├── train.py                                # 训练
├── val.py                                  # 测试
├── README.md
├── data
│   ├── fashion-mnist                       # fashion-mnist数据集
│   │   ├── t10k-images-idx3-ubyte.gz
│   │   ├── t10k-labels-idx1-ubyte.gz
│   │   ├── train-images-idx3-ubyte.gz
│   │   └── train-labels-idx1-ubyte.gz
│   ├── mnist                               # mnist数据集
│   │   ├── t10k-images-idx3-ubyte.gz
│   │   ├── t10k-labels-idx1-ubyte.gz
│   │   ├── train-images-idx3-ubyte.gz
│   │   └── train-labels-idx1-ubyte.gz
│   ├── samples
│   │   ├── mnist-by-hand                   # 自己手写的0-9
│   │   └── mnist-test                      # 从mnist test里面摘取出来的图片，用于inference
│   └── trained_models                      # 训练好的模型
│       ├── mlp-final.json
│       └── alxnet-final.json
└── src
    ├── activation.py                       # 激活函数
    ├── args.py                             # 定义输入参数
    ├── configuration.py                    # 定义训练参数
    ├── data_loader.py                      # 加载数据接口
    ├── layer.py                            # 定义NN的layer
    ├── logger.py
    ├── loss.py                             # 定义loss函数
    ├── lr_scheduler.py                     # 定义学习率调整函数
    ├── optim.py                            # 定义优化方法
    ├── symbol                              # 模型文件列表
    │   ├── alexnet.py                      # alexnet
    │   ├── mlp.py                          # multi-layer perceptron
    ├── timer.py                            # 时间测量封装接口
    └── utils.py
```

### 模型结构

#### multi-layer perceptron:

|layer|params|outputs|inputs|
|-----|------|------|-------|
|input layer|(768, 512)|(N, 512)|(N, 768)|
|relu| |(N, 512)|(N, 512)|
|hidden layer|(512, 1024)|(N, 1024)|(N, 512)|
|relu| |(N, 1024)|(N, 1024)|
|output layer|(1024, 10)|(N, 10)|(N, 1024)


#### alexnet:

|layer|kernel|pad|stride|outputs|inputs|
|-----|------|---|------|-------|------|
|layer1(conv)|(32, 3, 4, 4)|(0, 0)|2|(N, 32, 13, 13)|(N, 1, 28, 28)|
|relu||||(N, 32, 13, 13)|(N, 32, 13, 13)|
|layer2(conv)|(64, 32, 3, 3)|(1, 1)|2|(N, 64, 7, 7)|(N, 32, 13, 13)|
|relu||||(N, 64, 7, 7)|(N, 64, 7, 7)|
|layer3(conv)|(128, 64, 3, 3)|(1, 1)|2|(N, 128, 4, 4)|(N, 64, 7, 7)|
|relu||||(N, 128, 4, 4)|(N, 128, 4, 4)|
|fc1|(128x4x4, 1024)|||(N, 1024)|(N, 128x4x4)|
|fc2|(1024, 10)|||(N, 10)|(N, 1024)|

N is the batch size.

### results

|model|train_error|test_error|forward_time(s)|
|-----|-----------|----------|----------------|
|mlp|0.0228|0.0301|0.0036|
|alexnet||0.0201|0.2198|

alenxt 在fashion-mnist的准确率为0.76250

### usage

*training*:

mlp: `python3 train.py`

alexnet: `python3 train.py --config_id alexnet`

*testing*:

mlp: `python3 val.py`

alexnet:
`python3 val.py --config_id alexnet --ckpt_path data/trained_models/alexnet-final.json`

*inference*:

mlp: `python3 demo.py`

alexnet: `python3 demo.py --symbol_name alexnet --ckpt_path data/trained_models/alexnet-final.json`
### TODOs

- [ ] data augmentation

### references

- LeCun, Yann, Corinna Cortes, and C. J. Burges. "MNIST handwritten digit database."
- Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton. "Imagenet classification with deep convolutional neural networks." Advances in neural information processing systems. 2012.
- Glorot X, Bengio Y. Understanding the difficulty of training deep feedforward neural networks[J]. Journal of Machine Learning Research, 2010, 9:249-256.

