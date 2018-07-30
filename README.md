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
├── README.md
├── data
│   ├── fashion-mnist
│   │   ├── t10k-images-idx3-ubyte.gz
│   │   ├── t10k-labels-idx1-ubyte.gz
│   │   ├── train-images-idx3-ubyte.gz
│   │   └── train-labels-idx1-ubyte.gz
│   ├── handwrite.png
│   ├── mnist
│   │   ├── t10k-images-idx3-ubyte.gz
│   │   ├── t10k-labels-idx1-ubyte.gz
│   │   ├── train-images-idx3-ubyte.gz
│   │   └── train-labels-idx1-ubyte.gz
│   ├── samples
│   │   ├── mnist-by-hand
│   │   └── mnist-test
│   └── trained_models
│       └── mlp-final.json
├── src
│   ├── activation.py
│   ├── args.py
│   ├── configuration.py
│   ├── data_loader.py
│   ├── layer.py
│   ├── logger.py
│   ├── loss.py
│   ├── lr_scheduler.py
│   ├── optim.py
│   ├── symbol
│   │   ├── alexnet.py
│   │   ├── mlp.py
│   ├── timer.py
│   └── utils.py
├── demo.py
├── requirements.txt
├── train.py
└── val.py
```

### model architecture

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
|relu||||(N, 64, 7, 7)|(N, 64, 7, 7)|
|layer3(conv)|(128, 64, 3, 3)|(1, 1)|2|(N, 128, 4, 4)|(N, 64, 7, 7)|
|relu||||(N, 128, 4, 4)|(N, 128, 4, 4)|
|fc|(128x4x4, 1024)|||(N, 1024)|(N, 128x4x4)|
|fc|(1024, 10)|||(N, 10)|(N, 1024)|

N is the batch size.

### results

|model|train_error|test_error|forward_time(ms)|
|-----|-----------|----------|----------------|
|mlp|0.0227|0.0301|3.6|
|alexnet|||


### usage

train:

`python3 train.py --config_id mlp`

validate:

`python3 val.py --config_id mlp `

demo:

`python3 demo.py --symbol_name mlp --ckpt_path ckpt/mlp-v4/mlp-99.json --test_dir data/samples/mnist-test`

`python3 demo.py --symbol_name mlp --ckpt_path ckpt/mlp-v4/mlp-99.json --test_dir data/samples/mnist-test`

### TODOs

- [ ] data augmentation

### inferences

