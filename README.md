# neural networks implement by numpy

implemented model:

- multiply layer perception
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

### results

|model|train_error|test_error|forward_time(ms)|
|mlp|0.0227|0.0301|3.6|
|alexnet|||


### usage

train:

`python3 train.py --config_id alexnet`

validate:

`python3 val.py --config_id mlp --ckpt_path ckpt/mlp-v4/mlp-99.json`

demo:

`python3 demo.py --symbol_name mlp --ckpt_path ckpt/mlp-v4/mlp-99.json --test_dir data/samples/mnist-test`

### TODOs

- [ ] data augmentation
