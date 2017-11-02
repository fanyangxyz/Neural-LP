# Neural LP

This is the implementation of Neural Logic Programming, proposed in the following paper:

[Differentiable Learning of Logical Rules for Knowledge Base Reasoning](https://arxiv.org/abs/1702.08367).
Fan Yang, Zhilin Yang, William W. Cohen.
NIPS 2017.

## Requirements
- Python 2.7
- Numpy 
- Tensorflow 1.0.1

## Quick start
The following command starts training a dataset about family relations, and stores the experiment results in the folder `exps/demo/`.

```
python src/main.py --datadir=datasets/family --exps_dir=exps/ --exp_name=demo
```

Wait for around 8 minutes, navigate to `exps/demo/`, there is `rules.txt` that contains learned logical rules. 