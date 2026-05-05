# Learning reveals invisible structure in low-rank RNNs
This repository contains code for our work extending the low-rank RNN framework from activity to learning dynamics. 
We derive a closed-form, low-dimensional system of ODEs governing learning in overlap space, exact for linear RNNs and asymptotically exact for nonlinear RNNs in the large-N Gaussian limit using dynamical mean-field theory. 
We then use this framework to study two central phenomena in neural learning, degeneracy and memory. We further demonstrate implications and testable predictions for biological learning experiments.

![Framework](https://github.com/yoavger/learning_reveals_invisible_structure_lr_rnns/blob/main/figures/framework.png)

- To reproduce the figures from the paper, refer to `code/figures.ipynb`. The notebook provides options to either load existing data (if available) or train and save new data.
- A compact demonstration of the learning dynamics equations for both linear and nonlinear theory is provided in `code/linear.ipynb` and `code/nonlinear.ipynb`.
