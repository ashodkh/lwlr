# Local Weighted Linear Regression (lwlr)
This is a package for local weighted linear regression. 

The way local weighted lienar regression predicts a value $y_{\text{pred},i}$ for an input $x_{\text{test},i}$ is by finding the $k$ nearest neighbors of $x_{\text{test},i}$ in the training set $x_{\text{train}}$ and using these neighbors (along with their labels $y_{\text{train}}$) to fit a straight line locally. 

$y_{\text{pred},i} = \boldsymbol{\theta_i}^Tx_{\text{test},i},$

where $\boldsymbol{\theta}_i$ is obtained by minimizing the local cost function (given the set of $k$ nearest neighbors in the training set as NN)

$C_i = \sum_{j\in{NN}}w_{\text{train},j}(y_{\text{train},j}-\boldsymbol{\theta}^T_ix_{\text{train},j})^2.$
  
The weights $w_{\text{train},j}$ implemented in this package are 'constant', 'inverse_distance', and 'inverse_distance_squared'. There is also the option to add custom weights. This minimization problem has an analytical solution $\boldsymbol{\theta}=(\textbf{X}^T\textbf{WX})^{-1}(\textbf{X}^T\textbf{WY})$, with $\textbf{X}$ being the matrix of features, $\textbf{W}$ a diagonal matrix of the weights, and $\textbf{Y}$ the matrix of outcomes.

# Installation
```
pip install lwlr
```

**Dependencies**
- Python >= 3.6
- sklearn
- numpy

# Usage
The model can be used with a few simple lines of code:
```
from lwlr import LWLR

model = LWLR(weight_type='inverse_distance')
y_pred = model.predict(x_test, x_train, y_train, nn=5)
```
A detailed example is found in the usage_example notebook. 


