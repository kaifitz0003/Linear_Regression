"""
Simple Linear Regression

The code below trains a model to predict y when given X by giving it training data and its results.
We only have a single column X so it is Simple Linear Regression
We create the training data and the testing data. We feed the training data to the y = mx + b model, and try to find the best m and b through this equation.
We give the model its error so it can learn and become more accurate. For example, when the model receives 1 for an input, it should return 2 because they correspond in the training data.
Since the data is simple, we picked a simpler model. If the data was more complex, we would have picked a different model.
Neural Networks work very well with many types of data which is why the're so common.
Convolutional Neural Networks work well with images.
Transformers (type of neural network) work very well with text data.
Graph Neural Networks work very well with social media data and medicine data.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt



# NumPy is broader than Pytorch which is why I had NumPy arrays instead of Pytorch tensors.
dataset = 'Salary'
if dataset == 'Salary':
  salary = pd.read_csv('https://github.com/ybifoundation/Dataset/raw/main/Salary%20Data.csv').astype('float32') # Salary is a Pandas Dataframe.
  np_salary = salary.to_numpy() # Change it to a numpy array because its not easy to go directly from Pandas to Pytorch
  X = np_salary[:,0] # Takes the 1st column of np_salary which contains the number of years working
  y = np_salary[:,1] # Takes the 2nd column of np_salary which contains the salary
  # Change below Numpy Arrays to Torch Tensors.
  # Done with Sklearn: https://www.kaggle.com/code/ybifoundation/simple-linear-regression
elif dataset == 'Manual':
  X = np.array([[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]], dtype = torch.float32)
  y = np.array([[2],[4],[6],[8],[10],[12],[14],[16],[18],[20]], dtype = torch.float32)

# Change 1d to 2d with 1 column
X = X.reshape(-1,1)

# Split the Data
X_train, X_test, y_train,  y_test = train_test_split(X,y)




# Convert NumPy array to Pytorch tensor
X_train = torch.from_numpy(X_train)
y_train = torch.from_numpy(y_train)
X_test = torch.from_numpy(X_test)



# Model
model = nn.Linear(1,1) # Puts 1 input through the model (X) y = mx+b and gives 1 output (y).

# Training/Parameter Estimation
# The model is given the # of years worked and the salary, and tries to figure out m & b.
lr = 1e-2
n_epochs = 500
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(params=model.parameters(),lr = lr)
for epoch in range (n_epochs):
  for i in range (len(X_train)):
    pred = model(X_train[i])
    loss = loss_fn(pred, y_train[i])
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
m = model.weight.data # m is the weight (slope)
b = model.bias.data # b is the bias (y intercept)

# Testing/Prediction
y_pred = model(X_test)

# Plotting
print(y_pred)
print(y_test)
#plt.scatter(y_pred.detach().numpy(), y_test)
#plt.scatter(X_train, y_train)
#X_plot = [X.min(),  X.max()
#y_plot = [X_train.min()*m+b,X_train.max()*m+b]

X_plot =  torch.tensor([0,11])
y_plot = X_plot*m[0,0]+b[0]

plt.plot(X_plot, y_plot, label = 'Prediction Data (Line best fit)')
plt.scatter(X_train,y_train, label = 'Training Data', s = 10)

plt.scatter(X_test,y_pred.detach().numpy(), marker='x', label = 'Real $ (Testing Data)', c = 'red', s = 80)
plt.scatter(X_test, y_test, label = 'Predicted $ (Testing Data)', marker='x')
plt.xlabel('Years Working')
plt.ylabel('Salary $')
plt.legend()
