import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Data

X = np.array([[1,2],[2,7],[3,6],[4,1]], dtype = np.float32) # Multiple Linear Regression
y = np.array([8,25,24,11], dtype = np.float32)
X = torch.from_numpy(X) # Turns the numpy arrays into Torch tensors
y = torch.from_numpy(y)
model = nn.Linear(2,1)

X_train, X_test, y_train, y_test = train_test_split(X,y)

# Training
lossfn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
for epoch in range (5000):
  #print(epoch)
  for i in range (len(X_train)):
    out = model(X_train[i]) # Pick the i-th row. Also the same as X_train[i,:]
    loss = lossfn(out, y_train[i])
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

# Testing

y_pred = model(X_test)

lossfn(y_test,y_pred)
print(model.weight.data.numpy()) # Each X column will have a weight
print(model.bias.data.numpy()) # In the equation w0 + w1*X1 + w2*X2... w0 is the bias.

# Plotting
w0 = model.bias.data.numpy()[0]
X1 = X[:,0]
X2 = X[:,1]
w1 = model.weight.data.numpy()[0,0]
w2 = model.weight.data.numpy()[0,1]

X1,X2 = np.meshgrid(X1,X2)
out = w0 + w1*X1+w2*X2
out
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot_surface(X1,X2,out)
ax.scatter(X[:,0],X[:,1],y, c = 'red')
