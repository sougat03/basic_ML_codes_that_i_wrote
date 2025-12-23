#Linear Regression

import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('Housing.csv') #area vs price, got the dataset from (https://www.kaggle.com/datasets/neurocipher/house-price-for-linear-regression)

   #we defined loss funcrtion, the input parameters m,b are from 
   #the straight line equation (y = mx + b). tells us how much we are off from actual result

def loss_function(m , b, points):
    
    total_loss = 0 #starts at zero
    for i in range(len(points)): #now we add all indivisual squared errors and divide with points
         x = points.iloc[i].area
         y = points.iloc[i].price
         total_loss += (y - (m * x + b)) ** 2
    total_loss / float(len(points))

    return total_loss / float(len(points))


#but we are not going to use the loss function in our actual actual optimisation.
#will use the expression that i derived in the notes that i am going to attach hopefully
#we are trying to minimize the loss function in terms of m and b

def gradient_descent(m_now, b_now, points, L): #L is learning rate
     m_gradient = 0
     b_gradient = 0
     n = len(points)

     for i in range(n):
          x = points.iloc[i].area
          y = points.iloc[i].price
          
          m_gradient += -(2/n) * x * (y - (m_now * x + b_now)) #_now means this value this moment in time
          b_gradient += -(2/n) * (y - (m_now * x + b_now)) 
          
     #now we have to consider which direction to move to
     #so new m and b are going to be
     
     m = m_now - m_gradient * L #L determines how much we move
     b = b_now - b_gradient * L
     
     return m,b


m = 0
b = 0
L = 1e-8
epochs = 1000 #number of iterations

for i in range(epochs):
     if i % 200 == 0:
          print(f"Epoch : {i}")
     m, b = gradient_descent(m, b, data, L)

print(m, b)

plt.scatter(data.area, data.price, color = "black")
plt.plot(list(range(2000, 12000)), [m * x + b for x in range(2000, 12000)], color = "red")
plt.show()