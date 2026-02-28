import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv("Salary_dataset.csv")

def loss_function(m, b, points):    
    total_error = 0
    for i in range(len(points)):
        x = points.iloc[i].Experience
        y = points.iloc[i].Salary
        total_error += (y - (m*x + b)) ** 2
    total_error / float(len(points))

def gradient_descent(m_now, b_now, points, learning_rate):
    m_gradient = 0
    b_gradient = 0

    n = len(points)

    for i in range(len(points)):
        x = points.iloc[i].Experience
        y = points.iloc[i].Salary

        m_gradient += -(2/n) * x * (y - (m_now * x + b_now))
        b_gradient += -(2/n) * (y - (m_now * x + b_now))

    m = m_now - m_gradient * learning_rate
    b = b_now - b_gradient * learning_rate

    return m, b

def normalize(data):
    data = np.array(data)
    min_val = np.min(data)
    max_val = np.max(data)
    
    # Avoid division by zero
    if max_val - min_val == 0:
        return np.zeros(data.shape)  # or return data
    
    normalized_data = (data - min_val) / (max_val - min_val)
    return normalized_data


# Normalize Experience
data['Experience'] = normalize(data['Experience'])
data['Salary'] = normalize(data['Salary'])

print(data)

m = 0
b = 0
Learn = 0.01
epochs = 2000

for i in range(epochs):
    m, b = gradient_descent(m, b, data, Learn)

print(m,b)

predicted_salary = m * data.Experience + b

plt.scatter(data.Experience, data.Salary, color="black", label="Data Points")
plt.plot(data.Experience, predicted_salary, color="red", label="Regression Line")
plt.xlabel("Normalized Experience")
plt.ylabel("Salary")
plt.title("Salary vs Experience")
plt.legend()
plt.show()