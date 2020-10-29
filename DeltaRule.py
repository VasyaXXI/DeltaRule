import numpy as np

def sigmoid(self):
    return(1/(1+np.exp(-self)))

x_inputs = np.array([[1,0],
                     [1,1],
                     [1,0],
                     [0,1]])

print("Входные значения:", '\n', x_inputs)
x_outputs = np.array([[1,0,1,0]]).T
np.random.seed(1)
weight = np.around(2 * np.random.random((2,1)) - 1, decimals=2)

print("Случайные веса:", '\n', weight)
for i in range(1000): 
    input_layer = x_inputs
    outputs = np.around(sigmoid(np.dot(input_layer, weight)), decimals=1)
    # Обучение по дельта правилу
    err = x_outputs - outputs
    new_weight = np.around(np.dot(input_layer.T, err * (outputs * (1 - outputs))), decimals=2)
    weight += new_weight

print("Веса после обучения:", '\n', weight)
print("Результат:", '\n', outputs)
