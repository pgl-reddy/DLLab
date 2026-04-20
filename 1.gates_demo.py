weight1, weight2, bias = 1, 1, 1

#inputs for NOR gates
inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
out =  [1,0,0,0]
learning_rate = 0.1
epochs = 15

def activate(x):
    return 1 if x>=0 else 0

def train_perceptron(inputs, out, learning_rate, epochs):
    global weight1, weight2, bias
    for epoch in range(epochs):
        total_error = 0
        for i in range(len(inputs)):
            A, B = inputs[i]
            target_output = out[i]
            output = activate(weight1 * A + weight2 * B + bias)
            error = target_output - output
            weight1 += learning_rate * error * A
            weight2 += learning_rate * error * B
            bias += learning_rate * error
            total_error += abs(error)
        if total_error == 0:
            print(epoch)
            break

train_perceptron(inputs, out, learning_rate, epochs)

for i in range(len(inputs)):
    A, B = inputs[i]
    output = activate(weight1 * A + weight2 * B + bias)
    print(f"Input: ({A}, {B})  Output: {output}")
