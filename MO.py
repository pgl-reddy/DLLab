weights = [0.3, 0.2, 0.9]
 
def ele_mul(number, vector):
    output = [0, 0, 0]
    assert(len(output) == len(vector))
    for i in range(len(vector)):
        output[i] = number * vector[i]
    return output
 
def neural_network(input, weights):
    pred = ele_mul(input, weights)
    return pred
 
win = [0.65, 0.8, 0.8, 0.9]
 
for i in range(len(win)):
    input = win[i]
    pred = neural_network(input, weights)
    print(f"Match {i+1} predicted value:", pred)
