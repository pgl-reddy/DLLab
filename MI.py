weights = [0.2, 0.1, 0]
 
win  = [0.65, 0.8, 0.8, 0.9]
toss = [8.5, 9.5, 9.9, 9]
fan  = [1.2, 1.3, 0.5, 1]
 
for i in range(len(win)):
    prediction = (
        win[i]  * weights[0] +
        toss[i] * weights[1] +
        fan[i]  * weights[2]
    )
    print(f"Match {i+1} predicted value:", prediction)
