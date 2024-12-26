import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('Advertising.csv')

radio_ad_spent = data['radio']
sales = data['sales']


def train(spendings, sales, w, b, alpha, epochs):
    lines = []
    for e in range (epochs):
        w,b=update_w_and_b(spendings, sales, w, b, alpha)

        if e % 3000 == 0:
            lines.append((w, b, e))
        
        if e % 400 == 0:
            print("epoch:", e, "loss: ", avg_loss(spendings, sales, w, b))
    return w, b, lines

def update_w_and_b(spendings, sales, w, b, alpha):
    dl_dw=0.0
    dl_db=0.0
    N = len(spendings)

    for i in range(N):
        dl_dw += -2 * spendings[i] * (sales[i] - (w * spendings[i] + b))
        dl_db += -2*(sales[i] - (w*spendings[i] + b))

    w = w - (1/float(N)) *dl_dw*alpha
    b = b - (1/float(N)) *dl_db*alpha

    return w, b

def avg_loss(spendings, sales, w, b):
    N = len(spendings)
    total_error = 0.0
    for i in range(N):
        total_error += (sales[i] - (w*spendings[i] + b))** 2
    return total_error / float(N)

def predict(x,w,b):
    return w*x+b

w,b, lines = train(radio_ad_spent, sales, 0.0, 0.0, 0.001, 15000)
print("Final weight:", w)
print("Final bias:", b)
x_new=23.0
y_new=predict(x_new, w, b)

print("Predicted sales for x_new =", x_new, "is:", y_new)

plt.figure(figsize=(8, 6))
plt.scatter(radio_ad_spent, sales, color='blue', edgecolors='yellow', label='Data Points')
plt.scatter(x_new, y_new, color='green', label=f'Predicted (x={x_new})', zorder=5)

for (w_line, b_line, epoch) in lines:
    plt.plot(radio_ad_spent, w_line * radio_ad_spent + b_line, label=f'Epoch {epoch}', linestyle='--')

plt.title('Radio Advertising vs Sales')
plt.xlabel('Money Spent on Radio Advertising')
plt.ylabel('Sales')
plt.legend()
plt.show()
