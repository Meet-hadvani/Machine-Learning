from random import randint
import matplotlib.pyplot as plt
train_limit = 1000
train_count = 100

train_input = list()
train_output = list()

for i in range(train_count):
	area = randint(0, train_limit)
	size = randint(0, train_limit)
	prize = (1.5*area) + (10*size)
	plt.plot(prize, size, label="stars", color="green", marker = "*")
	train_input.append([area, size])
	train_output.append(prize)

from sklearn.linear_model import LinearRegression
predictor = LinearRegression(n_jobs = -1)
predictor.fit(X=train_input, y=train_output)

X_test = [[100,800]]
outcome = predictor.predict(X=X_test)
coefficient = predictor.coef_
print("outcome - {}".format(outcome))

plt.plot(outcome, X_test[0][1], label="stars", color="red", marker = "*")
plt.xlabel("Prize")
plt.ylabel("Size")
plt.title("Prize prediction")
plt.show()