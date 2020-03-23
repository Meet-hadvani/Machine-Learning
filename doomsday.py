import matplotlib.pyplot as plt

train_input = list()
train_output = list()

year = [1947, 1949, 1953, 1960, 1963, 1968, 1969, 1972, 1974, 1980, 1981, 1984, 1988, 1990, 1991, 1995, 1998, 2002, 2007, 2010, 2012, 2015, 2017, 2018]
carbon = [310, 312, 315, 316, 318, 326, 330, 335, 338, 342, 343, 350, 353, 356, 360, 368, 370, 372, 380, 390, 391, 403, 406, 410]
tempreture = [-0.1, -0.15, -0.18, -0.9, 0.5, -0.9, 0.5, 0.01, -0.9, 0.26, 0.31, 0.1, 0.39, 0.45, 0.4, 0.44, 0.61, 0.62, 0.66, 0.71, 0.63, 0.9, 0.91, 0.85]
time = [7, 3, 2, 7, 12, 7, 10, 12, 9, 7, 4, 3, 6, 10, 17, 14, 9, 7, 5, 6, 5, 3, 2.5, 2]

for i in range(24):
	plt.plot(year[i], time[i], label="stars", color="green", marker = "*")	
	train_input.append([carbon[i], tempreture[i]])
	train_output.append(time[i])

from sklearn.linear_model import LinearRegression
predictor = LinearRegression(n_jobs = -1)
predictor.fit(train_input, train_output)

X_test = [[412, 0.9]]
outcome_2020 = predictor.predict(X=X_test)
plt.plot(2020, outcome_2020, label="stars", color="red", marker = "*")

# 2020, 412, 0.88
X_test = [[432, 1.1]]
outcome_2021 = predictor.predict(X=X_test)
plt.plot(2021, outcome_2021, label="stars", color="red", marker = "*")

print("outcome - {}".format(outcome_2020))
plt.xlabel("year")
plt.ylabel("time")
plt.title("Doomsday time prediction")
plt.show()
