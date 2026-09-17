import math

PI = 3.14159

def power_grid_forecast(consumption_data):
	# 1) Subtract the daily fluctuation (10 * sin(2π * i / 10)) from each data point.
	# 2) Perform linear regression on the detrended data.
	# 3) Predict day 15's base consumption.
	# 4) Add the day 15 fluctuation back.
	# 5) Round, then add a 5% safety margin (rounded up).
	# 6) Return the final integer.
	n = len(consumption_data)
	x = list(range(1, n + 1))
	y = [consumption_data[i - 1] - 10 * math.sin(2 * PI * i/ 10) for i in x]
	sum_x = sum(x)
	sum_y = sum(y)
	sum_xy = sum(xi * yi for xi, yi in zip(x, y))
	sum_xx = sum(xi ** 2 for xi in x)
	den = n * sum_xx - sum_x ** 2
	if den == 0:
		m = 0
		c = sum_y / n 
	else:
		m = (n * sum_xy - sum_x * sum_y) / den
		c = (sum_y - m * sum_x) / n
	day_15 = math.ceil(round(m * 15 + c + 10 * math.sin(2 * PI * 15 / 10)) * 1.05)
	return day_15