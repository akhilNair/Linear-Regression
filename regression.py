# For linear regression we use equation of line y =mX + b

# Choose any random value of m and b for each row of data and using X value into equation (y = mX +b ), 
# which will give us a new y. This new y aka y_predicted will be compared to actual y from same row of data,
#  and then we will take the square of the difference of errors,  this process will be repeated for each row 
# of data and then we will calculate total error by summing up error of each row and normalized (average) it by 
# dividing it by count of all data rows. Compute using error function f(x) = ((y_initial – y_predicted)^2) / Number of data rows


# Minimizing errors using partial derivatives/ gradient descent



import csv
from numpy import *


def compute_error(intercept,slope,points):
	
	total_error = 0
	for i in range(0,len(points)):
		x = points[i,2]
		y = points[i,10]
		total_error += ((y - (slope * x + intercept))**2)
		
	return total_error / float(len(points))



# The Step Gradient Descent function => use the error/ cost function and then minimize it using partial deriviatives	
def step_gradient(b_current,m_current,points,learning_rate,iteration):
	#gradient descent
	N = float(len(points))
	b_gradient = 0
	m_gradient= 0
	for i in range(len(points)):
		
		x = points[i,2]
		y = points[i,10]
		b_gradient += -(2/N)*(y -((m_current * x) + b_current))
		m_gradient += -(2/N)*x*(y -((m_current * x) + b_current))
	new_b = b_current - (learning_rate * b_gradient)
	new_m = m_current - (learning_rate * m_gradient)
	return [new_b,new_m]
			



def gradient_descent_runner(points,b,m,learning_rate,num_iterations):
	b1 = b
	m1 = m
	
	for i in range(0,num_iterations):
		b1,m1 = step_gradient(b1,m1,array(points),learning_rate,i)
		
	return [b1,m1]
	
	
def start():
	#read data from csv
	data = genfromtxt('.//datasets//diabetes1.csv',skip_header=1,delimiter = ',')
	'''with open('.//datasets//diabetes1.csv') as d:
		next(d,None)
		a = csv.reader(d,delimiter = ',',quotechar ='"')
		data = list(a)'''

	#hyperparameter to be used as tuning knobs. decided how fast model trains. too low --> too slow to converge.  to high --> will never converge
	learning_rate =  0.001
	
	#initial y-intercept
	initial_b = 1
	
	#initial slope
	initial_m = 1
	
	#choose number of iterations to run linear regression
	num_iterations = 1000
	
	print("Starting gradient descent at b = {0}, m = {1}, error = {2}".format(initial_b, initial_m, compute_error(initial_b,initial_m,data)))
	print("Running...")	
	
	[b,m] = gradient_descent_runner(data,initial_b,initial_m,learning_rate,num_iterations)
	print("After {0} iterations b = {1}, m = {2}, error = {3}".format(num_iterations, b, m, compute_error(b,m,data)))
	
	x_test= input("\n Enter BMI to get Blood Sugar\n")
	print("Test/Sample BMI is: {0}".format(x_test))
	y_test =m * float(x_test) + b
	print("Blood Sugar is {0} ".format(y_test))





if __name__ == '__main__':
	start()