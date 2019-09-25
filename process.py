import numpy as np 
import matplotlib.pyplot as plt 

domain = ['Grocery_and_Gourmet_Food', 'Musical_Instruments', 'Office_Products', 'Video_Games']
my_color = ['#107c10','#DC3C00','#7719AA','#0078D7','#DC6141','#4269A5','#39825A','#DC6141']



def single_domain(filename):
	data = np.genfromtxt(filename)
	index = np.argsort(data[:,-1])[:5]

	sub_data = data[index]

	print(filename)
	print(np.mean(sub_data, axis=0))

def multi_domain(domain):
	for directory in domain:
		single_domain('./'+directory+'/loss.txt')

def plot_lines(domain):
	for directory in domain:
		# filename = './'+directory+'/loss.txt'
		plot_line(directory)

def get_line_cross_files(domain):
	data = []
	for directory in domain:
		filename = './'+directory+'/loss.txt'
		sub_data = np.genfromtxt(filename)
		plt_data = sub_data[:,-1:]
		plot_lines(plt_data)

def plot_lines(data):
	print(data.shape)
	rows, cols = data.shape

	index = [i+1 for i in range(rows)]
	for i in range(cols):
		sub_data = data[:,i]
		plt.plot(index, sub_data, color = my_color[i], label = 'name '+str(i))
	plt.grid(linestyle = '-.')
	plt.legend()
	plt.show()




if __name__ == '__main__':
	# multi_domain(domain)
	get_line_cross_files(domain)