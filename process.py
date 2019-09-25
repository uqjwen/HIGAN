import numpy as np 

domain = ['Grocery_and_Gourmet_Food', 'Musical_Instruments', 'Office_Products', 'Video_Games']

def single_domain(filename):
	data = np.genfromtxt(filename)
	index = np.argsort(data[:,-1])[:5]

	sub_data = data[index]

	print(filename)
	print(np.mean(sub_data, axis=0))

def multi_domain(domain):
	for directory in domain:
		single_domain('./'+directory+'/loss.txt')



if __name__ == '__main__':
	multi_domain(domain)