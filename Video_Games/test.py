import sys 
import numpy as np 
cols,rows = 14680,8713

mat = np.random.randint(0,5,(rows, cols))
for i in range(cols):
	sys.stdout.write('\r{}'.format(i))
	vec = mat[i]

	index = np.random.randint(0,rows,(200))

	vec = vec[index]
	sub_mat = mat[i:].T[index].T

	res = vec*sub_mat
