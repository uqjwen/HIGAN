import pickle
import numpy as np 

def main():
	prefix = '/home/wenjh/aHIGAN/Office_Products/'
	filename = prefix+'Office_Products_5.json'
	data = pickle.load(open(filename.split('.')[0]+'.pkl','rb'))
	vec_u_text  = data['vec_u_text']
	vec_i_text  = data['vec_i_text']
	vec_uit = np.array(data['vec_uit'])
	pmtt_file = filename.split('.')[0]+'_pmtt.npy'
	
	pmtt = np.load(pmtt_file).astype(int)


	vec_uit = vec_uit[pmtt]
	vec_u_text = vec_u_text[pmtt]
	vec_i_text = vec_i_text[pmtt]

	uid = 4492
	iid = 1027
	uindex = np.where(vec_uit[:,0] == uid)[0]

	iindex = np.where(vec_uit[uindex, 1] == iid)[0]

	print(vec_u_text[uindex[iindex]])
	print(vec_i_text[uindex[iindex]])


if __name__ == '__main__':
	main()