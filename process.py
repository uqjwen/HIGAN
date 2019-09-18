import json 

def readfile(filename):
	f = open(filename, encoding='utf-8')
	data = []
	for line in f.readlines():
		line = json.loads(line)
		data.append(line)
	f.close()
	return data


if __name__ == '__main__':
	filename = './data/Musical_Instruments_5.json'
	data = readfile(filename)
	print(data[0])