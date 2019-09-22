import numpy as np 

# iwatt = np.load('iwatt.npy')
# uwatt = np.load('uwatt.npy')

# print(iwatt.shape)
# a = iwatt[1,0,:10]
# b = uwatt[1,0,:10]
# print(a)
# print(b)

# uside = uwatt[:,0:1,:]
# a = np.sum(uside*iwatt,axis=-1)


# index = np.random.randint(len(iwatt))
# print(uwatt[index,0]-iwatt[index,0])
# a = uwatt[index]
# b = iwatt[index]

# a = a[0:1]
# print(a.shape, b.shape)

# dist = np.sum((a-b)**2,axis=-1)**0.5
# dist = -dist
# softmax = np.exp(dist)/np.sum(np.exp(dist))
# print(softmax)
# # print(a[:100])

# ucnn = np.load('ucnn.npy')
# icnn = np.load('icnn.npy')

# print(ucnn.shape, icnn.shape)
# usample = ucnn[0,-1,:]
# isample = icnn[0,-1,:]

# print(usample)
# print(isample)
vec1 = np.load('docuser.npy')
vec2 = np.load('docitem.npy')

print(vec1.shape, vec2.shape)
# print(vec1[0,0])
# print(vec2[0,0])

print(vec1[0,1,0] - vec1[2,0,0])
