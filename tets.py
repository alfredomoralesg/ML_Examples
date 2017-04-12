import numpy as np
from multiprocessing import Process

def targ():
	print np.random.randint(8,24,10)

for i in range(100):
	Process(target=targ,args=()).start()
	#print np.random.randint(8,24,10)