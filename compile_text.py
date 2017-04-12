import os

govtxt=""
path='../from_github/Topic_Models-master/data/'
for fname in os.listdir(path+"govdates/"):
	f=open(path+"govdates/"+fname)
	text=f.read().replace('"','')
	govtxt=govtxt+text+"/n"

opptxt=""
path='../from_github/Topic_Models-master/data/'
for fname in os.listdir(path+"oppdates/"):
	f=open(path+"oppdates/"+fname)
	text=f.read().replace('"','')
	opptxt=opptxt+text+"/n"


f=open("govtxt.txt",'w')
f.write(govtxt)
f.close()

f=open("opptxt.txt",'w')
f.write(opptxt)
f.close()