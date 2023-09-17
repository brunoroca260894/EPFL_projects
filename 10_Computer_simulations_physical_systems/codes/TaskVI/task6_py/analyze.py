
"""Blocking analysis
Print running average
Blocking analysis scheme
Running error

Igor Reshetnyak 2017
"""

import math,sys,pickle,os.path,pylab,time

if len(sys.argv)<2 :
    print ("No file to analyze")
    exit()

datafile=sys.argv[1]
file_name=datafile+''
if os.path.isfile(file_name)==False:
 print ('file does not exist')
 exit()

input=open(file_name,'r')
#samples=pickle.load(input)

samples=[]

for line in input:
 data=line.split()
 samples.append(float(data[1]))

input.close()

N=len(samples)

print (N)

#The first algorithm
def AvandError(sample,N):
 Av=sum(sample)/float(N)
 Error=math.sqrt(sum([(x-Av)**2 for x in sample]))/float(N)
 return Av,Error

Av,Error=AvandError(samples,N)

#The bunching algorithm

def makebunch(sample):
 new_list=[]
 while sample != []:
  x=sample.pop(0)
  y=sample.pop(0)
  new_list.append((x+y)/2.)
 return new_list

sample1=samples[:]
Avs2=[]

Errors2=[]
step=0

sample1=makebunch(sample1)

while len(sample1)>4:
 print (step)
 step+=1
 N2=len(sample1)
 Av2,Error2=AvandError(sample1,N2)
 Avs2.append(Av2)
 Errors2.append(Error2)
 sample1=makebunch(sample1)

pylab.plot(range(1,step+1),Errors2,'ro')
pylab.axhline(y=Error,color='b')
pylab.axis([1,step,0,2*max(Errors2)])
pylab.xlabel('Bunching step')
pylab.ylabel('Error')
pylab.savefig(datafile+'Error2.png')
pylab.clf()

#Real time evaluation
Avs3=[samples[0]]
Errors3=[samples[0]**2]

for i in range(1,N):
 Avs3.append(samples[i]+Avs3[i-1])
 Errors3.append(samples[i]**2+Errors3[i-1])

Avs3=[Avs3[i]/float(i+1) for i in range(N)]
Errors3=[math.sqrt((Errors3[i]/float(i+1)-Avs3[i]**2)/float(i+1)) for i in range(N)]



pylab.plot(range(N),Errors3,'r')
pylab.axhline(y=Error,color='b')
pylab.axis([1,N-1,0,2*max(Errors3)])
pylab.xlabel('Step')
pylab.ylabel('Error')
pylab.savefig(datafile+'Error3.png')
pylab.clf()

pylab.plot(range(N),Avs3,'r')
pylab.axhline(y=Av,color='b')
#pylab.axis([1,N-1,1.1*min(Avs3),1.1*max(Avs3)])
pylab.xlabel('Step')
pylab.ylabel('Average')
pylab.savefig(datafile+'Average3.png')
pylab.clf()
