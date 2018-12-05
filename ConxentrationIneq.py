import numpy
from numpy import *
import matplotlib.pyplot as plt
data = numpy.random.binomial(1,0.25,(100000,1000))
#print(data)

NUM_TOSSES = 1000

"""this function calculate the pecentage of sequences that satisfy
  |Xm-E[X]|>=epsilon as a function of m, given that p = 0.25 and epsilon -
  the aprroximation error"""
def percentage(epsilon,dataM):

    per_arr = [float(0) for i in range(1000)]

    for k in range(100000):
        for j in range(1,1001):
               XmHat = (1/j)*dataM[k][j-1]
               if(XmHat-0.25>=epsilon):
                   per_arr[j-1] = per_arr[j-1] + (1/100000)

    return per_arr

def print_title(epsilon):
    if (epsilon == 0.5):
        plt.title("epsilon = 0.5")
    if (epsilon == 0.25):
        plt.title("epsilon = 0.25")
    if (epsilon == 0.1):
        plt.title("epsilon = 0.1")
    if (epsilon == 0.01):
        plt.title("epsilon = 0.01")
    if (epsilon == 0.001):
        plt.title("epsilon = 0.001")


#calculate the estimate Xm of the 5 first rows
for k in range(5):
    sum = 0
    estimation_arr = list()
    for i in range(NUM_TOSSES):
        sum += data[k][i]
        estimation_arr.append(sum/(i+1))

    estimation = (1 / NUM_TOSSES) * sum
    plt.plot([i for i in range(NUM_TOSSES)],estimation_arr,label= "estimate bias")
    plt.xlabel("m : number of tosses")
    plt.ylabel("estimation of coin bias")
plt.legend()
plt.title("estimation of coin bias as function of m")
plt.show()


"""this function caculate and plot for each approximation error (epsilon) of the
 trainig data, the upper bound of P(|Xm-E[X]|>=epsilon) usinig Chevicev and Hoffding
 conc. inequalities. in addition plotting the pecentage of sequences that satisfiy
  |Xm-E[X]|>=epsilon as a function of m"""
def concentratioIneq(epsilon,dataM):
    samples = list()
    CheviArr = list()

    HoffdingArr = list()
    for m in range(1,1001):
        CheviBound = 1/(4*m*epsilon**2)
        CheviArr.append(CheviBound)
        HoffdingBound = exp(-2*m*epsilon**2)
        HoffdingArr.append(HoffdingBound)
        samples.append(m)

    #calculate percentage given p=0.25

    percentageArr = percentage(epsilon,dataM)
    plt.plot(samples, CheviArr, label="CheviBound")
    plt.plot(samples, HoffdingArr, label="HoffdingBound")

    plt.xlabel("m- number of tosses")
    plt.ylabel("estimation of coin bias/Percentage")
    #print_title(epsilon)
    #plt.legend()
    #plt.show()

    plt.plot(samples, percentageArr, label="percentage")
    #plt.xlabel("m- number of tosses")
    #plt.ylabel("pecentage ")
    print_title(epsilon)
    plt.legend()
    plt.show()

dataM = cumsum(data, axis=1)
for epsilon in [0.5,0.25,0.1,0.01,0.001]:
    concentratioIneq(epsilon,dataM)
