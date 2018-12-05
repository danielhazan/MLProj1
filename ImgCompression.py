from numpy import diag,zeros
from numpy.linalg import svd,norm
from matplotlib.pyplot import *
import scipy.misc as misc

image = misc.ascent()
U,D,V = np.linalg.svd(image)
print(np.diag(D))
print(len(U))

def compress(k):
    U, D, V = np.linalg.svd(image)

    for i in range(511-k):
        D[511-i] = 0

    return D

def reconstruct(U,D,V):
    return np.dot(np.dot(U,np.diag(D)),V)

def compressRatio(k):
    nonzero = np.count_nonzero(compress(k))
    mone = 2*nonzero*len(D)+nonzero
    mechane = 2*512*len(D)+512
    return 1- mone/mechane
def forbinius(k):
    G = reconstruct(U,compress(k),V)
    F = np.subtract(image, G)
    return norm(F,'fro')




D= compress(10)
print("ratio --- " , compressRatio(400))
print("forb -- ", forbinius(100))


"""
#graph:forbinius norm
plot([i for i in range(511)], [forbinius(k) for k in range(511)])
axis([0,511,0,10000])
xlabel("compressFactor: k")
ylabel("forbinius: |Mk-Image|")
title("forbibnius distance as factor of k")
show()


#graph: compressRatio
plot([i for i in range(511)], [compressRatio(k) for k in range(511)])
axis([0,500,0,1])
xlabel("compressFactor: k")
ylabel("compressRatio")
title("compressFactor as function of k")
show()
"""
im1 = imshow(reconstruct(U,compress(30),V))
title("k=30")
show(im1)
im2= imshow(reconstruct(U,compress(100),V))
title("k=100")
show(im2)
im3= imshow(reconstruct(U,compress(200),V))
title("k=200")
show(im3)
im4= imshow(reconstruct(U,compress(300),V))
title("k = 300")
show(im4)
im5 = imshow(reconstruct(U,compress(480),V))
title("k=480")
show(im5)
