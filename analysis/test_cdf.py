import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


x=np.arange(1,7)
print(x)
frequency=np.array([3,8,4,5,3,650])
pdf=frequency/np.sum(frequency)
print(pdf)
cdf=np.cumsum(pdf)

# plt.plot(x,pdf, marker="o",label="PMF")
plt.plot(x,cdf,marker="o",label="CDF")
plt.xlim(0,700)
plt.ylim(0,1)
plt.xlabel("X")
plt.ylabel("Probability Values")
plt.title("CDF for discrete distribution")
plt.legend()
plt.show()

