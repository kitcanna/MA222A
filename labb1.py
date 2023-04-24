import numpy as np
import scipy.integrate as integrate 
import matplotlib.pyplot as plt

############## UPPG 4 ##############
L, tol= integrate.quad(lambda t:np.sqrt((5-5*np.cos(t))**2 + (5*np.sin(t))**2), 0, 4*np.pi)
print(L)
print(tol)
####################################
 
 
############## UPPG 3 ##############
t = np.linspace(0, 4*np.pi, 100); 
x = (5*t)-(5*np.sin(t)); 
y = 5-5*np.cos(t); 

plt.plot(x,y) 
plt.title('Periodisk rörelse längs en bana') 
plt.xlabel('x') 
plt.ylabel('y') 
plt.grid()      
# plt.show()

L, tol= integrate.quad(lambda t:np.sqrt((5-5*np.cos(t))**2 + (5*np.sin(t))**2), 0, 4*np.pi)
# print(L)
# print(tol)
#################################### 


############### UPPG 2 ##############
I, tol= integrate.quad(lambda x:np.exp(-x)*np.cos(x), 0, np.inf)
# print(I)
# print(tol)
###################################


############## UPPG 1 ##############
def trapets(fun,a,b,n):
    h = (b-a)/n # n delintervall
    x = np.linspace(a,b,n+1) # n+1 griddpunkter
    y = fun(x)
    return h*(np.sum(y[1:-1]) + 0.5*(y[0] + y[-1]))

f = lambda x:(np.log(1 + np.sin(x)) * np.cos(x))
# print ("n = 10: ", trapets(f, 0, np.pi/2, 10))
# print ("n = 100: ", trapets(f, 0, np.pi/2, 100))
# print ("n = 1000: ", trapets(f, 0, np.pi/2, 1000))
###################################

