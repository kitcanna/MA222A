import numpy as np
import scipy.integrate as integrate 
import matplotlib.pyplot as plt


# --------------- UPPG 3 ---------------
m = k = A = 1
w0 = np.sqrt(k/m)
w = [1.0, 0.9, 0.8]

fig ,ax = plt.subplots()

for value in w:
    f = lambda t,x: [ x[1], -(k/m)*x[0] + (A/m)*np.cos(value * t) ]
    sol = integrate.solve_ivp(f,[0,100],[0, 0],\
    t_eval=np.linspace(0,100,500)) 
    plt.plot(sol.t,sol.y[0], label='w={}'.format(value))

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend(fontsize=14)
ax.set_xlabel('x',fontsize=12)
ax.set_ylabel('y',fontsize=12)
ax.tick_params(labelsize=10)
plt.grid()
plt.title("Title", fontsize=10)
plt.show()
# ----------------------------------------


# --------------- UPPG 2 ---------------
p = 0.3 #(c/mL)
q = 1 #(g/L)
a = np.pi/2
b = 0

f = lambda x,y: [y[1], -p*y[1] - q*np.sin(y[0])]
sol = integrate.solve_ivp(f,[0,30],[a, b],\
t_eval=np.linspace(0,30,100)) 

fig ,ax = plt.subplots()
ax.plot(sol.t,sol.y[0]) 
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend(fontsize=14)
ax.set_xlabel('Tid (s)',fontsize=12)
ax.set_ylabel('Förskjutning',fontsize=12)
ax.tick_params(labelsize=10)
plt.grid()
plt.title("Förskjutning från pendelns jämnviktsläge", fontsize=10)
# plt.show()
# ----------------------------------------


# --------------- UPPG 1 ---------------
g = 9.82
m = 100
k = 40 

f = lambda x,y: g-((k/m)*(y**2)) 
sol = integrate.solve_ivp(f,[0,5],[20],\
t_eval=np.linspace(0,5,100)) 

fig ,ax = plt.subplots()
ax.plot(sol.t,sol.y[0]) 
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend(fontsize=14)
ax.set_xlabel('Tid [s]',fontsize=12)
ax.set_ylabel('Accelerationshastigheten [m/s^2]',fontsize=12)
ax.tick_params(labelsize=10)
plt.grid()
plt.title("Accelerationshastigheten hos en fallskärmshoppare efter utlösning av fallskärm", fontsize=10)
# plt.show()
# ----------------------------------------


# --------------- Euler ---------------
def euler(f,a,b,y0,n):
    x = np.linspace(a,b,n+1) # n+1 gridpunkter
    y = np.zeros(n+1)
    h = (b-a)/n       #Steglängd
    y[0] = y0
    for k in range(n):
        y[k+1] = y[k] + h*f(x[k],y[k])
    return x,y

f = lambda x,y: x-y
y0=1

(x1,y1) = euler(f,0,3,y0,50)
(x2,y2) = euler(f,0,3,y0,150)

x = np.linspace(0,3,50)
y = x-1+(y0+1)*np.exp(-x)  #Exakta lösningen
fig, ax =plt.subplots()

ax.plot(x1,y1,'+',label='euler, n=50')
ax.plot(x2,y2,'--',label='euler, n=150')
ax.plot(x,y,label='exakt')
ax.legend(fontsize=14)
ax.set_xlabel('x',fontsize=14)
ax.set_ylabel('y',fontsize=14)
ax.tick_params(labelsize=14)
#plt.show()
# ----------------------------------------

# --------------- Runge-Kuttas ---------------
def rk4(f,a,b,y0,n):
    import numpy as np
    x = np.linspace(a,b,n+1) # n+1 gridpunkter
    y = np.zeros(n+1)
    h = (b-a)/n       #Steglängd
    y[0] = y0
    for k in range(n):
        k0 = h*f(x[k],y[k])
        k1 = h*f(x[k]+h/2,y[k]+k0/2)
        k2 = h*f(x[k]+h/2,y[k]+k1/2)
        k3 = h*f(x[k]+h,y[k]+k2)
        y[k+1] = y[k]+(k0+2*(k1+k2)+k3)/6
    return x,y

(x1,y1) = rk4(f,0,3,1,30)

x = np.linspace(0,3,50)
y = x-1+(y0+1)*np.exp(-x) #Exakta lösningen
fig, ax =plt.subplots()

ax.plot(x1,y1,'+',label='rk4, n=30')
ax.plot(x,y,'-k',label='exakt')
ax.legend(fontsize=14)
ax.set_xlabel('x',fontsize=14)
ax.set_ylabel('y',fontsize=14)
ax.tick_params(labelsize=14)
#plt.show()
# ----------------------------------------
