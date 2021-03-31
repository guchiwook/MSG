
import numpy as np
import random
import matplotlib.pyplot as plt

def linearregressionfit(w,b,x,d):

    alpha = 0.001 #alpha는 학습률임 learning_rate

    x_num, x_num_factor = x.shape

    dw = - alpha * 2 / x_num * np.sum((w*x+b-d)*x)
    db = - alpha * 2 / x_num * np.sum(w*x+b-d)

    w = w + dw
    b = b + db

    return w, b

x = np.array([[8.70153760], [3.90825773], [1.89362433], [3.28730045],
              [7.39333004], [2.98984649], [2.25757240], [9.84450732], [9.94589513], [5.48321616]])

d = np.array([[5.64413093], [3.75876583], [3.87233310], [4.40990425],
              [6.43845020], [4.02827829], [2.26105955], [7.15768995], [6.29097441], [5.19692852]])

y = np.zeros(d.shape)

x_num, x_num_factor = x.shape
print('x.shape=',x.shape)

w = random.uniform(-0.5,0.5)
b = random.uniform(-0.5,0.5)
print(w,b)


num_epoch = 100
e = np.zeros((num_epoch,1))

for i in range(num_epoch):

    w, b = linearregressionfit(w,b,x,d)

    y = w*x+b

    e[i,0] = np.sum((d-y)*(d-y)) / x_num


print('y=%f*x+%f'%(w,b))

fig1 = plt.figure(1)
fig1_plt1 = plt.subplot(2,1,1)
fig1_plt2 = plt.subplot(2,1,2)

fig1_plt1.scatter(x,d,c='r',label='results')
fig1_plt1.plot(x,y,'b',label='predict')
fig1_plt1.set_title('Linear Regression: y=%0.2f*x+%0.2f'%(w,b))
fig1_plt1.legend(loc='best')

fig1_plt2.plot(e,label='error')
fig1_plt2.set_title('error')
fig1_plt2.set_xlabel('epoch')
fig1_plt2.legend(loc='best')

plt.subplots_adjust(wspace=1,hspace=1)

plt.show()

'''
fig1 = plt.figure(1)
fig1_plt1 = plt.subplot(1,1,1)


fig1_plt1.plot(arrE,'r',label='Error')

fig1_plt1.set_xlabel('Epoch')
fig1_plt1.set_ylabel('Average of Training error')
fig1_plt1.grid(True)
fig1_plt1.legend(loc='best')

plt.show()
'''