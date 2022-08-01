import numpy as np
from cvxopt import matrix, solvers
import numpy.random as npr
d = 10  # d variables
n = 100  # number of samples

def gene_X_data(n,d):
    X = np.zeros((d, n))
    for i in range(d):
     X[i, :] = np.random.uniform(size=n)
    return(X)

def gene_Y_data(X):
    Y = 3*X[0,:]- np.power(3*X[1,:]-1,2) + 2*np.cos(2*np.pi*X[2,:])/(2-np.cos(2*np.pi*X[2,:]))+np.sin(2*np.pi*X[3,:])+2*np.cos(2*np.pi*X[4,:])+3*np.power(np.sin(2*np.pi*X[4,:]),2)+4*np.power(np.cos(2*np.pi*X[4,:]),3)+5*np.sin(2*np.pi*X[4,:])+np.cos(2*np.pi*X[5,:])/(2-np.cos(2*np.pi*X[5,:]))
    Y = (Y - np.min(Y)) / (np.max(Y) - np.min(Y))
    return (Y)

##noise npr.normal(0,1,n) npr.standard_t(3,n) npr.chisquare(df=1, n)
train_X=gene_X_data(n,d)
val_X=gene_X_data(n,d)
test_X=gene_X_data(n,d)

train_Y = np.zeros((n, 1))
val_Y=np.zeros((n, 1))
test_Y=np.zeros((n, 1))
train_Y[:, 0] = gene_Y_data(train_X)+0.9*npr.normal(0,1,n)+0.1*npr.normal(0,3.5,n)
val_Y[:,0]=gene_Y_data(val_X)+0.9*npr.normal(0,1,n)+0.1*npr.normal(0,3.5,n)
test_Y[:,0]=gene_Y_data(test_X)

def init_data(n):
    X = np.zeros((n,1))
    X[:, 0]=np.ones(n)
    return(X)

def init_matrix_R_Rn(X,n,d):
  Rn = np.zeros((d,n,n))  # store R1j
  for ind in range(0, d):
    for i in range(0,n):
        for j in range(0,n):
            x1 = X[ind][i]
            x2 = X[ind][j]
            part1 = 1 / 4 * (np.square(x1 - 0.5) - 1 / 12) * (np.square(x2 - 0.5) - 1 / 12)
            part2 = 1 / 24 * (np.power(x1 - x2 - 0.5, 4) - 0.5 * np.square(x1 - x2-0.5) + 7 / 240)
            Rn[ind][i][j] =part1-part2
  R = np.sum(Rn, axis=0)
  R = np.mat(R)
  return Rn,R

def B_data(X):
  B = np.transpose(np.vstack((np.ones((1,n)), (X- 0.5))))
  B = np.mat(B)
  return(B)

##initialize C
def C_data(X,Y,B,R):
  C0 = (np.identity(X.shape[1]) - B * (B.T * B).I * B.T * R).I * (np.power(X.shape[1], -9/7) *np.identity(X.shape[1]) + R).I * (np.identity(X.shape[1]) - B * (B.T * B).I * B.T) * Y
  return(C0)

def belta_data(B,C0):
  belta = (B.T * B).I * B.T * C0
  return(belta)

def RLAND_main(sigma,S,lambda1,X,Y,n,d):
  b = np.zeros((n, 1))
  Y0=init_data(n)
  Y_update=init_data(n)*0
  V = np.zeros(shape=(X.shape[1], d))
  Rn, R = init_matrix_R_Rn(X, n, d)
  Rw = np.zeros(shape=(X.shape[1], X.shape[1]))
  theta = init_data(d)
  B = B_data(X)
  C0 = C_data(X, Y, B, R)
  belta = belta_data(B, C0)
  belta1 = belta_data(B, C0)
  for kk in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]:
    while np.mean((Y0-Y_update) ** 2)>0.00001:
     Y_update=Y0
     b[:, 0] = -1 / 2 * np.e ** (-np.power((Y[:, 0] - Y0[:, 0]) / (sigma), 2) / 2)
     for i in range(d):
       Rw += theta[i, 0] * np.power((C0.T * Rn[i] * C0)[0,0], 1)* Rn[i]
     C = np.linalg.inv(-Rw + np.identity(X.shape[1])) * np.diag(b[:, 0] / sigma) * (Y - B * belta1)
     for i in range(d):
       V[:, i] = (Rn[i] * C * np.power(C0.T * Rn[i] * C0,1)).reshape((X.shape[1],))
     P1 = np.append(-B,B, axis=1)
     P2 = np.append(P1,-V, axis=1)
     P = matrix(P2.T*np.diag(-b[:, 0] / sigma)*P2)

     q1 = np.append(lambda1 * np.power(abs(belta.T),1/5)-Y[:,0].T*np.dot(np.diag(b[:, 0] / sigma),B), lambda1 * np.power(abs(belta.T),1/5)+Y[:,0].T*np.dot(np.diag(b[:, 0] / sigma),B),axis=1)

     q=np.append(q1, C.T*V-np.dot(np.dot(-Y.T,np.diag(b[:, 0] / sigma)),V),axis=1)
     q=matrix(q.T)

     BB=[[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0, np.power(C0.T * Rn[0] * C0,-2)[0, 0], np.power(C0.T * Rn[1] * C0,-2)[0, 0],
       np.power(C0.T * Rn[2] * C0,-2)[0, 0], np.power(C0.T * Rn[3] * C0,-2)[0, 0], np.power(C0.T * Rn[4] * C0,-2)[0, 0],
       np.power(C0.T * Rn[5] * C0,-2)[0, 0], np.power(C0.T * Rn[6] * C0,-2)[0, 0], np.power(C0.T * Rn[7] * C0,-2)[0, 0],
       np.power(C0.T * Rn[8] * C0,-2)[0, 0], np.power(C0.T * Rn[9] * C0,-2)[0, 0]]]
     G111 = np.append(-np.identity((11)), np.zeros((11,21)), axis=1)
     G121= np.append(-np.identity((11))*0, -np.identity((11)), axis=1)
     G22=np.append(G121,np.zeros((11,10)), axis=1)
     G33=np.append(G111,G22,axis=0)
     G55=np.append(np.zeros((10,22)),-np.identity((10)), axis=1)
     G44=np.append(G33,G55,axis=0)
     G66=np.append(G44,BB,axis=0)
     G=matrix(G66)
     h = matrix(np.append(np.zeros((32,1)), np.array([[S]]), axis=0))
     param = solvers.qp(P, q, G, h)

     hat_beta = param['x'][0:11]-param['x'][11:22]
     belta1=np.mat(hat_beta[0:11])
     hat_theta = param['x'][22:]
     theta=np.mat(param['x'][22:])
     para = np.vstack((hat_beta, hat_theta))
    def predict(X, para,Rn,C,C0):
       y_pred = np.zeros((X.shape[1], 1))
       for i in range(1, d+1):
           y_pred +=(para[i, 0]* (X[i - 1, :] - 0.5)).reshape(-1,1) + para[10+i, 0]*np.power(C0.T * Rn[i - 1] * C0, -2)[0, 0] * (Rn[i - 1] * C)
       return y_pred+para[0,0]
    Y0 = predict(X, para, Rn, C,C0)
  return[para,C]

def find_para(sigma,S,lambda1,train_X,train_Y,val_X,val_Y,n,d):
 para,C=RLAND_main(sigma,S,lambda1,train_X,train_Y,n,d)
 B = B_data(train_X)
 Rn1, R1 = init_matrix_R_Rn(train_X, n, d)
 C0 = C_data(train_X, train_Y, B, R1)
 def predict(val_X,n,d,para,C,C0):
     Rn, R = init_matrix_R_Rn(val_X, n, d)
     y_pred = np.zeros((val_X.shape[1], 1))
     for i in range(1, d):
         y_pred += (para[i, 0] * (val_X[i - 1, :] - 0.5)).reshape(-1, 1) + para[10+i, 0]*np.power(C0.T * Rn[i - 1] * C0, -2)[0, 0] * (Rn[i - 1] * C)
     return y_pred + para[0,0]
 Y_hat = predict(val_X,n,d,para,C,C0)
 mse= np.mean((Y_hat - val_Y) ** 2)
 return(mse)

mse_max=10000000000000000000
S_para=0
sigma_para=0
lambda1_para=0
sigmag=np.zeros(11)
for i in range(11):
     sigmag[i]=0.5+0.1*i
Sg = np.zeros(11)
for i in range(26):
     Sg[i] = 0.5 + 0.1 * i
lambda1g=[0.001,0.005,0.01,0.05,0.1,0.5,1,5,10,50,100]
for sigma in sigmag:
   for S in Sg:
       for lambda1 in lambda1g:
         mse=find_para(sigma,S,lambda1,train_X,train_Y,val_X,val_Y,n,d)
         if mse<mse_max:
            mse_max=mse
            S_para=S
            lambda1_para=lambda1
            sigma_para=sigma

param,C= RLAND_main(sigma_para,S_para,lambda1_para,train_X, train_Y, n, d)
B = B_data(train_X)
Rn1, R1 = init_matrix_R_Rn(train_X, n, d)
C0 = C_data(train_X, train_Y, B, R1)
def predict(test_X,n,d,para,C,C0):
    Rn, R = init_matrix_R_Rn(test_X, n, d)
    y_pred = np.zeros((test_X.shape[1], 1))
    for i in range(1, d+1):
        y_pred +=(para[i, 0]* (test_X[i - 1, :] - 0.5)).reshape(-1,1) + para[10+i, 0]*np.power((C0.T * Rn[i - 1] * C0)[0,0],-2) * (Rn[i - 1] * C)
    return y_pred + para[0,0]
Y_hat = predict(test_X,n,d,param,C,C0)
mse =np.mean((Y_hat- test_Y) ** 2)
print(mse)

hat_beta0 = param[0:11,0]
hat_theta = param[11:21,0]
res_beta = np.where(np.array(np.abs(hat_beta0)) <=0.001, 0, 1)
res_theta=np.where(np.array(np.abs(hat_theta)) <= 0.001,0,1)
print('Identified linear variable index:',np.where(res_beta==1))
print('Identified nonlinear variable index:',np.where(res_theta==1))