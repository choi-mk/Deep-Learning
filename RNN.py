#!/usr/bin/env python
# coding: utf-8

# In[32]:


import numpy as np
import gzip
import matplotlib.pyplot as plt
import emo_utils
import emoji


X_train, t_train=emo_utils.read_csv("train_emoji.csv")
X_test, t_test=emo_utils.read_csv("test_emoji.csv")

X_train=[sentence.split() for sentence in X_train]
X_test=[sentence.split() for sentence in X_test]

g50d_wi, g50d_iw, g50d_wv=emo_utils.read_glove_vecs("glove.6B.50d.txt")
g100d_wi, g100d_iw, g100d_wv=emo_utils.read_glove_vecs("glove.6B.100d.txt")

ex_train50 = np.zeros((len(X_train), 10, 50))
ex_test50 = np.zeros((len(X_test), 10, 50))
ex_train100 = np.zeros((len(X_train), 10, 100))
ex_test100 = np.zeros((len(X_test), 10, 100))

for i in range(len(X_train)):
    for j in range(len(X_train[i])):
        ex_train50[i][j] = g50d_wv[X_train[i][j].lower()]
        
for i in range(len(X_test)):
    for j in range(len(X_test[i])):
        ex_test50[i][j] = g50d_wv[X_test[i][j].lower()]  
        
for i in range(len(X_train)):
    for j in range(len(X_train[i])):
        ex_train100[i][j] = g100d_wv[X_train[i][j].lower()]
        
for i in range(len(X_test)):
    for j in range(len(X_test[i])):
        ex_test100[i][j] = g100d_wv[X_test[i][j].lower()]  


# In[3]:


def xavier(input, output):
    limit = np.sqrt(6 / (input + output))
    return np.random.uniform(-limit, limit, size=(input, output))

def CE(y, t):
    error = -np.sum(t * np.log(y+ 1e-9)) / y.shape[0]     # add 1e-9 because of overflow
    return error

class Softmax:
    def __init__(self):
        self.y = None 
        self.t = None 
        
    def forward(self,x):
        p = np.exp(x-np.max(x, axis=1, keepdims=True))
        self.y = p / np.sum(p, axis=1, keepdims=True)
        return self.y

    def backward(self, d_out, learning_rate, opt_type):
        d_x = (self.y - self.t) / self.t.shape[0]
        return d_x

class Linear:
    def __init__(self, input_size, output_size):
        self.W = xavier(input_size, output_size)
        self.b = np.zeros(output_size)
        self.x = None

    def forward(self, x):
        self.x = x[:,-1,:]
        y = np.dot(self.x, self.W) + self.b
        return y

    def backward(self, d_out, learning_rate, opt_type):
        dW = np.dot(self.x.T, d_out)
        db = np.sum(d_out, axis=0)
        d_x = np.dot(d_out, self.W.T)
        self.W -= learning_rate * dW
        self.b -= learning_rate * db
        return d_x
    
def one_hot_encoding(x):
    label=np.zeros((len(x), 5))
    for i in range(len(x)):
        label[i][x[i]]=1
    return label

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# In[10]:


class ADAM:
    def __init__(self):
        self.b1=0.9
        self.b2=0.999
        self.fm=0
        self.sm=0
    def optimize(self, dparam, t, learning_rate):
        self.fm=self.b1*self.fm+(1-self.b1)*dparam
        self.sm=self.b2*self.sm+(1-self.b2)*(dparam**2)
        fu=self.fm/(1-self.b1**t)
        su=self.sm/(1-self.b2**t)
        param=learning_rate*fu/(np.sqrt(su)+1e-7)
        return param
    
class Dropout:
    def __init__(self, p):
        self.p=p
        self.mask=None
    def forward(self, x):
        self.mask=np.random.rand(*x.shape)>self.p
        return x*self.mask

    def backward(self, dh, learning_rate, opt_type):
        if dh.ndim==2:
            mask=self.mask[:,9,:]
        else:
            mask=self.mask
        return dh*mask

class RNN_cell:
    def __init__(self, inputs, outputs):
        self.Wx=xavier(inputs, outputs)
        self.Wh=xavier(inputs, inputs)
        self.b=np.zeros(outputs)
        self.adam_Wx=ADAM()
        self.adam_Wh=ADAM()
        self.adam_b=ADAM()
        self.t=1
        self.x=None
        self.h=None
        
    def forward(self, x, h_before):
        self.h_before=h_before
        self.x=x
        self.h=np.tanh(np.dot(x,self.Wx)+np.dot(h_before,self.Wh)+self.b)
        return self.h
        
    def backward(self, dh, learning_rate, opt_type):
        dtanh=(1-self.h**2)*dh
        db=np.sum(dtanh, axis=0)
        
        dWx=np.dot(self.x.T, dtanh)
        dWh=np.dot(self.h_before.T,dtanh)
        dx=np.dot(dtanh,self.Wx.T)
        dh_before=np.dot(dtanh,self.Wh.T)
        
        if opt_type=="SGD":
            self.Wx-=learning_rate*dWx
            self.Wh-=learning_rate*dWh
            self.b-=learning_rate*db
        else:
            self.Wx-=self.adam_Wx.optimize(dWx, self.t, learning_rate)
            self.Wh-=self.adam_Wh.optimize(dWh, self.t, learning_rate)
            self.b-=self.adam_b.optimize(db, self.t, learning_rate)
            self.t+=1
        return dx, dh_before
            
class vanila_RNN:
    def __init__(self, inputs, outputs):
        self.inputs=inputs
        self.outputs=outputs
        self.layers=[]
        for i in range(10):
            cell=RNN_cell(inputs, outputs)
            self.layers.append(cell)
        self.h_result=None
        self.dx_result=None
        
    def forward(self, x):
        self.x=x
        h_before=np.zeros_like(x[:,0,:])
        self.h_result=np.zeros_like(x)
        i=0
        for layer in self.layers:
            h_before=layer.forward(x[:,i,:], h_before)
            self.h_result[:,i,:]=h_before
            i+=1
        return self.h_result
    
    def backward(self, dhm, learning_rate, opt_type):
        layers=self.layers[::-1]
        self.dx_result=np.zeros_like(self.x)
        i=9
        dh=0
        if(dhm.ndim==2):
            newm=np.zeros_like(self.h_result)
            newm[:,9,:]=dhm
            self.h_result[:,9,:]=dhm
            dhm=newm
        for layer in layers:
            dx, dh = layer.backward(dhm[:,i,:]+dh, learning_rate, opt_type)
            self.dx_result[:,i,:]=dx
            i-=1
        return self.dx_result
                   

class RNN:
    def __init__(self, opt_type):
        self.layers = []
        self.opt_type=opt_type

        
    def accuracy(self, x, t):
        y = self.forwarding(x)
        y = np.argmax(y, axis=1)
        t_ans= np.argmax(t, axis=1)
        accuracy=np.sum(y==t_ans) / float(x.shape[0])
        print("Accuracy:",accuracy)
        return y
    
    def forwarding(self, x):
        for layer in self.layers:
            x=layer.forward(x)
        return x
        
    def backwarding(self, learning_rate):
        dh = 1
        layers=self.layers[::-1]
        for layer in layers:
            dh = layer.backward(dh, learning_rate, self.opt_type)

            
    def train(self, x_train, t_train, epoches, batch):
        learning_rate=0.001
        train_errors = []
        for epoch in range(1, epoches+1):
            batch_train=np.random.choice(x_train.shape[0], batch) 
            x_train_b=x_train[batch_train]
            t_train_b=t_train[batch_train]
            self.forwarding(x_train_b)
            self.layers[-1].t=t_train_b
            loss = CE(self.layers[-1].y, self.layers[-1].t)
            train_errors.append(loss)
            self.backwarding(learning_rate)
            self.accuracy(x_train, t_train)
        return train_errors
    


# In[36]:


class LSTM_cell:
    def __init__(self, inputs, outputs):
        self.Wx=xavier(inputs, 4*outputs)
        self.Wh=xavier(outputs, 4*outputs)
        self.b=np.zeros(4*outputs)
        self.input=inputs
        self.adam_Wx=ADAM()
        self.adam_Wh=ADAM()
        self.adam_b=ADAM()
        self.t=1
        
    def forward(self, x, h_before, c_before):
        self.c_before=c_before
        self.h_before=h_before
        self.x=x
        ans=np.dot(x,self.Wx)+np.dot(h_before,self.Wh)+self.b
        self.f_r=sigmoid(ans[:,:self.input])
        self.c_r=np.tanh(ans[:,self.input:2*self.input])
        self.i_r=sigmoid(ans[:,2*self.input:3*self.input])
        self.o_r=sigmoid(ans[:,3*self.input:4*self.input])
        self.c=(self.f_r*self.c_before)+(self.i_r*self.c_r)
        self.h=self.o_r*np.tanh(self.c)
        return self.h, self.c
    
    def backward(self, dh, dc, learning_rate, opt_type):
        a=dc+(dh*self.o_r)*(1-np.square(np.tanh(self.c)))
        do_r=dh*np.tanh(self.c)*self.o_r*(1-self.o_r)
        dc_r=a*self.i_r*(1-np.square(self.c_r))
        di_r=a*self.c_r*self.i_r*(1-self.i_r)
        df_r=a*self.c_before*self.f_r*(1-self.f_r)
        dc_before=a*self.f_r
        dtanh=np.hstack((df_r, dc_r, di_r, do_r))
        dx=np.dot(dtanh, self.Wx.T)
        dWx=np.dot(self.x.T, dtanh)
        #print(dWx)
        dWh=np.dot(self.h_before.T, dtanh)
        dh_before=np.dot(dtanh, self.Wh.T)
        db=np.sum(dtanh, axis=0)
        if opt_type=="SGD":
            self.Wx-=learning_rate*dWx
            self.Wh-=learning_rate*dWh
            self.b-=learning_rate*db
        else:
            self.Wx-=self.adam_Wx.optimize(dWx, self.t, learning_rate)
            self.Wh-=self.adam_Wh.optimize(dWh, self.t, learning_rate)
            self.b-=self.adam_b.optimize(db, self.t, learning_rate)
            self.t+=1
        return dx,dh_before, dc_before
            
            
            
            
class LSTM:
    def __init__(self, inputs, outputs):
        self.inputs=inputs
        self.outputs=outputs
        self.layers=[]
        for i in range(10):
            cell=LSTM_cell(inputs, outputs)
            self.layers.append(cell)        
    def forward(self, x):
        self.x=x
        i=0
        c_before=np.zeros_like(x[:,0,:])
        h_before=np.zeros_like(x[:,0,:])
        self.h_result=np.zeros_like(x)
        self.c_result=np.zeros_like(x)
        for layer in self.layers:
            h_before, c_before=layer.forward(x[:,i,:],h_before, c_before)
            self.h_result[:,i,:]=h_before
            self.c_result[:,i,:]=c_before
            i+=1
        return self.h_result
    def backward(self, dhm, learning_rate, opt_type):
        layers=self.layers[::-1]
        self.dx_result=np.ones_like(self.x)
        i=9
        dh=0
        dc=0
        dx=0
        if dhm.ndim==2:
            newm=np.zeros_like(self.h_result)
            newm[:,9,:]=dhm
            self.h_result[:,9,:]=dhm
            dhm=newm
        for layer in layers:
            dx, dh, dc=layer.backward(dhm[:,i,:]+dh, dc, learning_rate, opt_type)
            self.dx_result[:,i,:]=dx
            i-=1
        return self.dx_result
        


# In[20]:


t_test_one=one_hot_encoding(t_test)
t_train_one=one_hot_encoding(t_train)

emoji_dictionary = {"0": "\u2764\uFE0F",    # :heart: prints a black instead of red heart depending on the font
                    "1": ":baseball:",
                    "2": ":smile:",
                    "3": ":disappointed:",
                    "4": ":fork_and_knife:"}
def label_to_emoji(label):
    """
    Converts a label (int or string) into the corresponding emoji code (string) ready to be printed
    """
    for i in range(len(label)):
        print(emoji.emojize(emoji_dictionary[str(label[i])], language="alias"), end=" ")


# In[27]:


RNN_1=RNN("SGD")
RNN_1.layers.append(vanila_RNN(50,50))# output 사이즈 변경 가능하게 수정
RNN_1.layers.append(vanila_RNN(50,50))
RNN_1.layers.append(Linear(50 , 5))
RNN_1.layers.append(Softmax())
train_errors1=RNN_1.train(ex_train50, t_train_one, 2000, 100)


t_ans=RNN_1.accuracy(ex_test50, t_test_one)
label_to_emoji(t_ans)

#Loss graph
plt.plot(train_errors1, color='red', linestyle='-', label='train')  # 선 그래프 그리기
plt.legend()
plt.title('Loss Graph')  # 그래프 제목 설정
plt.xlabel('Epoch')  # x 축 레이블 설정
plt.ylabel('Loss')  # y 축 레이블 설정


# In[31]:


RNN_2=RNN("SGD")
RNN_2.layers.append(LSTM(50,50))
RNN_2.layers.append(LSTM(50,50))
RNN_2.layers.append(Linear(50 , 5))
RNN_2.layers.append(Softmax())
train_error2=RNN_2.train(ex_train50, t_train_one, 2000, 100)

t_ans=RNN_2.accuracy(ex_test50, t_test_one)
label_to_emoji(t_ans)

#Loss graph
plt.plot(train_error2, color='red', linestyle='-', label='train')  # 선 그래프 그리기
plt.legend()
plt.title('Loss Graph')  # 그래프 제목 설정
plt.xlabel('Epoch')  # x 축 레이블 설정
plt.ylabel('Loss')  # y 축 레이블 설정


# In[29]:


RNN_3=RNN("ADAM")
RNN_3.layers.append(LSTM(50,50))
RNN_3.layers.append(LSTM(50,50))
RNN_3.layers.append(Linear(50 , 5))
RNN_3.layers.append(Softmax())
train_error3=RNN_3.train(ex_train50, t_train_one, 2000, 100)

t_ans=RNN_3.accuracy(ex_test50, t_test_one)
label_to_emoji(t_ans)

#Loss graph
plt.plot(train_error3, color='red', linestyle='-', label='train')  # 선 그래프 그리기
plt.legend()
plt.title('Loss Graph')  # 그래프 제목 설정
plt.xlabel('Epoch')  # x 축 레이블 설정
plt.ylabel('Loss')  # y 축 레이블 설정


# In[37]:


RNN_4=RNN("SGD")
RNN_4.layers.append(LSTM(100,100))
RNN_4.layers.append(LSTM(100,100))
RNN_4.layers.append(Linear(100 , 5))
RNN_4.layers.append(Softmax())
train_error4=RNN_4.train(ex_train100, t_train_one, 2000, 100)

t_ans=RNN_4.accuracy(ex_test100, t_test_one)
label_to_emoji(t_ans)

#Loss graph
plt.plot(train_error4, color='red', linestyle='-', label='train')  # 선 그래프 그리기
plt.legend()
plt.title('Loss Graph')  # 그래프 제목 설정
plt.xlabel('Epoch')  # x 축 레이블 설정
plt.ylabel('Loss')  # y 축 레이블 설정


# In[38]:


RNN_5=RNN("SGD")
RNN_5.layers.append(LSTM(100,100))
RNN_5.layers.append(Dropout(0.3))
RNN_5.layers.append(LSTM(100,100))
RNN_5.layers.append(Dropout(0.3))
RNN_5.layers.append(Linear(100 , 5))
RNN_5.layers.append(Softmax())
train_error5=RNN_5.train(ex_train100, t_train_one, 2000, 100)

t_ans=RNN_5.accuracy(ex_test100, t_test_one)
label_to_emoji(t_ans)

#Loss graph
plt.plot(train_error5, color='red', linestyle='-', label='train')  # 선 그래프 그리기
plt.legend()
plt.title('Loss Graph')  # 그래프 제목 설정
plt.xlabel('Epoch')  # x 축 레이블 설정
plt.ylabel('Loss')  # y 축 레이블 설정


# In[39]:


RNN_6=RNN("ADAM")
RNN_6.layers.append(vanila_RNN(50,50))# output 사이즈 변경 가능하게 수정
RNN_6.layers.append(vanila_RNN(50,50))
RNN_6.layers.append(Linear(50 , 5))
RNN_6.layers.append(Softmax())
train_errors6=RNN_6.train(ex_train50, t_train_one, 2000, 100)


t_ans=RNN_6.accuracy(ex_test50, t_test_one)
label_to_emoji(t_ans)

#Loss graph
plt.plot(train_errors6, color='red', linestyle='-', label='train')  # 선 그래프 그리기
plt.legend()
plt.title('Loss Graph')  # 그래프 제목 설정
plt.xlabel('Epoch')  # x 축 레이블 설정
plt.ylabel('Loss')  # y 축 레이블 설정


# In[41]:


RNN_7=RNN("ADAM")
RNN_7.layers.append(LSTM(100,100))
RNN_7.layers.append(LSTM(100,100))
RNN_7.layers.append(Linear(100 , 5))
RNN_7.layers.append(Softmax())
train_error7=RNN_7.train(ex_train100, t_train_one, 2000, 100)

t_ans=RNN_7.accuracy(ex_test100, t_test_one)
label_to_emoji(t_ans)

#Loss graph
plt.plot(train_error7, color='red', linestyle='-', label='train')  # 선 그래프 그리기
plt.legend()
plt.title('Loss Graph')  # 그래프 제목 설정
plt.xlabel('Epoch')  # x 축 레이블 설정
plt.ylabel('Loss')  # y 축 레이블 설정


# In[ ]:




