import numpy as np
import gzip
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

train_image_path='train-images-idx3-ubyte.gz'
train_label_path='train-labels-idx1-ubyte.gz'
test_image_path='t10k-images-idx3-ubyte.gz'
test_label_path='t10k-labels-idx1-ubyte.gz'

def load_mnist_images(filename):
    with gzip.open(filename, 'rb') as file:
        data = np.frombuffer(file.read(), dtype=np.uint8, offset=16)
    return data.reshape(-1, 28, 28)
def load_mnist_labels(filename):
    with gzip.open(filename, 'rb') as file:
        data = np.frombuffer(file.read(), dtype=np.uint8, offset=8)
    return data

x_train = load_mnist_images(train_image_path)
t_train = load_mnist_labels(train_label_path)
x_test = load_mnist_images(test_image_path)
t_test = load_mnist_labels(test_label_path)

#Normalize
x_train=x_train/255.0
x_test=x_test/255.0

#change dimension
x_train = x_train.reshape(60000, 1, 28, 28)
x_test = x_test.reshape(10000, 1, 28, 28)

def one_hot_encoding(x):
    label=np.zeros((len(x), 10))
    for i in range(len(x)):
        label[i][x[i]]=1
    return label

#initialize
def xavier(n_type, data):
    if n_type=='L':
        inputs, outputs=data
        limit = np.sqrt(6 / (inputs + outputs))
        return np.random.uniform(-limit, limit, size=(inputs, outputs))
    else:
        inputs = np.prod(data[1:])
        outputs = data[0]
        limit = np.sqrt(6 / (inputs + outputs))
        return np.random.uniform(-limit, limit, size=data)

def im2col(data, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = data.shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1
    org = np.pad(data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')
    org=org.transpose(0,2,3,1)
    col = np.zeros((N, out_h, out_w, C, filter_h, filter_w))

    for h in range(filter_h):
        h_size = out_h*stride+h
        for w in range(filter_w):
            w_size = stride * out_w + w
            col[:, :, :, :, h, w] = org[:, h:h_size:stride, w:w_size:stride,:]

    col = col.reshape(N * out_h * out_w, filter_h * filter_w * C)
    return col

def col2im(col, data_shape, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = data_shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w)

    org = np.zeros((N, H + 2 * pad + stride - 1, W + 2 * pad + stride - 1,C))
    for h in range(filter_h):
        h_size = out_h*stride+h
        for w in range(filter_w):
            w_size = stride * out_w + w
            org[:, h:h_size:stride, w:w_size:stride,:] += col[:, :, :, :, h, w]

    return org[:, pad:H + pad, pad:W + pad,:].transpose(0,3,1,2)

class Linear:
    def __init__(self, input_size, output_size):
        self.W = xavier('L', (input_size, output_size))
        self.b = np.zeros(output_size)
        self.x = None
        self.y = None

    def forward(self, x):
        self.x = x 
        if self.x.ndim==4:
            self.x=self.x.reshape(self.x.shape[0], -1)
        return np.dot(self.x, self.W) + self.b

    def backward(self, d_out, learning_rate):
        dW = np.dot(self.x.T, d_out)
        db = np.sum(d_out, axis=0)
        d_x = np.dot(d_out, self.W.T)
        self.W -= learning_rate * dW
        self.b -= learning_rate * db
        return d_x
    
def CE(y, t):
    error = -np.sum(t * np.log(y+ 1e-9)) / y.shape[0]     # add 1e-9 because of overflow
    return error
    
class Softmax:
    def __init__(self):
        self.y = None 
        self.t = None 
        
    def forward(self,x):
        p = np.exp(x-np.max(x, axis=1, keepdims=True)) #overflow(add -max)
        self.y = p / np.sum(p, axis=1, keepdims=True)
        return self.y

    def backward(self, d_out, learning_rate):
        d_x = (self.y - self.t) / self.t.shape[0]
        return d_x

class Relu:
    def __init__(self):
        self.mark = None

    def forward(self, x):
        self.mark = (x>0)
        out = np.zeros(x.shape)
        out[self.mark] = x[self.mark]
        return out

    def backward(self, d_out, learning_rate=None):
        d_x = np.zeros(d_out.shape)
        d_x[self.mark] = d_out[self.mark]
        return d_x

class Convolution:
    def __init__(self,W_shape, stride=1, pad=0):
        self.W = xavier('C', W_shape)
        self.col_W = self.W.reshape(W_shape[0], -1).T
        self.b = np.zeros(W_shape[0])
        self.stride = stride
        self.pad = pad
        self.x_shape = None   
        self.col = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x_shape = x.shape
        FN, FC, FH, FW = self.W.shape
        N, C, H, W = self.x_shape
        self.col = im2col(x, FH, FW, self.stride, self.pad)
        out_h = 1 + (H + 2*self.pad - FH) // self.stride
        out_w = 1 + (W + 2*self.pad - FW) // self.stride
        out = (np.dot(self.col, self.col_W) + self.b).reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)
        return out

    def backward(self, d_out, learning_rate):
        FN, FC, FH, FW = self.W.shape
        d_out = d_out.transpose(0,2,3,1).reshape(-1, FN)
        self.db = np.sum(d_out, axis=0)
        self.dW = np.dot(self.col.T, d_out).T.reshape(FN, FC, FH, FW)
        dc = np.dot(d_out, self.col_W.T)
        dx = col2im(dc, self.x_shape, FH, FW, self.stride, self.pad)
        self.W-=learning_rate*self.dW
        self.b-=learning_rate*self.db
        return dx
    
class Max_Pooling:
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.pool_size=pool_h*pool_w
        self.stride = stride
        self.pad = pad
        self.x_shape = None
        self.max_index = None
        self.out_h=None
        self.out_w=None

    def forward(self, x):
        self.x_shape = x.shape
        N, C, H, W = self.x_shape
        self.out_h = 1 + (H - self.pool_h) // self.stride
        self.out_w = 1 + (W - self.pool_w) // self.stride
        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_size)
        self.max_index = np.argmax(col, axis=1)
        output = np.max(col, axis=1)
        result = output.reshape(N, self.out_h, self.out_w, C).transpose(0, 3, 1, 2)
        return result

    def backward(self, d_out,learning_rate):
        if d_out.ndim!=4:
            d_out=d_out.reshape(self.x_shape[0], self.x_shape[1], self.out_h, self.out_w)
        d_out = d_out.transpose(0, 2, 3, 1)
        dmax = np.zeros((d_out.size, self.pool_size))
        dmax[np.arange(self.max_index.size), self.max_index.flatten()] = d_out.flatten()
        dmax = dmax.reshape(d_out.shape[0] * d_out.shape[1] * d_out.shape[2], d_out.shape[3]*self.pool_size)
        dx = col2im(dmax, self.x_shape, self.pool_h, self.pool_w, self.stride, self.pad)
        return dx
    
class ConvNeuralNetwork:
    def __init__(self):
        self.layers = []
        
    def accuracy(self, x, t):
        y = self.forwarding(x)
        y = np.argmax(y, axis=1)
        t1= np.argmax(t, axis=1)
        accuracy=np.sum(y==t1) / x.shape[0]
        print("Accuracy:",accuracy)
        return accuracy

    def forwarding(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backwarding(self, learning_rate):
        d_out = 1
        layers=self.layers[::-1]
        for layer in layers:
            d_out = layer.backward(d_out, learning_rate)

    def train(self, x_train, t_train, x_test, t_test, batch, epoches):
        learning_rate = 0.01 
        train_errors = []
        test_errors=[]
        it = 20
        for epoch in range(1, epoches+1):
            print("Epoch", epoch)
            for i in range(it):
                batch_train = np.random.choice(x_train.shape[0], batch) 
                x_batch = x_train[batch_train] 
                t_batch = t_train[batch_train] 
                self.forwarding(x_batch)
                self.layers[-1].t=t_batch
                loss = CE(self.layers[-1].y, self.layers[-1].t)
                train_errors.append(loss)
                self.backwarding(learning_rate)
                batch_test = np.random.choice(x_test.shape[0], batch) 
                x_batch = x_test[batch_test] 
                t_batch = t_test[batch_test] 
                self.forwarding(x_batch)
                self.layers[-1].t=t_batch
                loss = CE(self.layers[-1].y, self.layers[-1].t)
                test_errors.append(loss)
                self.accuracy(x_batch, t_batch)
        return train_errors, test_errors


#CNN 실행
CNN = ConvNeuralNetwork()
CNN.layers.append(Convolution((32, 1, 5, 5)))
CNN.layers.append(Relu())
CNN.layers.append(Max_Pooling(2,2,2))
CNN.layers.append(Convolution((64, 32 , 5, 5)))
CNN.layers.append(Relu())
CNN.layers.append(Max_Pooling(2,2,2))
CNN.layers.append(Linear(1024,10))
CNN.layers.append(Softmax())
t_train_one=one_hot_encoding(t_train)
t_test_one=one_hot_encoding(t_test)
train_errors, test_errors=CNN.train(x_train, t_train_one, x_test, t_test_one, 300, 10)



#Loss graph
plt.plot(train_errors, color='red', linestyle='-', label='train')  # 선 그래프 그리기
plt.plot(test_errors, color='blue', linestyle='-', label='test')  # 선 그래프 그리기
plt.legend()
plt.title('Loss Graph')  # 그래프 제목 설정
plt.xlabel('Epoch')  # x 축 레이블 설정
plt.ylabel('Loss')  # y 축 레이블 설정



#confusion matrix
y_pred=CNN.forwarding(x_test)
result=[]
for row in y_pred:
    max_index=np.argmax(row)
    result.append(max_index)

cm = confusion_matrix(t_test, result)
cm1=np.zeros((10,10))
for i in range(len(cm)):
    row_sum=np.sum(cm[i])
    for j in range(len(cm[i])):
        cm1[i][j]=round(cm[i,j]/ row_sum,2)

plt.figure(figsize=(8, 6))
sns.heatmap(cm1, annot=True, fmt=".2f", cmap="Blues")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()



#image 3개 출력
top_indices = []
x_test = load_mnist_images(test_image_path)
for col in range(y_pred.shape[1]):
    col_values = y_pred[:, col]
    sorted_indices = np.argsort(-col_values)  
    top_indices.append(sorted_indices[:3])  

start=0
for a in top_indices:
    num=0
    print(start,":")
    for i in a:
        plt.subplot(1, 3, num + 1)
        plt.imshow(x_test[i], cmap='gray')
        print(round(y_pred[i][start]*100,2),"%")
        num+=1
    plt.show()
    start+=1

