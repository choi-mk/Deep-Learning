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
x_train=x_train.reshape(60000,784)
x_test=x_test.reshape(10000,784)


# In[2]:


def xavier(input, output):
    limit = np.sqrt(6 / (input + output))
    return np.random.uniform(-limit, limit, size=(input, output))

def one_hot_encoding(x):
    label=np.zeros((len(x), 10))
    for i in range(len(x)):
        label[i][x[i]]=1
    return label

def CE(y, t):
    error = -np.sum(t * np.log(y+ 1e-9)) / y.shape[0]     # add 1e-9 because of overflow
    return error
    

class Linear:
    def __init__(self, input_size, output_size):
        self.W = xavier(input_size, output_size)
        self.b = np.zeros(output_size)
        self.x = None

    def forward(self, x):
        self.x = x 
        y = np.dot(self.x, self.W) + self.b
        return y

    def backward(self, d_out, learning_rate):
        dW = np.dot(self.x.T, d_out)
        db = np.sum(d_out, axis=0)
        d_x = np.dot(d_out, self.W.T)
        self.W -= learning_rate * dW
        self.b -= learning_rate * db
        return d_x

class Softmax:
    def __init__(self):
        self.y = None 
        self.t = None 
        
    def forward(self,x):
        p = np.exp(x-np.max(x, axis=1, keepdims=True))
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

    
class NeuralNetwork:
    def __init__(self):
        self.layers = []

    def accuracy(self, x, t):
        y = self.forwarding(x)
        y = np.argmax(y, axis=1)
        t1= np.argmax(t, axis=1)
        accuracy=np.sum(y==t1) / float(x.shape[0])
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

    def train(self, x_train, t_train, x_test, t_test, epoches):
        learning_rate = 0.1 
        train_errors = []
        test_errors=[]
        for epoch in range(1, epoches+1):
            print("Epoch {0}/{1}".format(epoch, epoches))
            self.forwarding(x_train)
            self.layers[-1].t=t_train
            loss = CE(self.layers[-1].y, self.layers[-1].t)
            train_errors.append(loss)
            self.backwarding(learning_rate)
            self.forwarding(x_test)
            self.layers[-1].t=t_test
            loss = CE(self.layers[-1].y, self.layers[-1].t)
            test_errors.append(loss)
            self.accuracy(x_train, t_train)
        return train_errors, test_errors


#NN 실행
NN = NeuralNetwork()
NN.layers.append(Linear(784, 128))
NN.layers.append(Relu())
NN.layers.append(Linear(128,84))
NN.layers.append(Relu())
NN.layers.append(Linear(84,10))
NN.layers.append(Softmax())

t_train_one=one_hot_encoding(t_train)
t_test_one=one_hot_encoding(t_test)
train_errors, test_errors=NN.train(x_train, t_train_one, x_test, t_test_one, 100)


#Loss graph
plt.plot(train_errors, color='red', linestyle='-', label='train')  # 선 그래프 그리기
plt.plot(test_errors, color='blue', linestyle='-', label='test')  # 선 그래프 그리기
plt.legend()
plt.title('Loss Graph')  # 그래프 제목 설정
plt.xlabel('Epoch')  # x 축 레이블 설정
plt.ylabel('Loss')  # y 축 레이블 설정


#confusion matrix
y_pred=NN.forwarding(x_test)
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


# In[ ]:




