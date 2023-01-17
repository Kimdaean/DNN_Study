import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.init
import torch.nn as nn
import torch.optim as optim

#gpu 설정
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
#cuda가 사용가능하면 cuda를 쓰고 사용할 수 없다면 cpu를 쓰겠다

torch.manual_seed(777)
#랜덤 밸류를 고정해주는 함수 -> 시드가 무작위로 변환다면 학습 모델의 결과를 확인할 때 무엇이 문제인지 파악하기 어려워진다

if device=='cuda':
     torch.cuda.manual_seed_all(777)

learning_rate = 0.001 # 1e-3, 1e-4
training_epochs = 15
batch_size = 100

#MNIST dataset
mnist_train=dsets.MNIST(root="MNIST_data/",
                        train=True,
                        transform=transforms.ToTensor(), #input 데이터를 tensor로 변환
                        download=True) #데이터를 다운로드해서 쓰겠다

mnist_test=dsets.MNIST(root="MNIST_data/",
                        train=False, #test이므로 False로 해줌
                        transform=transforms.ToTensor(), #input 데이터를 tensor로 변환
                        download=True) #데이터를 다운로드해서 쓰겠다

data_loader = torch.utils.data.DataLoader(dataset=mnist_train,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          drop_last=True)

#모델 생성
class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        # L1 ImgIn shape=(?, 28, 28, 1)
        #    Conv     -> (?, 28, 28, 32)
        #    Pool     -> (?, 14, 14, 32)
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
        # L2 ImgIn shape=(?, 14, 14, 32)
        #    Conv      ->(?, 14, 14, 64)
        #    Pool      ->(?, 7, 7, 64)
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
        # Final FC 7x7x64 inputs -> 10 outputs
        self.fc = torch.nn.Linear(7 * 7 * 64, 10, bias=True)
        torch.nn.init.xavier_uniform_(self.fc.weight) #초기화

    def forward(self,x):
        out=self.layer1(x) #첫번째 레이어를 통과한 값을 out으로 받고
        out=self.layer2(out) #한번 통과한 값을 두번째 레이어에 다시 통과
        out=out.view(out.size(0),-1)
        out=self.fc(out)
        return out

model = CNN().to(device)

criterion=nn.CrossEntropyLoss().to(device)
optimizer=optim.Adam(model.parameters(), lr=1e-3, betas=(0.5, 0.999)) #betas: 기울기와 그 제곱의 평균을 계산하는 데 사용되는 계수(기본값: (0.9, 0.999))

#Traing Code
total_batch=len(data_loader) #batchsize 구하는 코드
for i in range(training_epochs):
    avg_cost=0
    for x,y in data_loader:
        x=x.to(device) #input data
        y=y.to(device) #label data

        optimizer.zero_grad() #이거를 꼭 넣어줘야 학습이 됨 *
        hypothesis=model(x)
        cost=criterion(hypothesis,y)
        cost.backward() #*
        optimizer.step() #*

        avg_cost+=cost/total_batch

    print(f"횟수: {i+1}\tCost={avg_cost}")

#Test Code
with torch.no_grad():
    X_test = mnist_test.test_data.view(len(mnist_test), 1, 28, 28).float().to(device)
    Y_test = mnist_test.test_labels.to(device)

    prediction = model(X_test)
    correct_prediction = torch.argmax(prediction, 1) == Y_test
    accuracy = correct_prediction.float().mean()
    print('Accuracy:', accuracy.item())
    