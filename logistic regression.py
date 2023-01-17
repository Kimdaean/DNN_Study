import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

x=torch.FloatTensor([[1,2],[2,3],[3,1],[4,3],[5,3],[6,2]])
y=torch.FloatTensor([[0],[0],[0],[1],[1],[1]])

w=torch.zeros((2,1),requires_grad=True)
# matrix 계산을 생각하면 왜 (1,2)가 아니라 (2,1)인지 이해됨
b=torch.zeros(1,requires_grad=True)

op=optim.SGD([w,b],lr=1)

for i in range(1001):
    h= torch.sigmoid(x.matmul(w)+b)
    #logistic을 구분해야하므로 sigmoid를 사용 -> 이게 linear regression과 다른점

    c=F.binary_cross_entropy(h,y)
    #loss을 찾기 위한 함수도 linear regression과 다름

    op.zero_grad()
    c.backward()
    op.step()

    if i%100==0:
        print(f"{i}/1000\tCost:{c.item()}")
p= h >= torch.FloatTensor([0.5])
print(p)