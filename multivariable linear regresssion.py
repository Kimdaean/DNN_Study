import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
x=torch.FloatTensor([[73,80,75],[93,88,93],[89,91,90],[96,98,100],[73,66,70]])
y=torch.FloatTensor([[152],[185],[180],[196],[142]])
w=torch.zeros((3,1),requires_grad=True)
b=torch.zeros(1,requires_grad=True)

op=optim.SGD([w,b],lr=1e-5)

for i in range(1,21):
    h = x.matmul(w) + b
    cost = F.mse_loss(h, y)
    op.zero_grad()
    cost.backward() # 기울기 0에 가까운 값을 찾는다
    op.step()
    print(f"{i}/21  hypothesis: {h.squeeze().detach()}    cost:{cost.item()}")
print(h.squeeze())

