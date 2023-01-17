import torch
import torch.optim as optim

x=torch.FloatTensor([[1],[2],[3]])
y=torch.FloatTensor([[2],[4],[6]])

w=torch.zeros(1,requires_grad=True)
b=torch.zeros(1,requires_grad=True)

optimizer=optim.SGD([w,b],lr=0.01)

for i in range(1,1001):
    h=x*w+b
    c=torch.mean((h-y)**2)
    if i%100==0:
        print(f"횟수: {i}\tw: {w.item()}\tc: {c.item()}")
    optimizer.zero_grad()
    c.backward()
    optimizer.step()