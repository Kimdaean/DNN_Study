import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# 멀티 GPU중 0번 GPU에만 모든 cuda 연산을 할당

# for reproducibility
torch.manual_seed(777) #시드를 고정하여 사용하면 학습시에 몇 번을 돌려도 같은 결과가 나오는 것을 알 수 있다.
if device == 'cuda':
    torch.cuda.manual_seed_all(777) #시드를 고정하여 사용하면 학습시에 몇 번을 돌려도 같은 결과가 나오는 것을 알 수 있다.

X = torch.FloatTensor([[0, 0], [0, 1], [1, 0], [1, 1]]).to(device) #cuda 최적화된 모델로 변환
Y = torch.FloatTensor([[0], [1], [1], [0]]).to(device) #cuda 최적화된 모델로 변환

# nn layers
linear = torch.nn.Linear(2, 1, bias=True)
sigmoid = torch.nn.Sigmoid()

# model
model = torch.nn.Sequential(linear, sigmoid).to(device)

# define cost/loss & optimizer
criterion = torch.nn.BCELoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=1)

for step in range(10001):
    optimizer.zero_grad()
    hypothesis = model(X)

    # cost/loss function
    cost = criterion(hypothesis, Y)
    cost.backward()
    optimizer.step()

    if step % 1000 == 0:
        print(step, cost.item())