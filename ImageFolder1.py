import matplotlib
import torchvision
from torchvision import transforms

from torch.utils.data import DataLoader
from matplotlib.pyplot import imshow

trans = transforms.Compose([
    transforms.Resize((64,128)) #사진의 크기 조정
])

train_data = torchvision.datasets.ImageFolder(root='custom_data/origin_data', transform=trans)
#ImageFolder를 사용해서 폴더에서 이미지를 가져옴 root를 사용해서 경로를 적고, transform을 통해 변환

for num, value in enumerate(train_data):
    data, label = value
    print(num, data, label)

    if (label == 0): #label은 폴더의 순서를 표현하는거 같음. 영상에서는 첫번째 폴더가 gray였음
        data.save('custom_data/train_data/gray/%d_%d.jpeg' % (num, label)) #데이터를 생성하고 폴더에 저장하는거까지
    else:
        data.save('custom_data/train_data/red/%d_%d.jpeg' % (num, label))