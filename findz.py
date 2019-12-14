import torch
from torchvision import datasets
from torchvision.transforms import transforms
from models.real_nvp import RealNVP


def main():
    dataroot='data/CelebA'
    DS=datasets.CelebA
    reqtrans=transforms.Compose([transforms.Resize(64),transforms.CenterCrop(64),transforms.ToTensor()])


    dataset = DS(dataroot,download=True,transform=reqtrans)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=256,
        shuffle=False,
        num_workers=4
    )
    print(len(dataloader))

    generator=RealNVP(num_scales=2, in_channels=3, mid_channels=64, num_blocks=8)
    generator.load_state_dict(torch.load('./savedmodels/{}/generator'.format('celeba')))
    generator=generator.to('cuda')

    with torch.no_grad():
        for i, (imgs, labels) in enumerate(dataloader):
            imgs,labels=imgs.to('cuda'),labels.to('cuda')    
            n=imgs.shape[0]
            z,_=generator(imgs)

            for j in range(n):
                for l in range(40):
                    if labels[j][l]==1:
                        numeachcat[l][1]+=1
                        zavg[l][1]+=z[j].to('cpu')
                    else:
                        numeachcat[l][0]+=1
                        zavg[l][0]+=z[j].to('cpu')

            if i%100==0:
                print(i)


    print(numeachcat)

if __name__ == '__main__':
    numeachcat=[[0,0] for i in range(40)]
    zavg=[[torch.zeros(3,64,64,requires_grad=False) for i in range(2)] for j in range(40)]
    main()
    torch.save(zavg,'./savedzvector/zavg.pth')