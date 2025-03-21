from torchvision import datasets, transforms
import torch

def load_MNIST(filter_label=None):
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
                ])),
            batch_size=10000, shuffle=True)
    
    imgs, labels = next(iter(test_loader)) # extract imgs
    
    if filter_label is not None:
        imgs = imgs[labels==filter_label]
        labels = labels[labels==filter_label]
    
    return imgs, labels
    
# please first download imagenet val set from 
# https://academictorrents.com/browse.php?search=imagenet&sort_field=seeders&sort_dir=DESC
def imagenet_loader(batch_size=10000, path='../data/imagenet_val'):
    transform = transforms.Compose([transforms.Resize((224, 224)), 
                                    # transforms.Resize(255), 
                                    # transforms.CenterCrop(224),  
                                    transforms.ToTensor(),
                                    # transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                    #                       std=[0.229, 0.224, 0.225]
                                    #                       ),
                                    ])
    imagenet_data = datasets.ImageNet(path, split='val', transform=transform)
    data_loader = torch.utils.data.DataLoader(imagenet_data,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=0
                                              )
    return data_loader

# return an array of imagenet images and labels
def load_imagenet(num_img=10000, filter_label=None, path='../data/imagenet', processor=None):
    data_loader = imagenet_loader(batch_size=num_img, path=path)
    
    imgs, labels = next(iter(data_loader)) 
    
    if processor is not None:
        imgs = processor(imgs, return_tensors="pt")['pixel_values']
    
    if filter_label is not None:
        imgs = imgs[labels==filter_label]
        labels = labels[labels==filter_label]
        
    print(f'Number of returned imgs = {len(imgs)}')
    
    return imgs, labels
    

def remove_inaccurate_imgs(imgs, labels, model):
    '''
    Returns (imgs, labels), where model classification agree with the true label
    '''
    with torch.no_grad():
        pred = model(imgs)
        mask = labels == torch.argmax(pred, dim=1)
        print(f'For {len(mask)} imgs, {len(mask)-torch.sum(mask).item()} are removed due to wrong classifications against true labels\n')
    return imgs[mask], labels[mask]
    