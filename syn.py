# Step 1: Import the necessary libraries
import torch
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn

class TopKLayer(nn.Module):
    def __init__(self, topk=0.1, revert=False):
        super(TopKLayer, self).__init__()
        self.revert=revert
        self.topk=topk

    def sparse_hw(self, x, tau, topk, device='cpu'):
        n, c, h, w = x.shape
        if topk == 1:
            return x
        x_reshape = x.view(n, c, h * w)
        topk_keep_num = int(max(1, topk * h * w))
        _, index = torch.topk(x_reshape.abs(), topk_keep_num, dim=2)
        if self.revert:
            # Useless
            mask = (torch.ones_like(x_reshape) - torch.zeros_like(x_reshape).scatter_(2, index, 1)).to(device)
        else:
            mask = torch.zeros_like(x_reshape).scatter_(2, index, 1).to(device)
        # print("mask percent: ", mask.mean().item())
        sparse_x = mask * x_reshape
        sparsity_x = 1.0 - torch.where(sparse_x == 0.0)[0].shape[0] / (n * c * h * w)
        # print("sparsity -- ({}): {}".format((n, c, h, w), sparsity_x)) ## around 9% decrease to 4% fired eventually this way
        if tau == 1.0:
            return sparse_x.view(n, c, h, w) 
        # print("--- tau", tau)
        tau_x = x * torch.FloatTensor([1. - tau]).to(device)
        # print("sum of x used", tau_x.sum())
        return sparse_x.view(n, c, h, w) * torch.FloatTensor([tau]).to(device) + tau_x

    def forward(self, x):
        return self.sparse_hw(x,1,self.topk)

def topK_AlexNet(pretrain_weigth,topk, **kwargs):
    if pretrain_weigth=="":
        alexnet = models.alexnet(pretrained=True)
        new_features = nn.Sequential(
            # layers up to the point of insertion
            *(list(alexnet.features.children())[:3]), 
            TopKLayer(topk),
            *(list(alexnet.features.children())[3:6]),
            TopKLayer(topk),
            *(list(alexnet.features.children())[6:8]),
            TopKLayer(topk),
            *(list(alexnet.features.children())[8:10]),
            TopKLayer(topk),
            *(list(alexnet.features.children())[10:]),
            TopKLayer(topk),
        )
        alexnet.features = new_features
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        alexnet = alexnet.to(device)
        alexnet = torch.nn.DataParallel(alexnet)
        return alexnet
# Step 2: Load the pre-trained AlexNet model
# alexnet = models.alexnet(pretrained=True)
alexnet = topK_AlexNet(pretrain_weigth = "", topk = 1)

# Step 3: Define the loss function and optimizer
criterion = torch.nn.MSELoss()
optimizer = optim.Adam(alexnet.parameters(), lr=0.01)

# Step 4: Define the image transformation process
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

import os

def synthesize_images(input_folder = '/Users/mira/Desktop/DLT/dataset2/test/3', output_folder = 'output/'):
    # Get all image files in the input folder
    image_files = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]

    for i, image_file in enumerate(image_files):
        image_path = os.path.join(input_folder, image_file)
        image = Image.open(image_path)
        image = transform(image).unsqueeze(0)
        image.requires_grad = True

        for j in range(30):
            optimizer.zero_grad()
            outputs = alexnet(image)
            loss = criterion(outputs, torch.zeros_like(outputs))
            loss.backward()
            optimizer.step()

        # Save the synthesized image with a new name in the output folder
        output_path = os.path.join(output_folder, f'o_{image_file}')
        to_pil = transforms.ToPILImage()
        synthesized_pil_image = to_pil(image.squeeze(0)).save(output_path)

    print(f'Synthesized images are saved in {output_folder}')
    return synthesized_pil_image

# Step 6: Run the imoutput_folder, lossput_folder, loss synthesis process
# synthesized_image = synthesize_images('0_244.jpg', "output/sth.jpg")
synthesize_images()
