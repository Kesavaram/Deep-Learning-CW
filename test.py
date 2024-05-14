import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
import os

class AlexNetEncoder(nn.Module):
    def __init__(self):
        super(AlexNetEncoder, self).__init__()
        alexnet = models.alexnet(pretrained=True)
        self.features = nn.Sequential(
            *(list(alexnet.features.children())[:])
        )

    def forward(self, x):
        x = self.features(x)
        return x
    
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=1, output_padding=1),  # 6x6 -> 12x12
            nn.ReLU(),
            nn.ConvTranspose2d(256, 192, kernel_size=3, stride=2, padding=1, output_padding=1),  # 12x12 -> 24x24
            nn.ReLU(),
            nn.ConvTranspose2d(192, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # 24x24 -> 48x48
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),   # 48x48 -> 96x96
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1, output_padding=1),     # 96x96 -> 192x192
            nn.ReLU(),
            nn.ConvTranspose2d(3, 3, kernel_size=4, stride=2, padding=1),                        # 192x192 -> 224x224
            nn.Sigmoid()  # Output in the range [0, 1]
        )

    def forward(self, x):
        for layer in self.deconv_layers:
            x = layer(x)
            print(x.shape)  # Print shape after each layer
        return x
    
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = AlexNetEncoder()
        self.decoder = Decoder()

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Instantiate the autoencoder
autoencoder = Autoencoder()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
autoencoder = autoencoder.to(device)
autoencoder = torch.nn.DataParallel(autoencoder)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

# Define the image transformation process
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def synthesize_images(input_folder, output_folder):
    # Get all image files in the input folder
    image_files = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]

    for image_file in image_files:
        image_path = os.path.join(input_folder, image_file)
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0).to(device)
        image.requires_grad = True

        for _ in range(30):
            optimizer.zero_grad()
            outputs = autoencoder(image)
            loss = criterion(outputs, image)
            loss.backward()
            optimizer.step()

        # Save the synthesized image with a new name in the output folder
        output_path = os.path.join(output_folder, f'o_{image_file}')
        to_pil = transforms.ToPILImage()
        synthesized_pil_image = to_pil(outputs.squeeze(0).cpu()).save(output_path)

    print(f'Synthesized images are saved in {output_folder}')

# Run the image synthesis process
input_folder = '/Users/mira/Desktop/DLT/dataset2/test/0'
output_folder = 'output/'
os.makedirs(output_folder, exist_ok=True)
synthesize_images(input_folder, output_folder)
