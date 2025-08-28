# --- Task 1: Define custom_transform pipeline ---
import torchvision.transforms as transforms

custom_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomRotation(45),
    transforms.ToTensor()
])

# --- Task 2: Load dataset with ImageFolder ---
from torchvision import datasets
dataset = datasets.ImageFolder('./images_dataSAT/', transform=custom_transform)

# --- Task 3: Print class names and indices ---
print("Classes:", dataset.classes)
print("Class-to-index:", dataset.class_to_idx)

# --- Task 4: Retrieve and display image shapes from batch in loader ---
from torch.utils.data import DataLoader
loader = DataLoader(dataset, batch_size=8, shuffle=True)
images, labels = next(iter(loader))
print("Batch image shape:", images.shape)  # Should be [batch, c, h, w]

# --- Task 5: Display images in custom loader batch ---
import matplotlib.pyplot as plt
imgs = images.permute(0, 2, 3, 1).numpy()  # change from [B,C,H,W] to [B,H,W,C]
for i in range(8):
    plt.subplot(2, 4, i+1)
    plt.imshow(imgs[i])
    plt.axis('off')
plt.show()

