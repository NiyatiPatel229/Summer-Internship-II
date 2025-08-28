# --- Task 1: Explain usefulness of random initialization ---
# Random initialization helps neural networks escape symmetry and ensures that each neuron learns something different, kickstarting the learning process.

# --- Task 2: Define train_transform pipeline ---
train_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

# --- Task 3: Define val_transform pipeline ---
val_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# --- Task 4: Create val_loader ---
val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# --- Task 5: Purpose of tqdm ---
# tqdm is used to provide a smart progress bar for loops.

# --- Task 6: Why reset train_loss/train_correct/train_total every epoch? ---
# These stats are epoch-specific; resetting them tracks accuracy/loss per epoch rather than accumulating across all epochs.

# --- Task 7: Why use torch.no_grad() in validation loop? ---
# torch.no_grad() disables gradient computations, saving memory and computation since no backpropagation occurs in validation.

# --- Task 8: Two metrics for training performance ---
# Accuracy and loss.

# --- Task 9: Plot model training loss ---
plt.plot(train_loss_history)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()

# --- Task 10: Retrieve all_preds/all_labels from val_loader ---
all_preds, all_labels = [], []
model.eval()
with torch.no_grad():
    for data, target in val_loader:
        output = model(data)
        preds = output.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(target.cpu().numpy())

