import torch
import torch.nn as nn
import numpy as np
import os
import random
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
from torchvision.models import resnet50

# ----------- CBAM & Model -----------
class GAM_Attention(nn.Module):
    def __init__(self, in_channels, out_channels, rate=4):
        super(GAM_Attention, self).__init__()
        # Channel attention
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // rate, kernel_size=1),
            nn.BatchNorm2d(in_channels // rate),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // rate, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels)
        )
        # Spatial attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(out_channels, out_channels // rate, kernel_size=1),
            nn.BatchNorm2d(out_channels // rate),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // rate, 1, kernel_size=1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Channel attention
        x_channel = self.channel_attention(x)
        x_channel = self.sigmoid(x_channel)
        x = x * x_channel

        # Spatial attention
        x_spatial = self.spatial_attention(x)
        x_spatial = self.sigmoid(x_spatial)
        x = x * x_spatial

        return x


class PositionRotationGAMModel(nn.Module):
    def __init__(self, num_pieces=12, num_rot=4):
        super().__init__()
        self.resnet = resnet50(pretrained=True)
        self.resnet.fc = nn.Identity()
        self.gam = GAM_Attention(in_channels=256, out_channels=256)
        self.gam1 = GAM_Attention(in_channels=512, out_channels=512)
        self.gam2 = GAM_Attention(in_channels=1024, out_channels=1024)
        self.gam3 = GAM_Attention(in_channels=2048, out_channels=2048)
        self.layer1 = nn.Sequential(*list(self.resnet.children())[:5])
        self.layer2 = self.resnet.layer2
        self.layer3 = self.resnet.layer3
        self.layer4 = self.resnet.layer4
        self.avgpool = self.resnet.avgpool
        hidden_dim = 2048

        self.position_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_pieces)
        )
        self.rotation_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_rot)
        )

    def forward(self, images):
        B, N, C, H, W = images.shape
        images_reshaped = images.view(B * N, C, H, W)
        x = self.layer1(images_reshaped)
        x = self.gam(x)
        x = self.layer2(x)
        x = self.gam1(x)
        x = self.layer3(x)
        x = self.gam2(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        feats = x.view(B, N, -1)
        pos_logits = self.position_head(feats)
        rot_logits = self.rotation_head(feats)
        return pos_logits, rot_logits, feats


# ----------- Dataset -----------
class PuzzlePiecesDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        for sample_name in os.listdir(root_dir):
            sample_path = os.path.join(root_dir, sample_name)
            if not os.path.isdir(sample_path):
                continue
            piece_paths, piece_positions, piece_rotations = [], [], []
            for fname in os.listdir(sample_path):
                if fname.startswith("piece_") and fname.endswith(".png"):
                    parts = fname.split("_")
                    pos = int(parts[1])
                    rot = int(parts[2].split(".")[0].replace("rot", ""))
                    piece_paths.append(os.path.join(sample_path, fname))
                    piece_positions.append(pos)
                    piece_rotations.append(rot)
            self.samples.append((piece_paths, piece_positions, piece_rotations))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        piece_paths, piece_positions, piece_rotations = self.samples[idx]
        N = len(piece_paths)
        indices = list(range(N))
        random.shuffle(indices)
        images = []
        shuffled_positions = []
        shuffled_rotations = []
        for i in indices:
            img = Image.open(piece_paths[i]).convert('RGB')
            if self.transform:
                img = self.transform(img)
            images.append(img)
            shuffled_positions.append(piece_positions[i])
            shuffled_rotations.append(piece_rotations[i])
        images = torch.stack(images, dim=0)
        positions = torch.tensor(shuffled_positions, dtype=torch.long)
        rotations = torch.tensor(shuffled_rotations, dtype=torch.long)
        return images, positions, rotations

# ----------- Losses -----------
def pairwise_spatial_loss(feats, positions, grid_shape=(3, 4), margin=1.0):
    B, N, D = feats.shape
    loss = 0.0
    for b in range(B):
        for i in range(N):
            for j in range(i + 1, N):
                dist = torch.norm(feats[b, i] - feats[b, j], p=2)
                pos_i, pos_j = positions[b, i].item(), positions[b, j].item()
                row_i, col_i = divmod(pos_i, grid_shape[1])
                row_j, col_j = divmod(pos_j, grid_shape[1])
                grid_dist = abs(row_i - row_j) + abs(col_i - col_j)
                is_adjacent = grid_dist == 1
                if is_adjacent:
                    loss += dist.pow(2)
                else:
                    loss += torch.max(margin - dist, torch.tensor(0.0, device=feats.device)).pow(2)
    return loss / (B * N * (N - 1) / 2)

def pos_to_coords(positions, grid_shape):
    rows = positions // grid_shape[1]
    cols = positions % grid_shape[1]
    return rows, cols

def spatial_distance_loss(pos_logits, targets, grid_shape=(3, 4)):
    batch_size = pos_logits.size(0)
    preds = torch.argmax(pos_logits, dim=-1)
    true_rows, true_cols = pos_to_coords(targets, grid_shape)
    pred_rows, pred_cols = pos_to_coords(preds, grid_shape)
    dist = torch.sqrt((true_rows.float() - pred_rows.float()) ** 2 + (true_cols.float() - pred_cols.float()) ** 2)
    return dist.mean()


def infer_with_base_piece(pos_logits, rot_logits, base_pos=0):
    """
    pos_logits: (N, num_pieces) logits for each piece
    rot_logits: (N, num_rot) logits for each piece
    base_pos: int

    """
    N = pos_logits.shape[0]
    num_pieces = pos_logits.shape[1]
    best_score = -float('inf')
    best_positions = None
    best_rotations = None

    for base_idx in range(N):
        all_indices = np.arange(N)
        remain_piece_indices = np.delete(all_indices, base_idx)
        remain_pos_indices = np.delete(all_indices, base_pos)
        cost_matrix = -pos_logits[remain_piece_indices][:, remain_pos_indices]
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        pred_positions = np.zeros(N, dtype=int)
        pred_positions[base_idx] = base_pos
        for i, j in zip(row_ind, col_ind):
            pred_positions[remain_piece_indices[i]] = remain_pos_indices[j]
        pred_rot = rot_logits.argmax(axis=-1)
        score = 0
        for i in range(N):
            score += pos_logits[i, pred_positions[i]]
        if score > best_score:
            best_score = score
            best_positions = pred_positions.copy()
            best_rotations = pred_rot.copy()
    return best_positions, best_rotations, best_score

# ----------- 保存预测 -----------
def save_predictions(epoch, preds, targets):
    os.makedirs('./result', exist_ok=True)
    filename = f"./result/epoch_{epoch}_results.txt"
    with open(filename, 'w') as f:
        f.write("Predictions vs Ground Truth (Position) - One Sample Per Line\n")
        f.write("Format: Predicted Values | Ground Truth Values\n")
        f.write("----------------------------------------------------\n")
        sample_idx = 0
        for pred_seq, true_seq in zip(preds, targets):
            pred_batch = pred_seq.cpu().numpy()  # Shape: (batch_size, ...)
            true_batch = true_seq.cpu().numpy()
            batch_size = pred_batch.shape[0]
            for b in range(batch_size):
                pred_str = ' '.join(map(str, pred_batch[b].flatten()))
                true_str = ' '.join(map(str, true_batch[b].flatten()))
                f.write(f"Sample {sample_idx + 1}: Pred: {pred_str} | True: {true_str}\n")
                sample_idx += 1
        f.write("----------------------------------------------------\n")
    print(f"Prediction results for epoch {epoch} saved to {filename}")

# ----------- Main -----------
if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_pieces = 9
    num_rot = 4
    model = PositionRotationGAMModel(num_pieces=num_pieces, num_rot=num_rot).to(device)
    print(model)
    train_dataset = PuzzlePiecesDataset(root_dir='data/3x3_miniimagenet_color/train', transform=transform)
    total_size = len(train_dataset)
    print(f"Total train dataset size: {total_size}")
    test_dataset = PuzzlePiecesDataset(root_dir='data/3x3_miniimagenet_color/test', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4)


    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    pos_loss_fn = nn.CrossEntropyLoss()
    rot_loss_fn = nn.CrossEntropyLoss()
    lambda_pairwise = 0.05
    grid_shape = (3, 3)

    best_pos_acc = 0.0
    num_epochs = 30
    for epoch in range(num_epochs):
        model.train()
        total_loss, total_correct_pos, total_correct_rot, total_samples = 0, 0, 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")
        for images, positions, rotations in pbar:
            images, positions, rotations = images.to(device), positions.to(device), rotations.to(device)
            pos_logits, rot_logits, feats = model(images)
            B, N, _ = pos_logits.shape
            pos_ce_loss = pos_loss_fn(pos_logits.view(B * N, -1), positions.view(-1))
            pos_dist_loss = spatial_distance_loss(pos_logits.view(B * N, -1), positions.view(-1), grid_shape=grid_shape)
            pos_loss = pos_ce_loss + 0.4 * pos_dist_loss
            rot_loss = rot_loss_fn(rot_logits.view(B * N, -1), rotations.view(-1))
            pairwise_loss = pairwise_spatial_loss(feats, positions)
            loss = pos_loss + rot_loss + lambda_pairwise * pairwise_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * B * N
            total_correct_pos += (pos_logits.argmax(dim=-1) == positions).sum().item()
            total_correct_rot += (rot_logits.argmax(dim=-1) == rotations).sum().item()
            total_samples += B * N
            avg_loss = total_loss / total_samples
            pos_acc = total_correct_pos / total_samples
            rot_acc = total_correct_rot / total_samples
            pbar.set_postfix(loss=avg_loss, pos_acc=pos_acc, rot_acc=rot_acc)
        scheduler.step()

        model.eval()
        total_correct_pos, total_correct_rot, total_samples = 0, 0, 0
        all_preds, all_targets = [], []
        with torch.no_grad():
            for images, positions, rotations in test_loader:
                images, positions, rotations = images.to(device), positions.to(device), rotations.to(device)
                pos_logits, rot_logits, _ = model(images)
                B, N, _ = pos_logits.shape
                for b in range(B):
                    pos_logits_np = pos_logits[b].cpu().numpy()
                    rot_logits_np = rot_logits[b].cpu().numpy()
                    best_positions, best_rotations, best_score = infer_with_base_piece(
                        pos_logits_np, rot_logits_np, base_pos=0
                    )

                    gt_positions = positions[b].cpu().numpy()
                    gt_rotations = rotations[b].cpu().numpy()
                    total_correct_pos += (best_positions == gt_positions).sum()
                    total_correct_rot += (best_rotations == gt_rotations).sum()
                    total_samples += N

                    all_preds.append(torch.tensor(best_positions).unsqueeze(0))
                    all_targets.append(positions[b].unsqueeze(0))

        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        pos_acc = total_correct_pos / total_samples
        rot_acc = total_correct_rot / total_samples
        print(f"Test Position Acc: {pos_acc:.4f} | Test Rotation Acc: {rot_acc:.4f}")
        save_predictions(epoch, [all_preds], [all_targets])
        if pos_acc > best_pos_acc:
            best_pos_acc = pos_acc
            os.makedirs('./model', exist_ok=True)
            torch.save(model.state_dict(), './model/position_rotation_best.pth')
            print(f"Best model saved at epoch {epoch+1} with pos_acc {best_pos_acc:.4f}")

    print("Finished training")
