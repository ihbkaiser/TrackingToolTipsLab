import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from data_loader import EndoVIS15Dataset
from torch.utils.data import Dataset, DataLoader
from modules import *
import time
import cv2
class IHBNet(nn.Module):
    def __init__(self):
        super(IHBNet, self).__init__()
        # Use pretrained ViT from timm
        self.encoder = timm.create_model("vit_base_patch16_224", pretrained=True)
        self.embed_dim = self.encoder.head.in_features  
        
        self.spatial_module = SpatialModule()
        
        # Positional Encoding
        self.pe = pe_layer(num_pos_feats=self.embed_dim // 2, scale=1.0)
        pe_embed = self.pe((14, 14))  # shape: (C, 14, 14)
        self.register_buffer('pe_embed', pe_embed)
        
        self.gat_layer = GATLayerIHB(self.embed_dim, self.embed_dim, num_of_heads=4, concat=True)
        self.gat_proj = nn.Linear(4 * self.embed_dim, self.embed_dim)
        self.head = nn.Conv2d(self.embed_dim, 1, 1)  # Output 1-channel logits

    def forward(self, x):
        b, c, h, w = x.shape
        
        # Extract features using ViT
        y = self.encoder.forward_features(x)  # (b, 197, embed_dim)
        y = y + self.encoder.pos_embed[:, 0:y.size(1), :]  # Align positional embeddings
        
        # Transformer normalization
        y = self.encoder.norm(y)  # (b, 197, embed_dim)
        
        cls_out = y[:, 0, :]       # (b, embed_dim)
        patch_tokens = y[:, 1:, :]  # (b, 196, embed_dim)

        # Reshape tokens to grid: (b, embed_dim, 14, 14)
        patch_tokens = patch_tokens.transpose(1, 2).view(b, self.embed_dim, 14, 14)

        # Apply spatial attention
        patch_tokens = self.spatial_module(patch_tokens)
        patch_tokens = patch_tokens + self.pe_embed  # (b, embed_dim, 14, 14)

        # Convert feature map into graph
        y_graph = image_to_graph(patch_tokens)  # (b, 196, embed_dim)
        edge_index = build_edge_index(14, 14)

        # Apply GATLayerIHB to each batch
        out_list = []
        for i in range(b):
            node_features = y_graph[i]  # (196, embed_dim)
            out_nodes, _ = self.gat_layer((node_features, edge_index))  # Ensure correct output
            out_list.append(out_nodes)
        
        out_graph = torch.stack(out_list, dim=0)  # (b, 196, embed_dim)
        out_graph = self.gat_proj(out_graph)

        # Reshape back to (b, embed_dim, 14, 14)
        out_feat = out_graph.permute(0, 2, 1).view(b, self.embed_dim, 14, 14)
        out = self.head(out_feat)  # Logits output (b, 1, 14, 14)
        
        return out, cls_out  # No sigmoid here!
def save_image(tensor, filename, save_dir="train_test"):
    """ Lưu tensor thành ảnh PNG """
    os.makedirs(save_dir, exist_ok=True)
    image = tensor.detach().cpu().numpy()[0, 0]  # Lấy ảnh đầu tiên trong batch, channel 0
    image = (image * 255).astype(np.uint8)  # Chuyển sang định dạng 0-255
    save_path = os.path.join(save_dir, filename)
    cv2.imwrite(save_path, image)
    print(f"✅ Saved {filename} at {save_path}")
def train_step(model, images, skeletons, optimizer, criterion, icon = 0):
    model.train()
    optimizer.zero_grad()
    
    pred, _ = model(images)  # Get logits (not probabilities)
    
    # Resize prediction to match skeletons
    pred_resized = F.interpolate(pred, size=skeletons.shape[2:], mode='bilinear', align_corners=False)
    skeletons = skeletons/255.0

    save_image(torch.sigmoid(pred_resized), f"pred_{icon}.png")
    save_image(skeletons, f"skeletons_{icon}.png")

    # Compute loss with BCEWithLogitsLoss
    loss = criterion(pred_resized, skeletons)
    
    loss.backward()
    optimizer.step()
    
    return loss.item()
from tqdm import tqdm  # Import tqdm để hiển thị tiến trình
import torch
import os
from tqdm import tqdm

def save_checkpoint(model, optimizer, epoch, loss, best_loss, save_dir="checkpoints_2"):
    """Lưu trọng số mô hình: last epoch và best epoch"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Lưu model của last epoch
    last_checkpoint_path = os.path.join(save_dir, "IHBNet_last.pth")
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, last_checkpoint_path)
    print(f"✅ Last model saved at {last_checkpoint_path}")

    # Nếu loss nhỏ nhất, lưu best model
    if loss < best_loss:
        best_checkpoint_path = os.path.join(save_dir, "IHBNet_best.pth")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, best_checkpoint_path)
        print(f"🏆 Best model saved at {best_checkpoint_path}")
        return loss  # Cập nhật best loss
    return best_loss

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = EndoVIS15Dataset(dataset_id=2)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)

    model = IHBNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    best_loss = float("inf")  # Khởi tạo best loss
    epochs = 50
    icon = 0
    for epoch in range(epochs):
        batch_losses = []
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}/{epochs}")

        for batch_idx, (images, _, skeletons) in progress_bar:
            images = images.to(device)
            skeletons = skeletons.to(device)
            loss = train_step(model, images, skeletons, optimizer, criterion, icon)
            icon = icon + 1
            batch_losses.append(loss)

            # Cập nhật tqdm với loss hiện tại
            progress_bar.set_postfix({"Batch Loss": f"{loss:.4f}"})

        avg_loss = sum(batch_losses) / len(batch_losses)
        print(f"\nEpoch {epoch+1}/{epochs} - Avg Loss: {avg_loss:.4f}")

        # 🔥 Lưu last epoch & best epoch
        best_loss = save_checkpoint(model, optimizer, epoch, avg_loss, best_loss)

if __name__ == "__main__":
    main()
