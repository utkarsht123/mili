import torch
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
import yaml

from modules.data import AUAIR_RGB_Dataset, HITUAV_Thermal_YOLO_Dataset
from modules.models import LightweightMultiModalDETR
from modules.loss import HungarianMatcher, SetCriterion
from modules.evaluation import evaluate

def collate_fn(batch):
    images, targets = zip(*batch)
    images = torch.stack(images, 0)
    return images, list(targets)

def main():
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device(config['environment']['device'])

    au_air_dataset = AUAIR_RGB_Dataset(config, split='train')
    hit_uav_train_dataset = HITUAV_Thermal_YOLO_Dataset(config, split='train')
    
    combined_dataset = ConcatDataset([au_air_dataset, hit_uav_train_dataset])
    print(f"✅ Combined dataset created with {len(combined_dataset)} total images.")
    
    val_dataset = HITUAV_Thermal_YOLO_Dataset(config, split='val')
    train_loader = DataLoader(combined_dataset, batch_size=config['training']['batch_size'], shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], collate_fn=collate_fn)
    
    model = LightweightMultiModalDETR(config).to(device)
    matcher = HungarianMatcher(cost_class=1, cost_bbox=5, cost_giou=2)
    criterion = SetCriterion(num_classes=len(config['data']['class_names']), matcher=matcher, weight_dict=config['loss_coefficients'], eos_coef=0.1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config['training']['learning_rate'], weight_decay=config['training']['weight_decay'])
    
    print("🚀 Starting training on combined AU-AIR and HIT-UAV datasets...")
    best_val_loss = float('inf')
    for epoch in range(config['training']['epochs']):
        model.train()
        for i, (images, targets) in enumerate(train_loader):
            # --- THIS IS THE FIX ---
            # Check if there are any annotations in the current batch.
            # The `any()` function checks if at least one target has boxes.
            if not any(t['labels'].numel() > 0 for t in targets):
                continue # Skip this batch if it's empty

            images = images.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            outputs = model(images)
            losses_dict = criterion(outputs, targets)
            weighted_loss = sum(losses_dict[k] * criterion.weight_dict[k.replace('loss_','')] for k in losses_dict.keys())
            loss = weighted_loss / config['training']['gradient_accumulation_steps']
            loss.backward()
            if (i+1) % config['training']['gradient_accumulation_steps'] == 0:
                optimizer.step()
                optimizer.zero_grad()
        
        avg_val_loss, precision, recall = evaluate(model, criterion, val_loader, device)
        print(
            f"✅ Epoch {epoch+1}/{config['training']['epochs']} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Precision: {precision:.4f} | "
            f"Recall: {recall:.4f}"
        )
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), config['demo']['model_path'])
            print(f"   -> 🎉 New best model saved to {config['demo']['model_path']}")
            
    print("\n📦 Training complete.")

if __name__ == '__main__':
    main()