import torch
import torch.nn as nn
import timm

class LightweightMultiModalDETR(nn.Module):
    def __init__(self, config):
        super().__init__()
        model_cfg, data_cfg, train_cfg = config['model'], config['data'], config['training']
        self.train_cfg = train_cfg
        
        self.backbone = timm.create_model(
            model_cfg['backbone'], pretrained=model_cfg['pretrained'], features_only=True
        )
        backbone_output_dim = self.backbone.feature_info[-1]['num_chs']
        
        # --- THIS IS THE CRITICAL FIX ---
        # This layer adapts the features from EfficientNet to the transformer's expected size.
        self.input_proj = nn.Conv2d(backbone_output_dim, model_cfg['transformer_dim'], kernel_size=1)
        
        self.query_embed = nn.Embedding(train_cfg['num_queries'], model_cfg['transformer_dim'])
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_cfg['transformer_dim'], nhead=model_cfg['transformer_nhead'], 
            dim_feedforward=model_cfg['transformer_dim']*4, dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=model_cfg['transformer_num_layers'])
        
        self.class_embed = nn.Linear(model_cfg['transformer_dim'], len(data_cfg['class_names']) + 1)
        self.bbox_embed = nn.Linear(model_cfg['transformer_dim'], 4)

    def forward(self, image):
        features = self.backbone(image)[-1]
        proj_features = self.input_proj(features)
        
        bs, _, h, w = proj_features.shape
        flat_features = proj_features.flatten(2).permute(0, 2, 1)
        queries = self.query_embed.weight.unsqueeze(0).repeat(bs, 1, 1)
        
        # We now feed the projected features into the transformer
        transformer_input = torch.cat([queries, flat_features], dim=1)
        transformer_output = self.transformer(transformer_input)
        query_output = transformer_output[:, :self.train_cfg['num_queries'], :]
        
        pred_logits = self.class_embed(query_output)
        pred_boxes = self.bbox_embed(query_output).sigmoid()
        
        return {"pred_logits": pred_logits, "pred_boxes": pred_boxes}