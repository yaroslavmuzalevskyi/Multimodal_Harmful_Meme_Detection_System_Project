

from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModel

from config import (
    TEXT_MODEL_NAME,
    VISION_MODEL_NAME,
    DEVICE,
    FIM_HIDDEN_DIM,
    CLASSIFIER_HIDDEN_DIM,
    FREEZE_BASE_MODELS,
)


class MultimodalHatefulMemeModel(nn.Module):
    def __init__(
        self,
        text_model_name: str = TEXT_MODEL_NAME,
        vision_model_name: str = VISION_MODEL_NAME,
        fim_hidden_dim: int = FIM_HIDDEN_DIM,
        classifier_hidden_dim: int = CLASSIFIER_HIDDEN_DIM,
        freeze_base_models: bool = FREEZE_BASE_MODELS,
    ):
        super().__init__()

        self.text_encoder = AutoModel.from_pretrained(text_model_name)

        self.vision_encoder = AutoModel.from_pretrained(vision_model_name)

        if freeze_base_models:
            for param in self.text_encoder.parameters():
                param.requires_grad = False
            for param in self.vision_encoder.parameters():
                param.requires_grad = False

        text_hidden_size = self.text_encoder.config.hidden_size
        vision_hidden_size = self.vision_encoder.config.hidden_size

        self.text_proj = nn.Linear(text_hidden_size, fim_hidden_dim)
        self.image_proj = nn.Linear(vision_hidden_size, fim_hidden_dim)

        # Learnable fusion block that ingests the statistics from the
        # factorized interaction tensor; defined here so it is trained
        # and persisted with checkpoints.
        self.fim_mlp = nn.Sequential(
            nn.Linear(2, fim_hidden_dim),
            nn.ReLU(),
            nn.Linear(fim_hidden_dim, 2 * fim_hidden_dim),
            nn.ReLU(),
        )

        classifier_input_dim = (
            text_hidden_size       
            + vision_hidden_size    
            + 2 * fim_hidden_dim    
        )

        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, classifier_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(classifier_hidden_dim, 1),  
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
    ) -> torch.Tensor:

        text_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        text_last_hidden = text_outputs.last_hidden_state  
        text_cls = text_last_hidden[:, 0, :]            

        vision_outputs = self.vision_encoder(
            pixel_values=pixel_values,
        )
        vision_last_hidden = vision_outputs.last_hidden_state  
        image_cls = vision_last_hidden[:, 0, :]             

        text_proj = self.text_proj(text_last_hidden)      
        image_proj = self.image_proj(vision_last_hidden)  


        interaction = torch.matmul(
            text_proj, image_proj.transpose(1, 2)
        )

        fim_mean = interaction.mean(dim=(1, 2))  
        fim_max = interaction.amax(dim=(1, 2))   



        fim_concat = torch.stack([fim_mean, fim_max], dim=-1)  


        fim_features = self.fim_mlp(fim_concat)  

        multimodal_repr = torch.cat(
            [text_cls, image_cls, fim_features], dim=-1
        )  

        logits = self.classifier(multimodal_repr) 
        return logits


def build_model() -> MultimodalHatefulMemeModel:

    model = MultimodalHatefulMemeModel()
    model.to(DEVICE)
    return model
