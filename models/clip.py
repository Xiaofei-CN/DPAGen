from transformers import CLIPVisionModelWithProjection
import torch.nn as nn
import torch
class CLIP(nn.Module):
    def __init__(self, pretrained_path):
        super().__init__()
        model = CLIPVisionModelWithProjection.from_pretrained(pretrained_path)
        model.eval()
        model.requires_grad_(False)
        self.model = model
    @torch.no_grad()
    def encode_image(self, image):
        last_hidden_states = self.model(image).last_hidden_state
        last_hidden_states_norm = self.model.vision_model.post_layernorm(last_hidden_states)
        return last_hidden_states_norm

# clip = CLIP("/home/xtf/AAAI25/NewCFLD/pretrained_models/image_encoder")
# imgae = torch.zeros((2,3,224,224))
# print(clip(imgae))
