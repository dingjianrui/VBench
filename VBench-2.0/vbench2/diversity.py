import torch
from torchvision.models import vgg19
from torch import nn
from torchvision import transforms
from collections import OrderedDict
import cv2
import numpy as np
import os
import json
from PIL import Image
from vbench2.utils import load_dimension_info
from tqdm import tqdm

STYLE_LAYER_INDICES = {0, 5, 10, 19, 28}
CONTENT_LAYER_INDEX = 30
ALL_LAYER_INDICES = STYLE_LAYER_INDICES | {CONTENT_LAYER_INDEX}


class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.features = vgg19(pretrained=True).features.eval()

    def forward(self, x):
        features = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in ALL_LAYER_INDICES:
                features.append(x.detach().cpu())
        return features


def gram_matrix(tensor):
    batch_size, channels, height, width = tensor.shape
    features = tensor.view(batch_size, channels, -1)
    gram = torch.bmm(features, features.transpose(1,2))
    gram = gram / (channels * height * width)
    return gram

def content_loss(content, target_content):
    return torch.mean(torch.abs(content - target_content))

def style_loss_from_grams(gram_a, gram_b):
    return torch.mean(torch.abs(gram_a - gram_b))

def evaluate(style_grams, content_features):
    content_diversity = 0
    style_diversity = 0
    len_seed = len(content_features)
    for i in range(len_seed):
        for j in range(i+1, len_seed):
            content_diversity += content_loss(content_features[i], content_features[j])
            for k in range(5):
                style_diversity += style_loss_from_grams(style_grams[i][k], style_grams[j][k])
    content_diversity/=(0.5*len_seed*(len_seed-1))
    style_diversity/=(2.5*len_seed*(len_seed-1))
    diversity=(content_diversity+1000*style_diversity)/2
    return content_diversity, 1000*style_diversity, diversity / 17.712 # Empirical maximum


def _read_frames_batched(video_path, batch_size):
    """Yield batches of preprocessed frames from a video without loading all into memory."""
    preprocess = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    video = cv2.VideoCapture(video_path)
    batch = []
    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
        image_np = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.uint8)
        image_pil = Image.fromarray(image_np)
        batch.append(preprocess(image_pil))
        if len(batch) == batch_size:
            yield torch.stack(batch)
            batch = []
    video.release()
    if batch:
        yield torch.stack(batch)


def Diversity(prompt_dict_ls, model, device, batch_size=32):
    final_score=0
    processed_json=[]
    for prompt_dict in tqdm(prompt_dict_ls):
        video_paths = prompt_dict['video_list']
        style_grams=[]
        content_features=[]
        for video_path in video_paths:
            # Accumulate precomputed gram matrices (small) for style layers,
            # and raw features only for the content layer (small spatial size).
            gram_chunks = [[] for _ in range(5)]
            content_chunks = []
            for batch in _read_frames_batched(video_path, batch_size):
                batch = batch.to(device)
                with torch.no_grad():
                    features = model(batch)
                # Style layers (indices 0-4): compute gram immediately to save memory
                for k in range(5):
                    gram_chunks[k].append(gram_matrix(features[k]))
                # Content layer (index 5): small (512x32x32), keep raw
                content_chunks.append(features[5])
                del batch, features
                torch.cuda.empty_cache()
            style_grams.append([torch.cat(gram_chunks[k], dim=0) for k in range(5)])
            content_features.append(torch.cat(content_chunks, dim=0))
            del gram_chunks, content_chunks

        content_diversity, style_diversity, diversity=evaluate(style_grams, content_features)
        diversity = torch.clamp(diversity, min=0, max=1)
        new_item={
                'video_path':video_paths[0],
                'video_results':diversity.tolist()
            }
        processed_json.append(new_item)
        final_score+=diversity
        del style_grams, content_features
    return final_score/len(prompt_dict_ls), processed_json

def compute_diversity(json_dir, device, submodules_dict, **kwargs):
    _, prompt_dict_ls = load_dimension_info(json_dir, dimension='diversity', lang='en')
    model = VGG().to(device)
    
    all_results, video_results = Diversity(prompt_dict_ls, model, device)
    all_results = sum([d['video_results'] for d in video_results]) / len(video_results)
    return all_results, video_results