import cv2
import math
import torch
import xformers
import numpy as np
import comfy.model_management
from . import utils
from einops import rearrange
from torch import einsum


def set_model_patch_replace(model, prompt_embedding, region, key, width, height):
    attn2 = None
    if key[0] == "input":
        attn2 = model.model.diffusion_model.input_blocks[key[1]][1].transformer_blocks[key[2]].attn2
    elif key[0] == "middle":
        attn2 = model.model.diffusion_model.middle_block[1].transformer_blocks[key[2]].attn2
    elif key[0] == "output":
        attn2 = model.model.diffusion_model.output_blocks[key[1]][1].transformer_blocks[key[2]].attn2

    to = model.model_options["transformer_options"]
    if "patches_replace" not in to:
        to["patches_replace"] = {}
    if "attn2" not in to["patches_replace"]:
        to["patches_replace"]["attn2"] = {}
    if key not in to["patches_replace"]["attn2"]:
        to["patches_replace"]["attn2"][key] = CrossAttentionPatch(prompt_embedding, region, attn2, width, height)
    else:
        to["patches_replace"]["attn2"][key].set_new_condition(prompt_embedding, region, attn2)


def get_center(pose_keypoints):
    min_x = 10e10
    max_x = -10e10
    xxx = np.array(pose_keypoints["pose_keypoints_2d"][::3])
    confidence = np.array(pose_keypoints["pose_keypoints_2d"][2::3])
    for x, c in zip(xxx, confidence):
        if c != 0 and x is not None:
            min_x = min(min_x, x)
            max_x = max(max_x, x)
    return min_x + (max_x - min_x) / 2


class CrossAttentionPatch:
    def __init__(self, prompt_embedding, region, attn, width, height):
        self.embeddings = [prompt_embedding]
        self.regions =  [region]
        self.attns = [attn]
        self.width = width
        self.height = height
    
    def set_new_condition(self, prompt_embedding, region, attn):
        self.embeddings.append(prompt_embedding)
        self.regions.append(region)
        self.attns.append(attn)

    def region_rewrite(self, hidden_states, query, region_list, height, width):
        dtype = query.dtype
        seq_lens = query.shape[1]
        downscale = math.sqrt(height * width / seq_lens)
        feat_height, feat_width = int(height // downscale), int(width // downscale)
        region_mask = torch.zeros((feat_height, feat_width))
        for region in region_list:
            start_h, start_w, end_h, end_w = region[-1]
            start_h, start_w, end_h, end_w = (
                math.ceil(start_h * feat_height),
                math.ceil(start_w * feat_width),
                math.floor(end_h * feat_height),
                math.floor(end_w * feat_width),
            )
            region_mask[start_h:end_h, start_w:end_w] += 1

        query = rearrange(query, "b (h w) c -> b h w c", h=feat_height, w=feat_width)
        hidden_states = rearrange(hidden_states, "b (h w) c -> b h w c", h=feat_height, w=feat_width)
        new_hidden_state = torch.zeros_like(hidden_states)
        new_hidden_state[:, region_mask == 0, :] = hidden_states[:, region_mask == 0, :]
        for region in region_list:
            region_key, region_value, region_box = region
            start_h, start_w, end_h, end_w = region_box
            start_h, start_w, end_h, end_w = (
                math.ceil(start_h * feat_height),
                math.ceil(start_w * feat_width),
                math.floor(end_h * feat_height),
                math.floor(end_w * feat_width),
            )

            attention_region = (
                einsum("b h w c, b n c -> b h w n", query[:, start_h:end_h, start_w:end_w, :], region_key) #* attn.scale
            )

            attention_region = attention_region.softmax(dim=-1)
            attention_region = attention_region.to(dtype)

            hidden_state_region = einsum("b h w n, b n c -> b h w c", attention_region, region_value)
            new_hidden_state[:, start_h:end_h, start_w:end_w, :] += (
                hidden_state_region
                / (region_mask.reshape(1, *region_mask.shape, 1)[:, start_h:end_h, start_w:end_w, :]).to(query.device)
            )

        new_hidden_state = rearrange(new_hidden_state, "b h w c -> b (h w) c")
        return new_hidden_state

    def __call__(self, query, key, value, extra_options):
        is_cross =  query.shape[1] != value.shape[1]
        org_dtype = query.dtype

        query = utils.head_to_batch_dim(extra_options["n_heads"], query)
        key = utils.head_to_batch_dim(extra_options["n_heads"], key)
        value = utils.head_to_batch_dim(extra_options["n_heads"], value)
        
        hidden_states = xformers.ops.memory_efficient_attention(query, key, value)
        hidden_states = hidden_states.to(query.dtype)
        
        if is_cross:
            region_list = []
            idx = 0
            for prompt_embeds, region, attn in zip(self.embeddings, self.regions, self.attns):
                idx += 1
                if len(prompt_embeds.shape) == 4:
                    region_key = attn.to_k(prompt_embeds[:, self.cross_attention_idx, ...])
                    region_value = attn.to_v(prompt_embeds[:, self.cross_attention_idx, ...])
                else:
                    region_key = attn.to_k(prompt_embeds)
                    region_value = attn.to_v(prompt_embeds)

                region_key = utils.head_to_batch_dim(extra_options["n_heads"], region_key)
                region_value = utils.head_to_batch_dim(extra_options["n_heads"], region_value)
                region_list.append((region_key, region_value, region))

            hidden_states = self.region_rewrite(
                hidden_states=hidden_states,
                query=query,
                region_list=region_list,
                height=self.height,
                width=self.width,
            )
        hidden_states = utils.batch_to_head_dim(extra_options["n_heads"], hidden_states)
        return hidden_states.to(dtype=org_dtype)


class RegionalAttentionProcessorNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", ),
                "body_regions": ("AREAS", ),
                "width": ("INT", {"default": 1024, "min": 0, "step": 1}),
                "height": ("INT", {"default": 1024, "min": 0, "step": 1}),
                "prompt_embedding_0": ("CONDITIONING", ),
                "negative_embedding_0": ("CONDITIONING", ),
                "prompt_embedding_1": ("CONDITIONING", ),
                "negative_embedding_1": ("CONDITIONING", ),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_atenntion_processor"
    CATEGORY = "SALT/Multisubject"
    # All patches with cross attention
    PATCH_KEYS = [
        # input
        ("input", 4, 0), ("input", 4, 1),
        ("input", 5, 0), ("input", 5, 1),
        ("input", 7, 0), ("input", 7, 1), ("input", 7, 2), ("input", 7, 3), ("input", 7, 4),
        ("input", 7, 5), ("input", 7, 6), ("input", 7, 7), ("input", 7, 8), ("input", 7, 9),
        ("input", 8, 0), ("input", 8, 1), ("input", 8, 2), ("input", 8, 3), ("input", 8, 4),
        ("input", 8, 5), ("input", 8, 6), ("input", 8, 7), ("input", 8, 8), ("input", 8, 9),
        # middle
        ("middle", 0, 0), ("middle", 0, 1), ("middle", 0, 2), ("middle", 0, 3), ("middle", 0, 4), 
        ("middle", 0, 5), ("middle", 0, 6), ("middle", 0, 7), ("middle", 0, 8), ("middle", 0, 9),
        # output
        ("output", 0, 0), ("output", 0, 1), ("output", 0, 2), ("output", 0, 3), ("output", 0, 4),
        ("output", 0, 5),  ("output", 0, 6), ("output", 0, 7), ("output", 0, 8), ("output", 0, 9),
        ("output", 1, 0), ("output", 1, 1), ("output", 1, 2), ("output", 1, 3), ("output", 1, 4),
        ("output", 1, 5), ("output", 1, 6), ("output", 1, 7), ("output", 1, 8), ("output", 1, 9),
        ("output", 2, 0), ("output", 2, 1), ("output", 2, 2), ("output", 2, 3), ("output", 2, 4),
        ("output", 2, 5), ("output", 2, 6), ("output", 2, 7), ("output", 2, 8), ("output", 2, 9),
        ("output", 3, 0), ("output", 3, 1),
        ("output", 4, 0), ("output", 4, 1),
        ("output", 5, 0), ("output", 5, 1),
    ]

    def apply_atenntion_processor(
        self,
        model,
        body_regions,
        width,
        height,
        **kwargs,
    ):
        prompts = [v for k, v in kwargs.items() if "prompt_embedding" in k]
        neagative_prompts = [v for k, v in kwargs.items() if "negative_embedding" in k]
        self.dtype = torch.float16 if comfy.model_management.should_use_fp16() else torch.float32
        self.device = comfy.model_management.get_torch_device()
        self.work_model = model.clone()
        for prompt_embedding, negative_prompt_embedding, region in zip(prompts, neagative_prompts, body_regions):
            prompt_embedding = prompt_embedding[0][0].to(device=self.device, dtype=self.dtype)
            negative_prompt_embedding = negative_prompt_embedding[0][0].to(device=self.device, dtype=self.dtype)
            prompt_embedding = torch.cat([negative_prompt_embedding, prompt_embedding], dim=0)
            for k in self.PATCH_KEYS:
                set_model_patch_replace(self.work_model, prompt_embedding, region, k, width, height)
        return (self.work_model,)


class RegionalPromptingNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose": ("POSE_KEYPOINT", ),
            },
        }

    RETURN_TYPES = ("AREAS", "FACE_AREAS", "LIST_OF_MASK", "IMAGE")
    RETURN_NAMES = ("body_regions", "face_regions", "mask_regions", "debug_image")

    FUNCTION = "get_regions"
    CATEGORY = "SALT/Multisubject"

    COLORS = [
        [0, 0, 128],
        [0, 128, 0],
        [0, 128, 128],
        [128, 0, 0],
        [128, 0, 128],
        [128, 128, 0],
        [128, 128, 128],
        [0, 0, 64],
        [0, 64, 0],
        [0, 64, 64],
        [64, 0, 0],
        [64, 0, 64],
        [64, 64, 0],
        [64, 64, 64],
    ]

    def get_regions(self, pose):
        body_regions = []
        face_regions = []
        # Sort poses based on the x-coordinate of the first keypoint
        pose[0]["people"] = sorted(pose[0]["people"], key=lambda x: get_center(x), reverse=False)
        for pose_keypoints in pose[0]["people"]:
            min_y, min_x, max_y, max_x = 10e6, 10e6, 0, 0
            xxx = np.array(pose_keypoints["pose_keypoints_2d"][::3])
            yyy = np.array(pose_keypoints["pose_keypoints_2d"][1::3])
            confidence = np.array(pose_keypoints["pose_keypoints_2d"][2::3])
            # Find body regions
            for x, y, c in zip(xxx, yyy, confidence):
                if c == 0:
                    continue
                min_x = min(min_x, x)
                max_x = max(max_x, x)
                min_y = min(min_y, y)
                max_y = max(max_y, y)
            body_regions.append((min_y, min_x, max_y, max_x))
            # Find face regions
            face_min_y, face_min_x, face_max_y, face_max_x = 10e6, 10e6, 0, 0
            face_indexes = [0, 14, 15, 16, 17]
            for x, y, c in zip(xxx[face_indexes], yyy[face_indexes], confidence[face_indexes]):
                if c == 0:
                    continue
                face_min_x = min(face_min_x, x)
                face_max_x = max(face_max_x, x)
                face_min_y = min(face_min_y, y)
                face_max_y = max(face_max_y, y)
            if pose_keypoints["face_keypoints_2d"] is not None:
                face_xxx = np.array(pose_keypoints["face_keypoints_2d"])[::3]
                face_yyy = np.array(pose_keypoints["face_keypoints_2d"])[1::3]
                face_confidence = np.array(pose_keypoints["face_keypoints_2d"])[2::3]
                for x, y, c in zip(face_xxx, face_yyy, face_confidence):
                    if c == 0:
                        continue
                    face_min_x = min(face_min_x, x)
                    face_max_x = max(face_max_x, x)
                    face_min_y = min(face_min_y, y)
                    face_max_y = max(face_max_y, y)
            face_regions.append((face_min_y, face_min_x, face_max_y, face_max_x))
        
        # Generate mask for each region
        height = pose[0]["canvas_height"]
        width = pose[0]["canvas_width"]
        debug_image = np.zeros((height, width, 3), dtype=np.uint8)
        masks = []
        for idx, (min_y, min_x, max_y, max_x) in enumerate(body_regions + face_regions):
            # Scale coordinates
            min_y = int(round(min_y * height))
            min_x = int(round(min_x * width))
            max_y = int(round(max_y * height))
            max_x = int(round(max_x * width))
            mask = np.zeros((height, width, 3), dtype=np.uint8)
            cv2.rectangle(mask, (min_x, min_y), (max_x, max_y), self.COLORS[idx % len(self.COLORS)], -1)
            mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
            masks.append(torch.unsqueeze(torch.tensor(mask, dtype=torch.float32) / 255.0, 0))
            # Draw a debug image to show a regions
            cv2.rectangle(debug_image, (min_x, min_y), (max_x, max_y), self.COLORS[idx % len(self.COLORS)], 5)
        debug_image = torch.unsqueeze(torch.tensor(debug_image), 0)
        return body_regions, face_regions, masks, debug_image


class GetRegionalMaskNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "masks": ("LIST_OF_MASK",),
                "person_number": ("INT", {"default": 0, "min": 0, "step": 1}),
                "part": (["body", "head"], ),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)

    FUNCTION = "get_mask"
    CATEGORY = "SALT/Multisubject"

    def get_mask(self, masks, person_number, part):
        if part == "head":
            person_number += len(masks) // 2
        return masks[person_number]
