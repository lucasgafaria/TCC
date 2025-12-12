#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generic_final_with_gt_png.py - Versão final (CPU-only) com carregamento robusto de checkpoints
Inclui: visualização comparativa em um único PNG com 8 colunas:
 Imagem | GT | Pred | Grad-CAM | Grad-CAM++ | IG | GuidedBP | Guided Grad-CAM
Usa automaticamente GTs na pasta: D:/Codes/Python/Task09_Spleen/Task09_Spleen/labelsTr
Autor: Lucas
"""

import os
import time
import csv
from glob import glob
from typing import Tuple, Optional, Dict, Any, List
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import pandas as pd
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F

# MONAI imports
from monai.bundle import ConfigParser
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader
from monai.transforms import (
    LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
    ScaleIntensityRanged, RandCropByPosNegLabeld, ToTensord, Compose
)

# Captum (IG) - optional
HAS_CAPTUM = False
try:
    from captum.attr import IntegratedGradients
    HAS_CAPTUM = True
except Exception:
    print("⚠️ Captum não encontrado — IG será pulado. Para usar IG instale: pip install captum")
    HAS_CAPTUM = False

# -------------------- USER CONFIG (ajuste conforme necessário) --------------------
BUNDLE_DIR = r"D:/Codes/Python/bundles/spleen_ct_segmentation"
TASK09_DIR = r"D:/Codes/Python/Task09_Spleen/Task09_Spleen"
OUT_DIR = r"D:/Codes/Python/TCC/testeTudoCinq"
PLOTS_DIR = os.path.join(OUT_DIR, "plots")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# FORCE CPU
DEVICE = torch.device("cpu")

# If you want to run training in this script set to True and configure related params.
RUN_TRAINING = True

NUM_EPOCHS = 150
CHECKPOINT_EPOCHS = [0, 1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 155, 160, 165, 170, 175, 180]
NUM_VAL_SAMPLES_TO_RUN_XAI = 2

# IG params (Captum) - reduce for CPU
IG_STEPS = 12  # reduzido para CPU (pode diminuir se OOM)
IG_BASELINE = None  # None -> zeros baseline

# General
IOU_THRESHOLD = 0.5
TOPK_PER = 0.2  # used in frac_inside / top-k masking
ROI_SIZE = (96, 96, 96)
SW_BATCH = 1
OVERLAP = 0.5

# -------------------- Helpers (I/O / metrics) --------------------
def save_nifti(volume_np: np.ndarray, out_path: str, affine: Optional[np.ndarray] = None):
    if affine is None:
        affine = np.eye(4)
    img = nib.Nifti1Image(volume_np.astype(np.float32), affine)
    nib.save(img, out_path)


def compute_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_bin = (pred > 0).astype(np.uint8)
    gt_bin = (gt > 0).astype(np.uint8)
    inter = (pred_bin & gt_bin).sum()
    denom = pred_bin.sum() + gt_bin.sum()
    if denom == 0:
        return 1.0 if inter == 0 else 0.0
    return 2.0 * inter / (denom + 1e-8)


def calculate_iou(explanation_map: np.ndarray, gt_mask: np.ndarray, threshold: float = IOU_THRESHOLD) -> float:
    if explanation_map is None:
        return 0.0
    exp = np.array(explanation_map, dtype=np.float32)
    if exp.size == 0 or np.all(exp == 0):
        return 0.0
    exp_min = float(exp.min()); exp_max = float(exp.max())
    exp_norm = (exp - exp_min) / (exp_max - exp_min + 1e-8)
    exp_bin = exp_norm > float(threshold)
    gt_bin = np.array(gt_mask > 0, dtype=np.bool_)
    intersection = np.logical_and(exp_bin, gt_bin).sum()
    union = np.logical_or(exp_bin, gt_bin).sum()
    return float(intersection) / float(union) if union > 0 else 0.0


def cam_topk_mask(cam: np.ndarray, top_perc: float = TOPK_PER) -> np.ndarray:
    flat = cam.flatten()
    if flat.size == 0:
        return np.zeros_like(cam, dtype=np.uint8)
    kth = np.percentile(flat, 100*(1-top_perc))
    return (cam >= kth).astype(np.uint8)


def compute_cam_metrics(cam: np.ndarray, gt_mask: np.ndarray, top_perc: float = TOPK_PER) -> Dict[str, Any]:
    cam_norm = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    mask_cam = cam_topk_mask(cam_norm, top_perc=top_perc)
    gt_bin = (gt_mask > 0).astype(np.uint8)
    inter = (mask_cam & gt_bin).sum(); union = (mask_cam | gt_bin).sum()
    iou = inter / (union + 1e-8) if union > 0 else 0.0
    frac_inside = (cam_norm * gt_bin).sum() / (cam_norm.sum() + 1e-8)
    def com(bin_arr):
        coords = np.array(np.nonzero(bin_arr))
        if coords.size == 0:
            return None
        return coords.mean(axis=1)
    com_cam = com(mask_cam); com_gt = com(gt_bin)
    com_shift = float(np.linalg.norm(com_cam - com_gt)) if (com_cam is not None and com_gt is not None) else None
    return {"iou": float(iou), "frac_inside": float(frac_inside), "com_shift": com_shift}

# -------------------- Tensor utilities (padding / bbox) --------------------
def pad_to_multiple(tensor: torch.Tensor, multiple: int = 16) -> Tuple[torch.Tensor, tuple]:
    pads = []
    for dim in tensor.shape[2:]:
        remainder = int(dim % multiple)
        pads.append(0 if remainder == 0 else multiple - remainder)
    pad = (0, pads[2], 0, pads[1], 0, pads[0])  # x,y,z pairs reversed order for F.pad
    padded = F.pad(tensor, pad)
    return padded, pad


def unpad_tensor(tensor: torch.Tensor, pad: tuple) -> torch.Tensor:
    # pad = (0, xpad, 0, ypad, 0, zpad)
    zpad = pad[5]; ypad = pad[3]; xpad = pad[1]
    z = tensor.shape[2] - zpad
    y = tensor.shape[3] - ypad
    x = tensor.shape[4] - xpad
    return tensor[..., :z, :y, :x]


def bbox_from_mask(mask: np.ndarray, margin: int = 8):
    coords = np.where(mask == 1)
    if coords[0].size == 0:
        return None
    zmin, ymin, xmin = [int(coords[i].min()) for i in range(3)]
    zmax, ymax, xmax = [int(coords[i].max()) for i in range(3)]
    zmin = max(0, zmin - margin); ymin = max(0, ymin - margin); xmin = max(0, xmin - margin)
    zmax = min(mask.shape[0]-1, zmax + margin)
    ymax = min(mask.shape[1]-1, ymax + margin)
    xmax = min(mask.shape[2]-1, xmax + margin)
    return (slice(zmin, zmax+1), slice(ymin, ymax+1), slice(xmin, xmax+1))


def find_last_conv3d(module: nn.Module) -> Optional[nn.Conv3d]:
    last = None
    for m in module.modules():
        if isinstance(m, nn.Conv3d):
            last = m
    return last

# -------------------- Robust state_dict adaptation & loading --------------------
def normalize_key_for_matching(k: str) -> str:
    prefixes = ("module.", "model.", "network.", "net.")
    new = k
    changed = True
    while changed:
        changed = False
        for p in prefixes:
            if new.startswith(p):
                new = new[len(p):]
                changed = True
    return new


def build_compatible_state_dict(model_sd: Dict[str, torch.Tensor], ck_sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    model_keys = list(model_sd.keys())
    ck_keys = list(ck_sd.keys())
    norm_to_ck = {}
    for k in ck_keys:
        nk = normalize_key_for_matching(k)
        norm_to_ck.setdefault(nk, []).append(k)
    final_map = {}
    used_ck = set()
    for mk in model_keys:
        mk_norm = normalize_key_for_matching(mk)
        candidates = norm_to_ck.get(mk_norm, [])
        chosen = None
        for ck in candidates:
            if ck in used_ck:
                continue
            try:
                if hasattr(model_sd[mk], "shape") and hasattr(ck_sd[ck], "shape") and model_sd[mk].shape == ck_sd[ck].shape:
                    chosen = ck; break
            except Exception:
                continue
        if chosen:
            final_map[mk] = ck_sd[chosen]; used_ck.add(chosen)
    for mk in model_keys:
        if mk in final_map:
            continue
        if mk in ck_sd and mk not in used_ck:
            if hasattr(model_sd[mk], "shape") and hasattr(ck_sd[mk], "shape") and model_sd[mk].shape == ck_sd[mk].shape:
                final_map[mk] = ck_sd[mk]; used_ck.add(mk)
    ck_tokens_map = {k: k.split(".") for k in ck_keys}
    for mk in model_keys:
        if mk in final_map:
            continue
        mk_tokens = mk.split(".")
        best_ck = None; best_score = 0
        for ck, tokens in ck_tokens_map.items():
            if ck in used_ck:
                continue
            score = 0
            for a, b in zip(reversed(mk_tokens), reversed(tokens)):
                if a == b:
                    score += 1
                else:
                    break
            if score > best_score:
                best_score = score; best_ck = ck
        if best_ck and best_score >= 2:
            try:
                if hasattr(model_sd[mk], "shape") and hasattr(ck_sd[best_ck], "shape") and model_sd[mk].shape == ck_sd[best_ck].shape:
                    final_map[mk] = ck_sd[best_ck]; used_ck.add(best_ck)
            except Exception:
                pass
    remaining_ck = [k for k in ck_keys if k not in used_ck]
    shape_to_ck = {}
    for ck in remaining_ck:
        shp = getattr(ck_sd[ck], "shape", None)
        shape_to_ck.setdefault(shp, []).append(ck)
    for mk in model_keys:
        if mk in final_map:
            continue
        shp = getattr(model_sd[mk], "shape", None)
        cands = shape_to_ck.get(shp, [])
        if len(cands) == 1:
            final_map[mk] = ck_sd[cands[0]]
            used_ck.add(cands[0])
            shape_to_ck[shp].remove(cands[0])
    mapped_sd = {}
    for mk in model_keys:
        if mk in final_map:
            mapped_sd[mk] = final_map[mk]
    return mapped_sd


def safe_load_state_dict(net: nn.Module, ckpt_path: str) -> bool:
    if not os.path.exists(ckpt_path):
        print(f"❌ Checkpoint não encontrado: {ckpt_path}")
        return False
    try:
        try:
            ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)  # type: ignore
        except TypeError:
            ck = torch.load(ckpt_path, map_location="cpu")
    except Exception as e:
        print("Erro carregando checkpoint com torch.load:", e)
        return False
    if isinstance(ck, dict):
        if "state_dict" in ck and isinstance(ck["state_dict"], dict):
            sdict = ck["state_dict"]
        elif "model" in ck and isinstance(ck["model"], dict):
            sdict = ck["model"]
        elif "network" in ck and isinstance(ck["network"], dict):
            sdict = ck["network"]
        elif "net" in ck and isinstance(ck["net"], dict):
            sdict = ck["net"]
        elif "state_dict_ema" in ck and isinstance(ck["state_dict_ema"], dict):
            sdict = ck["state_dict_ema"]
        else:
            sdict = ck
    else:
        print("Formato de checkpoint não reconhecido (esperado dict). Abortando load.")
        return False
    sdict_norm = {}
    for k, v in sdict.items():
        newk = k
        for p in ("module.",):
            if newk.startswith(p):
                newk = newk[len(p):]
        sdict_norm[newk] = v
    try:
        net.load_state_dict(sdict_norm)
        print("✅ Pesos carregados (strict=True) a partir do checkpoint:", os.path.basename(ckpt_path))
        return True
    except Exception as e:
        print("🔔 Falha ao carregar com strict=True. Tentando strict=False. Motivo:", e)
    try:
        net.load_state_dict(sdict_norm, strict=False)
        print("✅ Pesos carregados (strict=False) a partir do checkpoint (após strip 'module.').")
        return True
    except Exception as e:
        print("⚠️ Falha ao carregar com strict=False usando only-strip normalization. Motivo:", e)
    try:
        model_sd = net.state_dict()
        mapped_sd = build_compatible_state_dict(model_sd, sdict)
        if not mapped_sd:
            mapped_sd = build_compatible_state_dict(model_sd, sdict_norm)
        if mapped_sd:
            try:
                net.load_state_dict(mapped_sd, strict=False)
                print("✅ Pesos parcialmente carregados via mapeamento por normalização/sufixos/shape (strict=False).")
                missing = set(model_sd.keys()) - set(mapped_sd.keys())
                if missing:
                    print(f"⚠️ Ainda faltando {len(missing)} parâmetros no modelo (estes permanecerão random-inicializados).")
                return True
            except Exception as e_map:
                print("❌ Falha ao carregar mapeamento parcial via load_state_dict:", e_map)
        else:
            print("⚠️ Nenhum mapeamento útil encontrado entre checkpoint e modelo.")
    except Exception as e:
        print("❌ Erro ao tentar mapeamento avançado de chaves:", e)
    return False

# -------------------- XAI Methods (mesmas implementações com proteções) --------------------
def gradcam3d_from_model(net: nn.Module, input_tensor: torch.Tensor, target_class: int = 1,
                         margin: int = 8, device: Optional[torch.device] = None) -> Tuple[np.ndarray, np.ndarray]:
    if device is None: device = DEVICE
    net = net.to(device)
    net.eval()
    x = input_tensor.to(device)
    with torch.no_grad():
        out = sliding_window_inference(x, roi_size=ROI_SIZE, sw_batch_size=SW_BATCH, predictor=net, overlap=OVERLAP)
    mask_pred = out.argmax(dim=1).squeeze().cpu().numpy()
    bbox = bbox_from_mask(mask_pred, margin=margin)
    if bbox is None:
        zc, yc, xc = [s//2 for s in mask_pred.shape]
        half = min(64, min(mask_pred.shape)//2)
        bbox = (slice(zc-half, zc+half), slice(yc-half, yc+half), slice(xc-half, xc+half))
    zsl, ysl, xsl = bbox
    x_crop = x[..., zsl, ysl, xsl]
    x_crop_padded, pad_info = pad_to_multiple(x_crop, multiple=16)
    target_layer = find_last_conv3d(net)
    if target_layer is None:
        raise RuntimeError("No nn.Conv3d found in model for Grad-CAM")
    activations = {}
    gradients = {}
    def fwd_hook(module, inp, out):
        activations['value'] = out
    def bwd_hook(module, gin, gout):
        gradients['value'] = gout[0]
    h_fwd = target_layer.register_forward_hook(fwd_hook)
    try:
        h_bwd = target_layer.register_full_backward_hook(bwd_hook)
    except Exception:
        h_bwd = target_layer.register_backward_hook(bwd_hook)
    try:
        net.zero_grad(set_to_none=True)
        x_crop_padded = x_crop_padded.clone().detach().to(device).requires_grad_(True)
        out_crop = net(x_crop_padded)
        out_ch = out_crop.shape[1]
        tc = min(max(0, target_class), out_ch-1)
        score = out_crop[:, tc].mean()
        score.backward()
        A = activations.get('value', None)
        G = gradients.get('value', None)
        if A is None or G is None:
            raise RuntimeError("Não foram capturadas ativações/gradientes na camada alvo.")
        weights = G.mean(dim=(2,3,4), keepdim=True)
        cam = torch.sum(weights * A, dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam - cam.min()
        if cam.max().abs() > 0:
            cam = cam / (cam.max() + 1e-8)
        cam_up = F.interpolate(cam, size=x_crop_padded.shape[2:], mode="trilinear", align_corners=False)
        cam_up = unpad_tensor(cam_up, pad_info)
        cam_up_np = cam_up.squeeze().detach().cpu().numpy()
    finally:
        try:
            h_fwd.remove(); h_bwd.remove()
        except Exception:
            pass
        torch.cuda.empty_cache(); gc.collect()
    cam_full = np.zeros_like(mask_pred, dtype=np.float32)
    cam_full[zsl, ysl, xsl] = cam_up_np
    return cam_full, mask_pred

def gradcampp3d_from_model(net: nn.Module, input_tensor: torch.Tensor, target_class: int = 1,
                           margin: int = 8, device: Optional[torch.device] = None) -> Tuple[np.ndarray, np.ndarray]:
    if device is None: device = DEVICE
    net = net.to(device)
    net.eval()
    x = input_tensor.to(device)
    with torch.no_grad():
        out = sliding_window_inference(x, roi_size=ROI_SIZE, sw_batch_size=SW_BATCH, predictor=net, overlap=OVERLAP)
    mask_pred = out.argmax(dim=1).squeeze().cpu().numpy()
    bbox = bbox_from_mask(mask_pred, margin=margin)
    if bbox is None:
        zc, yc, xc = [s//2 for s in mask_pred.shape]
        half = min(64, min(mask_pred.shape)//2)
        bbox = (slice(zc-half, zc+half), slice(yc-half, yc+half), slice(xc-half, xc+half))
    zsl, ysl, xsl = bbox
    x_crop = x[..., zsl, ysl, xsl]
    x_crop_padded, pad_info = pad_to_multiple(x_crop, multiple=16)
    target_layer = find_last_conv3d(net)
    if target_layer is None:
        raise RuntimeError("No nn.Conv3d found in model for Grad-CAM++")
    activations = {}
    gradients = {}
    def fwd_hook(module, inp, out):
        activations['value'] = out
    def bwd_hook(module, gin, gout):
        gradients['value'] = gout[0]
    h_fwd = target_layer.register_forward_hook(fwd_hook)
    try:
        h_bwd = target_layer.register_full_backward_hook(bwd_hook)
    except Exception:
        h_bwd = target_layer.register_backward_hook(bwd_hook)
    net.zero_grad(set_to_none=True)
    x_cp = x_crop_padded.clone().detach().to(device).requires_grad_(True)
    try:
        out_cp = net(x_cp)
        out_ch = out_cp.shape[1]
        tc = min(max(0, target_class), out_ch-1)
        score = out_cp[:, tc].mean()
        score.backward(create_graph=True)
        A = activations.get('value', None)
        G = gradients.get('value', None)
        if A is None or G is None:
            raise RuntimeError("Ativações/gradientes não capturados para Grad-CAM++.")
        g2 = G * G
        g3 = g2 * G
        denom = 2 * g2 + (A * g3).sum(dim=(2,3,4), keepdim=True)
        denom = denom + 1e-12
        alpha = g2 / denom
        pos_grad = F.relu(G)
        weights = (alpha * pos_grad).sum(dim=(2,3,4), keepdim=True)
        cam = (weights * A).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam - cam.min()
        if cam.max().abs() > 0:
            cam = cam / (cam.max() + 1e-8)
        cam_up = F.interpolate(cam, size=x_cp.shape[2:], mode="trilinear", align_corners=False)
        cam_up = unpad_tensor(cam_up, pad_info)
        cam_up_np = cam_up.squeeze().detach().cpu().numpy()
    except RuntimeError as e:
        print("⚠️ Grad-CAM++ falhou (OOM/autograd) — fallback para Grad-CAM. Erro:", e)
        try:
            h_fwd.remove(); h_bwd.remove()
        except Exception:
            pass
        return gradcam3d_from_model(net, input_tensor, target_class=target_class, margin=margin, device=device)
    except Exception as e:
        print("⚠️ Grad-CAM++ erro -> fallback Grad-CAM:", e)
        try:
            h_fwd.remove(); h_bwd.remove()
        except Exception:
            pass
        return gradcam3d_from_model(net, input_tensor, target_class=target_class, margin=margin, device=device)
    try:
        h_fwd.remove(); h_bwd.remove()
    except Exception:
        pass
    cam_full = np.zeros_like(mask_pred, dtype=np.float32)
    cam_full[zsl, ysl, xsl] = cam_up_np
    return cam_full, mask_pred

# Guided Backprop
class GuidedBackprop3D:
    def __init__(self, model: nn.Module):
        self.model = model
        self.handlers = []
        self._register_hooks()
    def _register_hooks(self):
        for module in self.model.modules():
            if isinstance(module, nn.ReLU):
                def relu_backward_hook(module, grad_input, grad_output):
                    if grad_output is None:
                        return None
                    try:
                        new_grad_output = tuple(F.relu(go) if (go is not None) else None for go in grad_output)
                        if grad_input is None:
                            return None
                        res = []
                        for gi in grad_input:
                            if gi is None:
                                res.append(None)
                            else:
                                go0 = new_grad_output[0]
                                try:
                                    gi_new = torch.zeros_like(gi)
                                    if go0.shape == gi.shape:
                                        gi_new = torch.clamp(go0, min=0.0)
                                    else:
                                        gi_new = go0.mean() * torch.ones_like(gi, device=gi.device)
                                    res.append(gi_new)
                                except Exception:
                                    res.append(None)
                        return tuple(res)
                    except Exception:
                        return None
                try:
                    h = module.register_full_backward_hook(relu_backward_hook)
                except Exception:
                    h = module.register_backward_hook(lambda m, gi, go: relu_backward_hook(m, gi, go))
                self.handlers.append(h)
    def generate_gradients(self, input_tensor: torch.Tensor, net: nn.Module, target_class: int = 1, device: Optional[torch.device]=None) -> np.ndarray:
        if device is None: device = DEVICE
        net = net.to(device)
        net.eval()
        x = input_tensor.to(device)
        x = x.clone().detach().requires_grad_(True)
        out = sliding_window_inference(x, roi_size=ROI_SIZE, sw_batch_size=SW_BATCH, predictor=net, overlap=OVERLAP)
        out_ch = out.shape[1]
        tc = min(max(0, target_class), out_ch-1)
        score_map = out[:, tc]
        score = score_map.view(score_map.shape[0], -1).mean(dim=1).sum()
        net.zero_grad(set_to_none=True)
        score.backward()
        grad = x.grad.detach().cpu().squeeze().numpy()
        if grad.ndim == 4:
            grad = np.mean(grad, axis=0)
        grad = np.maximum(grad, 0.0)
        if grad.max() > 0:
            grad = grad / (grad.max() + 1e-8)
        return grad
    def remove_hooks(self):
        for h in self.handlers:
            try: h.remove()
            except Exception: pass
        self.handlers = []

def guided_gradcam_from_components(gradcam_vol: np.ndarray, guided_grad_vol: np.ndarray) -> np.ndarray:
    g = gradcam_vol * guided_grad_vol
    if g.max() > 0:
        g = (g - g.min()) / (g.max() - g.min() + 1e-8)
    return g

def integrated_gradients_3d(net: nn.Module, input_tensor: torch.Tensor, target_class: int = 1,
                            steps: int = 12, baseline: Optional[torch.Tensor] = None,
                            margin: int = 8, device: Optional[torch.device] = None) -> Tuple[np.ndarray, np.ndarray]:
    if not HAS_CAPTUM:
        raise RuntimeError("Captum não instalado.")
    if device is None: device = DEVICE
    net = net.to(device)
    net.eval()
    x = input_tensor.to(device)
    with torch.no_grad():
        out = sliding_window_inference(x, roi_size=ROI_SIZE, sw_batch_size=SW_BATCH, predictor=net, overlap=OVERLAP)
    mask_pred = out.argmax(dim=1).squeeze().cpu().numpy()
    bbox = bbox_from_mask(mask_pred, margin=margin)
    if bbox is None:
        zc, yc, xc = [s//2 for s in mask_pred.shape]
        half = min(64, min(mask_pred.shape)//2)
        bbox = (slice(zc-half, zc+half), slice(yc-half, yc+half), slice(xc-half, xc+half))
    zsl, ysl, xsl = bbox
    x_crop = x[..., zsl, ysl, xsl]
    x_crop_padded, pad_info = pad_to_multiple(x_crop, multiple=16)
    x_crop_padded = x_crop_padded.clone().detach().requires_grad_(True).to(device)
    if baseline is None:
        baseline_t = torch.zeros_like(x_crop_padded).to(device)
    else:
        baseline_t = baseline.to(device)
    class WrappedModel(nn.Module):
        def __init__(self, net_ref, target_cls):
            super().__init__()
            self.net_ref = net_ref
            self.target_cls = target_cls
        def forward(self, xin):
            out_net = self.net_ref(xin)
            class_map = out_net[:, self.target_cls]
            class_map_flat = class_map.view(class_map.shape[0], -1)
            return class_map_flat.mean(dim=1)
    wrapped = WrappedModel(net, target_class)
    ig = IntegratedGradients(wrapped)
    try:
        attributions = ig.attribute(x_crop_padded, baselines=baseline_t, n_steps=steps)
    except RuntimeError as e:
        print("⚠️ IG runtime error (possível OOM) -> tentando em CPU/fewer steps. Error:", e)
        try:
            wrapped_cpu = WrappedModel(net.cpu(), target_class)
            ig_cpu = IntegratedGradients(wrapped_cpu)
            x_cpu = x_crop_padded.cpu()
            baseline_cpu = baseline_t.cpu()
            attributions = ig_cpu.attribute(x_cpu, baselines=baseline_cpu, n_steps=max(4, steps//4))
            attributions = attributions.to(DEVICE) if DEVICE.type == "cuda" else attributions
        except Exception as e2:
            print("⚠️ IG fallback falhou:", e2)
            raise e2
    at = attributions.detach()
    at = at - at.min()
    at = at / (at.max() + 1e-8)
    at_unpadded = unpad_tensor(at, pad_info)
    at_np = at_unpadded.squeeze().cpu().numpy()
    ig_full = np.zeros_like(mask_pred, dtype=np.float32)
    ig_full[zsl, ysl, xsl] = at_np
    return ig_full, mask_pred

# -------------------- Visualization helper (single PNG with 8 columns) --------------------
def save_comparison_png(
    img_tensor: torch.Tensor,
    gt_np: np.ndarray,
    pred_np: np.ndarray,
    cam: Optional[np.ndarray],
    campp: Optional[np.ndarray],
    ig: Optional[np.ndarray],
    gbp: Optional[np.ndarray],
    guidedcam: Optional[np.ndarray],
    save_path: str,
    title: str = "",
):
    # Garantir que img_tensor existe e tem formato correto
    if img_tensor is None or img_tensor.ndim != 5:
        raise ValueError("img_tensor inválido ao gerar PNG")

    # Extrair volume da imagem
    img_np = img_tensor[0, 0].cpu().numpy()

    # Se GT for None, cria volume vazio compatível
    if gt_np is None:
        gt_np = np.zeros_like(img_np)

    # Se predição for None, cria volume vazio
    if pred_np is None:
        pred_np = np.zeros_like(img_np)

    # (CORREÇÃO PRINCIPAL) — Escolher slice com maior área de GT
    sum_per_slice = gt_np.sum(axis=(1, 2))
    if sum_per_slice.max() > 0:
        zmid = int(np.argmax(sum_per_slice))
    else:
        # fallback caso GT esteja realmente vazia
        zmid = img_np.shape[0] // 2

    # Agora sim, slices válidos
    slice_img = img_np[zmid]
    slice_gt = gt_np[zmid]
    slice_pred = pred_np[zmid]

    fig, axs = plt.subplots(1, 8, figsize=(36, 5))

    axs[0].imshow(slice_img, cmap="gray")
    axs[0].set_title("Imagem")
    axs[0].axis("off")

    axs[1].imshow(slice_gt, cmap="Reds", alpha=0.8)
    axs[1].set_title("GT")
    axs[1].axis("off")

    axs[2].imshow(slice_img, cmap="gray")
    axs[2].imshow(slice_pred, cmap="Blues", alpha=0.5)
    axs[2].set_title("Pred")
    axs[2].axis("off")

    def plot_map(ax, m, t):
        ax.imshow(slice_img, cmap="gray")
        if m is not None:
            ax.imshow(m[zmid], cmap="jet", alpha=0.45)
        ax.set_title(t)
        ax.axis("off")

    plot_map(axs[3], cam, "Grad-CAM")
    plot_map(axs[4], campp, "Grad-CAM++")
    plot_map(axs[5], ig, "IG")
    if gbp is not None:
        axs[6].imshow(slice_img, cmap="gray")
        axs[6].imshow(gbp[zmid], cmap="hot", alpha=0.6)
        axs[6].set_title("GuidedBP")
        axs[6].axis("off")
    else:
        axs[6].imshow(slice_img, cmap="gray"); axs[6].set_title("GuidedBP (None)"); axs[6].axis("off")

    plot_map(axs[7], guidedcam, "Guided Grad-CAM")

    if title:
        plt.suptitle(title, fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=150)
    plt.close(fig)

# -------------------- Pipeline: load model / data --------------------
def make_network_from_bundle(bundle_dir: str, load_weights: bool = False) -> nn.Module:
    cfg_infer = os.path.join(bundle_dir, "configs", "inference.json")
    ckpt = os.path.join(bundle_dir, "models", "model.pt")
    parser = ConfigParser()
    if not os.path.exists(cfg_infer):
        raise FileNotFoundError(f"Inference config not found in bundle: {cfg_infer}")
    parser.read_config(cfg_infer)
    net = parser.get_parsed_content("network")
    if not isinstance(net, nn.Module):
        raise RuntimeError("Parsed network from bundle is not an nn.Module instance.")
    if load_weights and os.path.exists(ckpt):
        try:
            try:
                state = torch.load(ckpt, map_location="cpu", weights_only=True)  # type: ignore
            except TypeError:
                state = torch.load(ckpt, map_location="cpu")
            sdict = state.get("state_dict", state)
            sdict_norm = {k[len("module."): ] if k.startswith("module.") else k: v for k,v in sdict.items()}
            try:
                net.load_state_dict(sdict_norm)
                print("-> Pesos carregados do bundle (model.pt) com strict=True")
            except Exception:
                net.load_state_dict(sdict_norm, strict=False)
                print("-> Pesos carregados do bundle (model.pt) com strict=False")
        except Exception as e:
            print("-> Erro carregando pesos do bundle (model.pt):", e)
    else:
        if load_weights:
            print("-> model.pt não encontrado no bundle; a rede será inicializada (random init).")
        else:
            print("-> Rede instanciada sem carregar pesos do bundle (random init)")
    return net

# -------------------- Main: process checkpoints and run XAI --------------------
CSV_PATH = os.path.join(OUT_DIR, "metrics_by_epoch.csv")
LABELS_DIR = os.path.join(TASK09_DIR, "labelsTr")

def process_checkpoints_and_run_xai():
    ckpts = sorted(glob(os.path.join(OUT_DIR, "checkpoint_epoch*.pt")),
                   key=lambda x: int(''.join(filter(str.isdigit, os.path.basename(x))) or 0))
    if not ckpts:
        raise RuntimeError(f"No checkpoint found in {OUT_DIR} (procure por checkpoint_epoch*.pt)")
    print("Checkpoints encontrados:", ckpts)

    images = sorted(glob(os.path.join(TASK09_DIR, "imagesTr", "*.nii*")))
    labels = sorted(glob(os.path.join(TASK09_DIR, "labelsTr", "*.nii*")))
    if len(images) == 0 or len(labels) == 0:
        raise RuntimeError("Nenhuma imagem/label encontrada nos diretórios Task09 especificados.")
    data_dicts = [{"image": i, "label": l} for i, l in zip(images, labels)]
    split = int(0.8 * len(data_dicts))
    val_files = data_dicts[split:]

    val_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 1.5), mode=("bilinear","nearest")),
        ScaleIntensityRanged(keys=["image"], a_min=-200, a_max=200, b_min=0.0, b_max=1.0, clip=True),
        ToTensord(keys=["image","label"])
    ])

    base_fields = [
        "epoch", "train_loss", "val_dice",
        "cam_iou", "cam_frac_inside", "cam_com_shift",
        "campp_iou", "campp_frac_inside", "campp_com_shift",
        "ig_iou", "ig_frac_inside", "ig_com_shift",
        "gbp_iou", "gbp_frac_inside", "gbp_com_shift",
        "guidedcam_iou", "guidedcam_frac_inside", "guidedcam_com_shift"
    ]
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=base_fields)
            writer.writeheader()

    for ckpt_path in ckpts:
        basename = os.path.basename(ckpt_path)
        digits = ''.join(filter(str.isdigit, basename))
        epoch_num = int(digits) if digits else 0
        print(f"\n--- PROCESSANDO CHECKPOINT EPOCH {epoch_num} ---")

        net = make_network_from_bundle(BUNDLE_DIR, load_weights=False)
        loaded = safe_load_state_dict(net, ckpt_path)
        if not loaded:
            print(f"⚠️ Falha ao carregar pesos do checkpoint {ckpt_path}. A rede continuará com inicialização random (verifique as chaves do checkpoint).")
        else:
            print(f"✅ Checkpoint {ckpt_path} carregado com sucesso (total/parcial).")

        net.to(DEVICE)
        net.eval()

        dices = []
        with torch.no_grad():
            for vf in val_files:
                data = val_transforms({"image": vf["image"], "label": vf["label"]})
                vimg = data["image"].unsqueeze(0).to(DEVICE)
                vlab = data["label"].squeeze().numpy()
                vout = sliding_window_inference(vimg, roi_size=ROI_SIZE, sw_batch_size=SW_BATCH, predictor=net, overlap=OVERLAP)
                pred_mask = vout.argmax(dim=1).squeeze().cpu().numpy()
                dices.append(compute_dice(pred_mask, vlab))
        val_dice = float(np.mean(dices)) if dices else None
        train_loss = None

        cams_metrics = []; campp_metrics = []; ig_metrics = []; gbp_metrics = []; guidedcam_metrics = []

        for i, sample in enumerate(val_files[:NUM_VAL_SAMPLES_TO_RUN_XAI]):
            print(f"→ Rodando XAI na sample {i}")
            data = val_transforms({"image": sample["image"], "label": sample["label"]})
            img_tensor = data["image"].unsqueeze(0)  # [1,1,Z,Y,X]
            label_np = data["label"].squeeze().numpy()

            mask_pred = None
            cam_vol = campp_vol = ig_vol = guided_grad = guidedcam = None

            # Grad-CAM
            try:
                cam_vol, mask_pred = gradcam3d_from_model(net, img_tensor, target_class=1, device=DEVICE)
                save_nifti(cam_vol, os.path.join(OUT_DIR, f"cam_epoch{epoch_num}_sample{i}.nii.gz"))
                m = compute_cam_metrics(cam_vol, label_np, top_perc=TOPK_PER)
                try:
                    m["iou"] = calculate_iou(cam_vol, label_np, threshold=IOU_THRESHOLD)
                except Exception:
                    m["iou"] = None
                cams_metrics.append(m)
            except Exception as e:
                print("Erro Grad-CAM:", e)

            # Grad-CAM++
            try:
                campp_vol, _ = gradcampp3d_from_model(net, img_tensor, target_class=1, device=DEVICE)
                save_nifti(campp_vol, os.path.join(OUT_DIR, f"campp_epoch{epoch_num}_sample{i}.nii.gz"))
                m = compute_cam_metrics(campp_vol, label_np, top_perc=TOPK_PER)
                try:
                    m["iou"] = calculate_iou(campp_vol, label_np, threshold=IOU_THRESHOLD)
                except Exception:
                    m["iou"] = None
                campp_metrics.append(m)
            except Exception as e:
                print("Erro Grad-CAM++:", e)

            # Integrated Gradients (Captum)
            if HAS_CAPTUM:
                try:
                    ig_vol, _ = integrated_gradients_3d(net, img_tensor, target_class=1, steps=IG_STEPS, baseline=None, device=DEVICE)
                    save_nifti(ig_vol, os.path.join(OUT_DIR, f"ig_epoch{epoch_num}_sample{i}.nii.gz"))
                    m = compute_cam_metrics(ig_vol, label_np, top_perc=TOPK_PER)
                    try:
                        m["iou"] = calculate_iou(ig_vol, label_np, threshold=IOU_THRESHOLD)
                    except Exception:
                        m["iou"] = None
                    ig_metrics.append(m)
                except Exception as e:
                    print("Erro IG:", e)
            else:
                print("IG não disponível (Captum ausente) -> pulado.")

            # Guided Backprop
            try:
                gbp = GuidedBackprop3D(net)
                guided_grad = gbp.generate_gradients(img_tensor, net, target_class=1, device=DEVICE)
                gbp.remove_hooks()
                save_nifti(guided_grad.astype(np.float32), os.path.join(OUT_DIR, f"gbp_epoch{epoch_num}_sample{i}.nii.gz"))
                m = compute_cam_metrics(guided_grad, label_np, top_perc=TOPK_PER)
                try:
                    m["iou"] = calculate_iou(guided_grad, label_np, threshold=IOU_THRESHOLD)
                except Exception:
                    m["iou"] = None
                gbp_metrics.append(m)
            except Exception as e:
                print("Erro Guided Backprop:", e)

            # Guided Grad-CAM
            try:
                gradcam_for_guided = None
                gfile_pp = os.path.join(OUT_DIR, f"campp_epoch{epoch_num}_sample{i}.nii.gz")
                gfile_c = os.path.join(OUT_DIR, f"cam_epoch{epoch_num}_sample{i}.nii.gz")
                if os.path.exists(gfile_pp):
                    gradcam_for_guided = nib.load(gfile_pp).get_fdata()
                elif os.path.exists(gfile_c):
                    gradcam_for_guided = nib.load(gfile_c).get_fdata()
                if gradcam_for_guided is None:
                    raise RuntimeError("Nenhum gradcam disponível para combinar com GuidedBP")
                guidedcam = guided_gradcam_from_components(gradcam_for_guided, guided_grad)
                save_nifti(guidedcam.astype(np.float32), os.path.join(OUT_DIR, f"guidedcam_epoch{epoch_num}_sample{i}.nii.gz"))
                m = compute_cam_metrics(guidedcam, label_np, top_perc=TOPK_PER)
                try:
                    m["iou"] = calculate_iou(guidedcam, label_np, threshold=IOU_THRESHOLD)
                except Exception:
                    m["iou"] = None
                guidedcam_metrics.append(m)
            except Exception as e:
                print("Erro Guided Grad-CAM:", e)

            # ensure we have a pred mask for visualization
            if mask_pred is None:
                try:
                    with torch.no_grad():
                        vout = sliding_window_inference(img_tensor.to(DEVICE), roi_size=ROI_SIZE, sw_batch_size=SW_BATCH, predictor=net, overlap=OVERLAP)
                    mask_pred = vout.argmax(dim=1).squeeze().cpu().numpy()
                except Exception:
                    mask_pred = np.zeros_like(label_np)

            # load GT from labelsTr folder explicitly (in case user wants original GT file) - prefer label_np already from transform
            # Infer GT filename from sample['label'] (we already have label_np)
            gt_np = label_np

            # save comparison PNG (single file)
            title = f"{os.path.basename(sample['image'])} - epoch{epoch_num} - sample{i}"
            out_png = os.path.join(OUT_DIR, f"compare_epoch{epoch_num}_sample{i}.png")
            try:
                save_comparison_png(img_tensor, gt_np, mask_pred, cam_vol, campp_vol, ig_vol, guided_grad, guidedcam, out_png, title=title)
            except Exception as e:
                print("Erro ao salvar PNG comparativo:", e)

        def agg(metrics_list):
            if metrics_list:
                avg_iou = float(np.mean([m["iou"] for m in metrics_list if m["iou"] is not None]))
                avg_frac = float(np.mean([m["frac_inside"] for m in metrics_list]))
                coms = [m["com_shift"] for m in metrics_list if m["com_shift"] is not None]
                avg_com = float(np.mean(coms)) if coms else None
                return avg_iou, avg_frac, avg_com
            return None, None, None

        cam_iou, cam_frac, cam_com = agg(cams_metrics)
        campp_iou, campp_frac, campp_com = agg(campp_metrics)
        ig_iou, ig_frac, ig_com = agg(ig_metrics)
        gbp_iou, gbp_frac, gbp_com = agg(gbp_metrics)
        guidedcam_iou, guidedcam_frac, guidedcam_com = agg(guidedcam_metrics)

        df_existing = pd.read_csv(CSV_PATH)
        if epoch_num in df_existing['epoch'].values:
            idx = df_existing.index[df_existing['epoch'] == epoch_num][0]
            df_existing.at[idx, "cam_iou"] = cam_iou
            df_existing.at[idx, "cam_frac_inside"] = cam_frac
            df_existing.at[idx, "cam_com_shift"] = cam_com
            df_existing.at[idx, "campp_iou"] = campp_iou
            df_existing.at[idx, "campp_frac_inside"] = campp_frac
            df_existing.at[idx, "campp_com_shift"] = campp_com
            df_existing.at[idx, "ig_iou"] = ig_iou
            df_existing.at[idx, "ig_frac_inside"] = ig_frac
            df_existing.at[idx, "ig_com_shift"] = ig_com
            df_existing.at[idx, "gbp_iou"] = gbp_iou
            df_existing.at[idx, "gbp_frac_inside"] = gbp_frac
            df_existing.at[idx, "gbp_com_shift"] = gbp_com
            df_existing.at[idx, "guidedcam_iou"] = guidedcam_iou
            df_existing.at[idx, "guidedcam_frac_inside"] = guidedcam_frac
            df_existing.at[idx, "guidedcam_com_shift"] = guidedcam_com
        else:
            newrow = {c: None for c in df_existing.columns}
            newrow.update({
                "epoch": epoch_num, "train_loss": train_loss, "val_dice": val_dice,
                "cam_iou": cam_iou, "cam_frac_inside": cam_frac, "cam_com_shift": cam_com,
                "campp_iou": campp_iou, "campp_frac_inside": campp_frac, "campp_com_shift": campp_com,
                "ig_iou": ig_iou, "ig_frac_inside": ig_frac, "ig_com_shift": ig_com,
                "gbp_iou": gbp_iou, "gbp_frac_inside": gbp_frac, "gbp_com_shift": gbp_com,
                "guidedcam_iou": guidedcam_iou, "guidedcam_frac_inside": guidedcam_frac, "guidedcam_com_shift": guidedcam_com
            })
            df_existing = pd.concat([df_existing, pd.DataFrame([newrow])], ignore_index=True)

        df_existing = df_existing.sort_values("epoch").reset_index(drop=True)
        df_existing.to_csv(CSV_PATH, index=False)
        print(f"Epoch {epoch_num} metrics updated in CSV.")

    print("=== ALL CHECKPOINTS PROCESSED ===")

# -------------------- Plot generator --------------------
def generate_plots_from_csv():
    if not os.path.exists(CSV_PATH):
        print("CSV não encontrado:", CSV_PATH); return
    df = pd.read_csv(CSV_PATH)
    df_plot = df.copy()

    plt.figure(figsize=(10,5))
    if "cam_iou" in df_plot: plt.plot(df_plot["epoch"], df_plot["cam_iou"], marker="o", label="Grad-CAM")
    if "campp_iou" in df_plot: plt.plot(df_plot["epoch"], df_plot["campp_iou"], marker="o", label="Grad-CAM++")
    if "ig_iou" in df_plot: plt.plot(df_plot["epoch"], df_plot["ig_iou"], marker="o", label="IG")
    if "gbp_iou" in df_plot: plt.plot(df_plot["epoch"], df_plot["gbp_iou"], marker="o", label="GuidedBP")
    if "guidedcam_iou" in df_plot: plt.plot(df_plot["epoch"], df_plot["guidedcam_iou"], marker="o", label="Guided Grad-CAM")
    plt.title("IoU - Métodos de Explicabilidade (por época)"); plt.xlabel("Época"); plt.ylabel("IoU"); plt.grid(True); plt.legend()
    plt.savefig(os.path.join(PLOTS_DIR, "iou_all_methods.png")); plt.close()

    plt.figure(figsize=(10,5))
    if "val_dice" in df_plot:
        plt.plot(df_plot["epoch"], df_plot["val_dice"], marker="o")
        plt.title("Evolução do Dice"); plt.xlabel("Época"); plt.ylabel("Dice"); plt.grid(True)
        plt.savefig(os.path.join(PLOTS_DIR, "dice_evolution.png")); plt.close()

    plt.figure(figsize=(10,5))
    for col, label in [("cam_frac_inside","Grad-CAM"), ("campp_frac_inside","Grad-CAM++"), ("ig_frac_inside","IG"), ("gbp_frac_inside","GuidedBP"), ("guidedcam_frac_inside","Guided Grad-CAM")]:
        if col in df_plot:
            plt.plot(df_plot["epoch"], df_plot[col], marker="o", label=label)
    plt.title("Fraction Inside (energia dentro da máscara)"); plt.xlabel("Época"); plt.ylabel("Fraction Inside"); plt.grid(True); plt.legend()
    plt.savefig(os.path.join(PLOTS_DIR, "frac_inside_all.png")); plt.close()

    plt.figure(figsize=(10,5))
    for col, label in [("cam_com_shift","Grad-CAM"), ("campp_com_shift","Grad-CAM++"), ("ig_com_shift","IG"), ("gbp_com_shift","GuidedBP"), ("guidedcam_com_shift","Guided Grad-CAM")]:
        if col in df_plot:
            plt.plot(df_plot["epoch"], df_plot[col], marker="o", label=label)
    plt.title("COM Shift (distância do centro de massa)"); plt.xlabel("Época"); plt.ylabel("Distância"); plt.grid(True); plt.legend()
    plt.savefig(os.path.join(PLOTS_DIR, "com_shift_all.png")); plt.close()

    print("Plots gerados em:", PLOTS_DIR)

# -------------------- Main entrypoint --------------------
def main():
    start = time.time()
    print("=== PIPELINE XAI - Versão final CPU-only (carregamento robusto) ===")
    process_checkpoints_and_run_xai()
    generate_plots_from_csv()
    print("Concluído em {:.1f}s".format(time.time() - start))

if __name__ == "__main__":
    main()