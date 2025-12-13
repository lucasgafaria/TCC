import os
import nibabel as nib
import numpy as np
import torch
import matplotlib.pyplot as plt
from monai.inferers import sliding_window_inference
from monai.transforms import (
    LoadImage,
    EnsureChannelFirst,
    Orientation,
    Spacing,
    ScaleIntensityRange
)
from monai.bundle import ConfigParser
import torch.nn as nn

BUNDLE_DIR = r"D:/Codes/Python/bundles/spleen_ct_segmentation"
TASK09_DIR = r"D:/Codes/Python/Task09_Spleen/Task09_Spleen"

OUT_DIR = r"D:/Codes/Python/TCC/testeTudo"
RESULT_DIR = os.path.join(OUT_DIR, "comparisons_final")
os.makedirs(RESULT_DIR, exist_ok=True)

CHECKPOINT_EPOCHS = [0, 1, 80, 150]
ROI_SIZE = (96, 96, 96)
DEVICE = torch.device("cpu")

def make_network_from_bundle(bundle_dir):
    cfg_infer = os.path.join(bundle_dir, "configs", "inference.json")
    parser = ConfigParser()
    parser.read_config(cfg_infer)
    net = parser.get_parsed_content("network")
    if not isinstance(net, nn.Module):
        raise RuntimeError("Rede inválida no bundle.")
    return net

def load_checkpoint(net, ckpt_path):
    ck = torch.load(ckpt_path, map_location="cpu")
    if "state_dict" in ck:
        ck = ck["state_dict"]

    new_ck = {}
    for k, v in ck.items():
        if k.startswith("module."):
            new_ck[k[7:]] = v
        else:
            new_ck[k] = v

    net.load_state_dict(new_ck, strict=False)
    print(f"✔ Checkpoint carregado: {ckpt_path}")


def predict_mask(net, image_tensor):
    with torch.no_grad():
        out = sliding_window_inference(
            image_tensor, roi_size=ROI_SIZE, sw_batch_size=1, predictor=net, overlap=0.5
        )
    return out.argmax(dim=1).squeeze().cpu().numpy()

def main():

    images = sorted([os.path.join(TASK09_DIR, "imagesTr", f) for f in os.listdir(os.path.join(TASK09_DIR, "imagesTr"))])
    labels = sorted([os.path.join(TASK09_DIR, "labelsTr", f) for f in os.listdir(os.path.join(TASK09_DIR, "labelsTr"))])

    split = int(0.8 * len(images))
    img_path = images[split]    
    gt_path = labels[split]

    print("🟦 Usando imagem:", img_path)
    print("🟩 Usando GT:", gt_path)

    img = LoadImage(image_only=True)(img_path)
    gt = LoadImage(image_only=True)(gt_path)

    img = EnsureChannelFirst()(img)
    img = Orientation(axcodes="RAS")(img)
    img = Spacing(pixdim=(1.5,1.5,1.5), mode="bilinear")(img)
    img = ScaleIntensityRange(a_min=-200, a_max=200, b_min=0, b_max=1)(img)

    gt = EnsureChannelFirst()(gt)
    gt = Orientation(axcodes="RAS")(gt)
    gt = Spacing(pixdim=(1.5,1.5,1.5), mode="nearest")(gt)

    img_np = img[0]
    gt_np = gt[0]

    zmid = img_np.shape[0] // 2

    for epoch in CHECKPOINT_EPOCHS:
        print(f"\n--- Gerando figura da época {epoch} ---")

        ckpt_path = os.path.join(OUT_DIR, f"checkpoint_epoch{epoch}.pt")
        if not os.path.exists(ckpt_path):
            print(f"❌ Checkpoint não encontrado: {ckpt_path}")
            continue

        net = make_network_from_bundle(BUNDLE_DIR)
        load_checkpoint(net, ckpt_path)
        net.eval().to(DEVICE)

        pred = predict_mask(net, torch.tensor(img_np).unsqueeze(0).unsqueeze(0).float())

        def load_map(name):
            p = os.path.join(OUT_DIR, f"{name}_epoch{epoch}_sample0.nii.gz")
            return nib.load(p).get_fdata() if os.path.exists(p) else None

        cam = load_map("cam")
        campp = load_map("campp")
        ig = load_map("ig")

        if cam is None or campp is None or ig is None:
            print(f"⚠ Mapas XAI faltando para época {epoch}.")
            continue

        cam_slice = cam[zmid]
        campp_slice = campp[zmid]
        ig_slice = ig[zmid]

        img_slice = img_np[zmid]
        gt_slice = gt_np[zmid]
        pred_slice = pred[zmid]

        fig, axs = plt.subplots(1, 6, figsize=(22, 4))

        axs[0].imshow(img_slice, cmap="gray")
        axs[0].set_title("Imagem")
        axs[1].imshow(gt_slice, cmap="gray")
        axs[1].set_title("GT")
        axs[2].imshow(img_slice, cmap="gray")
        axs[2].imshow(pred_slice, alpha=0.4)
        axs[2].set_title("Pred")

        axs[3].imshow(img_slice, cmap="gray")
        axs[3].imshow(cam_slice, alpha=0.4)
        axs[3].set_title("Grad-CAM")

        axs[4].imshow(img_slice, cmap="gray")
        axs[4].imshow(campp_slice, alpha=0.4)
        axs[4].set_title("Grad-CAM++")

        axs[5].imshow(img_slice, cmap="gray")
        axs[5].imshow(ig_slice, alpha=0.4)
        axs[5].set_title("IG")

        for ax in axs:
            ax.axis("off")

        out_path = os.path.join(RESULT_DIR, f"comparison_epoch{epoch}_sample0.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()

        print("✔ Figura salva em:", out_path)

    print("\n=== FINALIZADO ===")

if __name__ == "__main__":
    main()