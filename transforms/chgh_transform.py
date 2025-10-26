# 檔案: transforms/chgh_transform.py
# (已加入治本方案的最終版本)

from monai.transforms import (
    EnsureChannelFirstd,  
    Compose,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    Transposed, # <--- 1. 導入 Transposed
    EnsureTyped,  # <--- 2. 導入 EnsureTyped (取代 ToTensorD)
)

def get_train_transform(args):
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),

            # --- START CRITICAL FIX (治本方案) ---
            # 這是最關鍵的修正：
            # MONAI 載入 NIfTI 檔案時，維度通常是 (H, W, D)。
            # 我們將其轉置 (transpose) 為模型期望的 (D, H, W) 格式。
            Transposed(keys=["image", "label"], indices=[0, 3, 1, 2]),
            # --- END CRITICAL FIX ---
            
            # 確保資料類型正確，這比舊的 ToTensorD 更推薦
            EnsureTyped(keys=["image", "label"]),
            
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(args.space_x, args.space_y, args.space_z),
                mode=("bilinear", "nearest"),
            ),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=args.a_min, 
                a_max=args.a_max,
                b_min=getattr(args, 'b_min', 0.0), # 使用 getattr 提供預設值
                b_max=getattr(args, 'b_max', 1.0),
                clip=True,
            ),
            # 注意：CropForeground 可能會改變影像大小，需要在 Spacing 之後
            # 如果您的流程中沒有 CropForegroundd，可以忽略此註解
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                pos=1,
                neg=1,
                num_samples=args.num_samples,
                image_key="image",
                image_threshold=0,
            ),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[0],
                prob=args.rand_flipd_prob,
            ),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[1],
                prob=args.rand_flipd_prob,
            ),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[2],
                prob=args.rand_flipd_prob,
            ),
            RandRotate90d(
                keys=["image", "label"],
                prob=args.rand_rotate90d_prob,
                max_k=3,
            ),
            RandShiftIntensityd(
                keys=["image"],
                offsets=0.10,
                prob=args.rand_shift_intensityd_prob,
            ),
            # ToTensorD 在 MONAI 新版本中通常由 EnsureTyped 自動處理，可以移除
        ]
    )


def get_val_transform(args):
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),

            # --- START CRITICAL FIX (治本方案) ---
            # 驗證集也必須進行同樣的維度轉置！
            Transposed(keys=["image", "label"], indices=[0, 3, 1, 2]),
            # --- END CRITICAL FIX ---
            
            EnsureTyped(keys=["image", "label"]),
            
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(args.space_x, args.space_y, args.space_z),
                mode=("bilinear", "nearest"),
            ),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=args.a_min, 
                a_max=args.a_max, 
                b_min=getattr(args, 'b_min', 0.0), 
                b_max=getattr(args, 'b_max', 1.0), 
                clip=True
            ),
            # ToTensorD 可以移除
        ]
    )


def get_inf_transform(keys, args):
    if len(keys) == 2:
        mode = ("bilinear", "nearest")
    elif len(keys) == 3:
        mode = ("bilinear", "nearest", "nearest")
    else:
        mode = ("bilinear",) 
        
    return Compose(
        [
            LoadImaged(keys=keys),
            EnsureChannelFirstd(keys=keys),

            # --- START CRITICAL FIX (治本方案) ---
            # 推斷時也必須進行同樣的維度轉置！
            Transposed(keys=['image'], indices=[0, 3, 1, 2]),
            # --- END CRITICAL FIX ---

            EnsureTyped(keys=keys),
            
            # 您的原始推斷流程包含了重採樣，這裡予以保留
            Orientationd(keys=keys, axcodes="RAS"),
            Spacingd(
                keys=keys,
                pixdim=(args.space_x, args.space_y, args.space_z),
                mode=mode,
            ),
            ScaleIntensityRanged(
                keys=['image'],
                a_min=args.a_min, 
                a_max=args.a_max,
                b_min=getattr(args, 'b_min', 0.0), 
                b_max=getattr(args, 'b_max', 1.0),
                clip=True,
                allow_missing_keys=True # 保持這個，因為 label 可能不存在
            ),
            # ToTensorD 可以移除
        ]
    )


def get_label_transform(keys=["label"]):
    # 這個函數用於在評估原始結果時重新載入標籤
    return Compose(
        [
            LoadImaged(keys=keys),
            EnsureChannelFirstd(keys=keys),
            # 注意：這裡不需要 Transposed，因為 Restored 函數會將 pred
            # 轉換回原始標籤的空間，所以我們需要載入原始方向的標籤來比較。
        ]
    )
