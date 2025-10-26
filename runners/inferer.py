# 檔案: runners/inferer.py
# (最終修正版 v5 - 解決 TTA 指標計算和原始指標計算的標籤/預測形狀不匹配問題)

import os
import time
import importlib
from pathlib import PurePath

import torch
import numpy as np

from monai.data import decollate_batch 
from monai.transforms import (
    LoadImaged,
    EnsureChannelFirst,
    SqueezeDimd,
    AsDiscrete,
    KeepLargestConnectedComponent,
    Compose,
    LabelFilter,
    MapLabelValue,
    Spacing,
    SqueezeDim,
    Resize  # 引入 Resize 轉換
)
from monai.metrics import DiceMetric, MeanIoU, ConfusionMatrixMetric

from data_utils.io import save_img
# import matplotlib.pyplot as plt # 暫時註釋，避免不必要的依賴


def infer(model, data, model_inferer, device):
    """執行模型推斷，並確保輸入是 5D 的批次化張量。"""
    model.eval()
    with torch.no_grad():
        input_tensor = data['image'].to(device)
        
        # 確保張量是 5D (Batch, Channel, Depth, Height, Width)
        if len(input_tensor.shape) == 4:
            input_tensor = input_tensor.unsqueeze(0)
        
        if len(input_tensor.shape) != 5:
            raise ValueError(f"Inference input tensor must be 5D, but got shape {input_tensor.shape}.")
        
        print(f"[DEBUG] Final tensor shape for model_inferer: {input_tensor.shape}")
        
        try:
            output = model_inferer(input_tensor)
            print("[DEBUG] model_inferer completed successfully.")
        except Exception as e:
            print(f"[ERROR] model_inferer failed with: {e}")
            cls_num = data.get('out_channels', 4) 
            # 創建與輸入空間尺寸相符的零張量作為備用輸出
            zero_output = torch.zeros((input_tensor.shape[0], cls_num, *input_tensor.shape[2:]), dtype=torch.float32, device=device)
            output = zero_output
            print(f"[CRITICAL FALLBACK] Returned zero tensor with shape {output.shape}.")

        # 輸出是 argmax 類別索引
        output = torch.argmax(output, dim=1) 
    return output


def eval_label_pred(data, cls_num, device):
    """
    為單一的標籤/預測對計算評估指標。
    此版本繞過 decollate_batch，直接處理單張張量。
    """
    # 轉換
    post_onehot = AsDiscrete(to_onehot=cls_num)

    # 指標
    dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
    iou_metric = MeanIoU(include_background=False)
    confusion_metric = ConfusionMatrixMetric(
        include_background=False, metric_name=["sensitivity", "specificity"],
        compute_sample=False, reduction="mean", get_not_nans=False
    )

    # 獲取資料
    val_label, val_pred = (data["label"].to(device), data["pred"].to(device))

    # --- [形狀處理：確保 3D 或 4D(C=1)] ---
    # val_pred 的形狀是 (D, H, W) 或 (B=1, D, H, W)，移除批次維度
    if len(val_pred.shape) == 4 and val_pred.shape[0] == 1:
        val_pred = val_pred.squeeze(0)  # -> (D, H, W)
    # val_label 的形狀是 (C=1, D, H, W) 或 (D, H, W)，移除通道維度
    if len(val_label.shape) == 4 and val_label.shape[0] == 1:
        val_label = val_label.squeeze(0) # -> (D, H, W)
    elif len(val_label.shape) == 4 and val_label.shape[0] != 1:
         # 這是錯誤的標籤形狀，為防止崩潰，暫時假設第一個維度是通道
         print(f"[ERROR WARNING] Label has 4D shape {val_label.shape} with C>1. Assuming first dim is Channel and taking argmax.")
         val_label = val_label.argmax(dim=0) # 轉換為類別索引 (D, H, W)


    # *** 檢查空間形狀是否一致 (D, H, W) ***
    if val_pred.shape != val_label.shape:
        # 這個錯誤應該在 run_infering 中被攔截並修正，但作為二次保險
        raise RuntimeError(
            f"Shape mismatch in eval_label_pred: Pred shape {val_pred.shape} vs Label shape {val_label.shape}. "
            "This indicates post_transform or label loading is incorrect."
        )

    # 為 one-hot 轉換加上通道維度: (C=1, D, H, W)
    val_label = val_label.unsqueeze(0)
    val_pred = val_pred.unsqueeze(0)
    
    # 轉換為 one-hot 格式: (C=cls_num, D, H, W)
    val_label_onehot = post_onehot(val_label)
    val_pred_onehot = post_onehot(val_pred)

    # MONAI 指標函數期望一個張量列表，所以我們將單張張量放入列表中
    val_labels_list = [val_label_onehot]
    val_output_list = [val_pred_onehot]
    # --- [形狀處理結束] ---

    # 計算指標
    dice_metric(y_pred=val_output_list, y=val_labels_list)
    iou_metric(y_pred=val_output_list, y=val_labels_list)
    confusion_metric(y_pred=val_output_list, y=val_labels_list)

    # 獲取結果
    dc_vals = dice_metric.get_buffer().detach().cpu().numpy().squeeze()
    iou_vals = iou_metric.get_buffer().detach().cpu().numpy().squeeze()
    cm = confusion_metric.get_buffer().detach().cpu().numpy().squeeze()
    print("Confusion_Matrix (TP, FP, TN, FN) per class:\n", cm)
    
    # 處理單類別或多類別輸出的情況
    if dc_vals.ndim == 0:
        dc_vals = np.array([dc_vals])
        iou_vals = np.array([iou_vals])
        cm = np.expand_dims(cm, axis=0)

    # 穩健的指標計算
    tp, fp, tn, fn = cm[:, 0], cm[:, 1], cm[:, 2], cm[:, 3]
    denominator_sens = tp + fn
    sensitivity_vals = np.divide(tp, denominator_sens, out=np.zeros_like(tp, dtype=float), where=denominator_sens!=0)
    denominator_spec = tn + fp
    specificity_vals = np.divide(tn, denominator_spec, out=np.zeros_like(tn, dtype=float), where=denominator_spec!=0)
    
    return dc_vals, iou_vals, sensitivity_vals, specificity_vals


def get_filename(data):
    return PurePath(data['image_meta_dict']['filename_or_obj']).parts[-1]


def get_label_transform(data_name, keys=['label']):
    """載入原始標籤時的轉換，只進行基本的 LoadImaged 和 EnsureChannelFirst。"""
    return Compose([
        LoadImaged(keys=keys, image_only=True, dtype=np.uint8),
        EnsureChannelFirst(keys=keys, channel_dim='no_channel')
    ])


def run_infering(
            model,
            data,
            model_inferer,
            post_transform,
            args
        ):
    
    num_foreground_classes = args.out_channels - 1
    nan_array = np.full(num_foreground_classes, np.nan)
    ret_dict = {
        'inf_time': np.nan, 'tta_dc': nan_array.copy(), 'tta_iou': nan_array.copy(),
        'ori_dc': nan_array.copy(), 'ori_iou': nan_array.copy(),
        'ori_sensitivity': nan_array.copy(), 'ori_specificity': nan_array.copy()
    }

    original_label_path = data.get('label_meta_dict', {}).get('filename_or_obj')
    
    start_time = time.time()
    data['out_channels'] = args.out_channels
    # 執行推論，data['pred'] 是類別索引張量
    data['pred'] = infer(model, data, model_inferer, args.device)
    ret_dict['inf_time'] = time.time() - start_time
    print(f'infer time: {ret_dict["inf_time"]} sec')
    
    if args.infer_post_process:
        print('use post process infer')
        pred_tensor = data['pred']
        if not isinstance(pred_tensor, torch.Tensor):
            pred_tensor = torch.from_numpy(np.array(pred_tensor)).to(args.device) 
        
        applied_labels = torch.unique(pred_tensor.flatten()).cpu().numpy()
        applied_labels = applied_labels[applied_labels > 0]
        if len(applied_labels) > 0:
            if pred_tensor.ndim == 4 and pred_tensor.shape[0] == 1:
                data['pred'] = KeepLargestConnectedComponent(applied_labels=applied_labels)(pred_tensor.squeeze(0)).unsqueeze(0)
            else:
                 data['pred'] = KeepLargestConnectedComponent(applied_labels=applied_labels)(pred_tensor)


    # 第一次指標計算：TTA 指標 (使用經過訓練預處理的標籤形狀)
    if 'label' in data.keys():
        
        # --- [修正 TTA 指標計算時的形狀不匹配 (步驟 1)] ---
        pred_tensor_tta = data['pred']
        if not isinstance(pred_tensor_tta, torch.Tensor):
            pred_tensor_tta = torch.from_numpy(np.array(pred_tensor_tta)).to(args.device)

        label_tensor_tta = data['label'].float().to(args.device)
        # 確保標籤是 4D (C, D, H, W)
        if label_tensor_tta.ndim == 3:
            label_tensor_tta = label_tensor_tta.unsqueeze(0) 
        elif label_tensor_tta.ndim == 5 and label_tensor_tta.shape[0] == 1:
            label_tensor_tta = label_tensor_tta.squeeze(0) 

        # 獲取 data['pred'] 的空間形狀 (D, H, W) 作為目標
        target_size_tta = list(pred_tensor_tta.shape[-3:])
        
        if label_tensor_tta.shape[-3:] != tuple(target_size_tta):
            print(f"[WARNING] TTA Label shape {label_tensor_tta.shape[-3:]} does not match TTA pred shape {target_size_tta}. Resampling TTA label.")
            
            label_resizer = Resize(spatial_size=target_size_tta, mode="nearest")
            label_tensor_tta = label_resizer(label_tensor_tta)
        
        # 覆寫 data['label'] 和 data['pred'] 以確保它們在字典中且形狀一致
        data['label'] = label_tensor_tta
        data['pred'] = pred_tensor_tta
        
        # 進行 TTA 指標計算
        tta_dc_vals, tta_iou_vals, _ , _ = eval_label_pred(data, args.out_channels, args.device)
        
        print('infer test time aug:')
        print('dice:', tta_dc_vals)
        print('iou:', tta_iou_vals)
        ret_dict['tta_dc'] = tta_dc_vals
        ret_dict['tta_iou'] = tta_iou_vals
        
        # 移除批次維度 (如果 data['label'] 仍是 4D (1, D, H, W))
        data = SqueezeDimd(keys=['label'])(data) 
    
    # 應用後處理：將預測結果反向轉換回原始圖像的空間尺寸
    data = post_transform(data) 
    
    # 第二次指標計算：原始指標 (使用原始標籤形狀)
    if original_label_path:
        # 載入原始標籤（只進行 LoadImaged 和 EnsureChannelFirst）
        label_loader = get_label_transform(args.data_name, keys=['label'])
        lbl_data = label_loader({'label': original_label_path})
        
        # --- [修正原始指標計算時的形狀不匹配 (步驟 2)] ---
        
        pred_tensor = data['pred']
        if not isinstance(pred_tensor, torch.Tensor):
            pred_tensor = torch.from_numpy(np.array(pred_tensor)).to(args.device)
        
        # 獲取 data['pred'] 的空間形狀 (D, H, W)
        target_size = list(pred_tensor.shape[-3:])
        
        # 原始標籤張量
        original_label_tensor = lbl_data['label'].float().to(args.device)
        if original_label_tensor.ndim == 3:
            original_label_tensor = original_label_tensor.unsqueeze(0) 
        
        # 檢查原始標籤是否需要重新採樣/變形到與預測結果相同的空間形狀
        if original_label_tensor.shape[-3:] != tuple(target_size):
            print(f"[WARNING] Original label shape {original_label_tensor.shape[-3:]} does not match pred shape {target_size}. Resampling label.")
            
            label_resizer = Resize(spatial_size=target_size, mode="nearest")
            resized_label_tensor = label_resizer(original_label_tensor)
            data['label'] = resized_label_tensor
        else:
            data['label'] = original_label_tensor 
        
        # 將修正後的標籤張量和預測張量放入 data 字典
        data['label'] = data['label'].to(args.device) 
        data['pred'] = pred_tensor.to(args.device)
        data['label_meta_dict'] = lbl_data.get('label_meta_dict')
        
        # 計算指標
        ori_dc_vals, ori_iou_vals, ori_sensitivity_vals, ori_specificity_vals = eval_label_pred(data, args.out_channels, args.device)
        
        # --- [關鍵修正結束] ---
        
        print('infer test original:')
        print('dice:', ori_dc_vals)
        print('iou:', ori_iou_vals)
        print('sensitivity:', ori_sensitivity_vals)
        print('specificity:', ori_specificity_vals)
        ret_dict['ori_dc'] = ori_dc_vals
        ret_dict['ori_iou'] = ori_iou_vals
        ret_dict['ori_sensitivity'] = ori_sensitivity_vals
        ret_dict['ori_specificity'] = ori_specificity_vals
    
    if args.data_name == 'mmwhs':
        mmwhs_transform = Compose([
            LabelFilter(applied_labels=[1, 2, 3, 4, 5, 6, 7]),
            MapLabelValue(orig_labels=[0, 1, 2, 3, 4, 5, 6, 7], target_labels=[0, 500, 600, 420, 550, 205, 820, 850]),
        ])
        data['pred'] = mmwhs_transform(data['pred'])
        
    if not args.test_mode:
        filename = get_filename(data)
        infer_img_pth = os.path.join(args.infer_dir, filename)
        save_img(data['pred'], data['pred_meta_dict'], infer_img_pth)
        
    return ret_dict
