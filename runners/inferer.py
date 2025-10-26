# 檔案: runners/inferer.py
# (最終修正版 v3 - 徹底解決 decollate_batch 問題)

import os
import time
import importlib
from pathlib import PurePath

import torch
import numpy as np

from monai.data import decollate_batch # 雖然不再主要使用，但為保留結構而導入
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
    SqueezeDim
)
from monai.metrics import DiceMetric, MeanIoU, ConfusionMatrixMetric

from data_utils.io import save_img
import matplotlib.pyplot as plt


def infer(model, data, model_inferer, device):
    """執行模型推斷，並確保輸入是 5D 的批次化張量。"""
    model.eval()
    with torch.no_grad():
        input_tensor = data['image'].to(device)
        
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
            zero_output = torch.zeros((input_tensor.shape[0], cls_num, *input_tensor.shape[2:]), dtype=torch.float32, device=device)
            output = zero_output
            print(f"[CRITICAL FALLBACK] Returned zero tensor with shape {output.shape}.")

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

    # --- [最終修正] 處理單張影像，而非批次 ---
    # `val_pred` 的形狀是 (B=1, D, H, W)，移除批次維度
    if len(val_pred.shape) == 4:
        val_pred = val_pred.squeeze(0)  # -> (D, H, W)

    # `val_label` 的形狀是 (C=1, D, H, W)，移除通道維度以統一處理
    if len(val_label.shape) == 4:
        val_label = val_label.squeeze(0) # -> (D, H, W)

    # 此時，val_label 和 val_pred 都應該是 3D 張量: (D, H, W)
    # 為 one-hot 轉換加上通道維度: (C=1, D, H, W)
    val_label = val_label.unsqueeze(0)
    val_pred = val_pred.unsqueeze(0)
    
    # 轉換為 one-hot 格式: (C=cls_num, D, H, W)
    val_label_onehot = post_onehot(val_label)
    val_pred_onehot = post_onehot(val_pred)

    # MONAI 指標函數期望一個張量列表，所以我們將單張張量放入列表中
    val_labels_list = [val_label_onehot]
    val_output_list = [val_pred_onehot]
    # --- [修正結束] ---

    # 計算指標
    dice_metric(y_pred=val_output_list, y=val_labels_list)
    iou_metric(y_pred=val_output_list, y=val_labels_list)
    confusion_metric(y_pred=val_output_list, y=val_labels_list)

    # 獲取結果
    dc_vals = dice_metric.get_buffer().detach().cpu().numpy().squeeze()
    iou_vals = iou_metric.get_buffer().detach().cpu().numpy().squeeze()
    cm = confusion_metric.get_buffer().detach().cpu().numpy().squeeze()
    print("Confusion_Matrix (TP, FP, TN, FN) per class:\n", cm)
    
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
    transform = importlib.import_module(f'transforms.{data_name}_transform')
    get_lbl_transform = getattr(transform, 'get_label_transform', None)
    return get_lbl_transform(keys)


def run_infering(
            model,
            data,
            model_inferer,
            post_transform,
            args
        ):
    # 此函數的其餘部分保持不變
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
    data['pred'] = infer(model, data, model_inferer, args.device)
    ret_dict['inf_time'] = time.time() - start_time
    print(f'infer time: {ret_dict["inf_time"]} sec')
    
    if args.infer_post_process:
        print('use post process infer')
        pred_tensor = data['pred']
        if not isinstance(pred_tensor, torch.Tensor):
            pred_tensor = torch.from_numpy(np.array(pred_tensor))
        applied_labels = torch.unique(pred_tensor.flatten()).cpu().numpy()
        applied_labels = applied_labels[applied_labels > 0]
        if len(applied_labels) > 0:
            data['pred'] = KeepLargestConnectedComponent(applied_labels=applied_labels)(pred_tensor)

    if 'label' in data.keys():
        tta_dc_vals, tta_iou_vals, _ , _ = eval_label_pred(data, args.out_channels, args.device)
        print('infer test time aug:')
        print('dice:', tta_dc_vals)
        print('iou:', tta_iou_vals)
        ret_dict['tta_dc'] = tta_dc_vals
        ret_dict['tta_iou'] = tta_iou_vals
        data = SqueezeDimd(keys=['label'])(data)
    
    data = post_transform(data)
    
    if original_label_path:
        label_loader = get_label_transform(args.data_name, keys=['label'])
        lbl_data = label_loader({'label': original_label_path})
        data['label'] = lbl_data['label']
        data['label_meta_dict'] = lbl_data.get('label_meta_dict')
        ori_dc_vals, ori_iou_vals, ori_sensitivity_vals, ori_specificity_vals = eval_label_pred(data, args.out_channels, args.device)
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
