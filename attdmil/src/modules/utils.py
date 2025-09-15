import torch as th
import os
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def move_to_device(nested_list, device):
    """
    Moves a nested list of tensors to the specified device.

    Args:
        nested_list (list): The nested list of tensors.
        device (torch.device): The device to which the tensors should be moved.
    """
    if isinstance(nested_list, th.Tensor):
        return nested_list.to(device)
    elif isinstance(nested_list, list):
        return [move_to_device(item, device) for item in nested_list]
    elif isinstance(nested_list, dict):
        return {key: move_to_device(value, device) for key, value in nested_list.items()}
    elif isinstance(nested_list, str):
        return nested_list
    else:
        raise TypeError("All elements must be either tensors or lists of tensors.")
    

def label_to_logits(label):

        if label == "normal":
            return th.tensor(0, dtype=th.float32)
        elif label == "tumor":
            return th.tensor(1, dtype=th.float32)

def logits_to_label(logit):
    """
    Convert a logit (0.0 or 1.0) back to a label.
    """
    if logit == 0:
        return "normal"
    elif logit == 1:
        return "tumor"
    else:
        raise ValueError(f"Invalid logit for label: {logit}")
        
def cls_to_logits(cls):
        if cls == "negative":
            return th.tensor(0, dtype=th.float32)
        elif cls == "micro":
            return th.tensor(1, dtype=th.float32)
        elif cls == "macro":
            return th.tensor(2, dtype=th.float32)
        
def logits_to_cls(logit):
    """
    Convert a logit (0.0, 1.0, or 2.0) back to a class.
    """
    if logit == 0:
        return "negative"
    elif logit == 1:
        return "micro"
    elif logit == 2:
        return "macro"
    else:
        raise ValueError(f"Invalid logit for class: {logit}")
    

def get_tumor_annotation(
        case_name: str,
):
    annotations_path = "/home/space/datasets/camelyon16/annotations"
    annotations_path = f"{annotations_path}/{case_name}.png"
    if not os.path.isfile(annotations_path):
        print(f"Image {case_name} not found in {annotations_path}")
        return None
    with Image.open(annotations_path) as img:
        rgba_array = np.array(img)
        r_channel = rgba_array[:, :, 0]
        g_channel = rgba_array[:, :, 1]
        b_channel = rgba_array[:, :, 2]

        red_mask = (r_channel == 255) & (g_channel == 0) & (b_channel == 0)
        positions = np.argwhere(red_mask)
        red_positions = {i: (x, y) for i, (y, x) in enumerate(positions)}
    return rgba_array 

def cut_off(
        y_instance_pred,
        vis_mode,
        top_k=10,
        threshold=0.9,
):
    if not isinstance(y_instance_pred, th.Tensor):
        y_instance_pred = th.tensor(y_instance_pred)

    if vis_mode == "raw":
        return y_instance_pred

    elif vis_mode == "log":
        y_instance_pred = np.clip(y_instance_pred, a_min=1e-10, a_max=None)
        y_instance_pred = th.tensor(y_instance_pred)

        return y_instance_pred
    
    elif vis_mode == "percentile":
        percentile_min = np.percentile(y_instance_pred, 1)
        percentile_max = np.percentile(y_instance_pred, 99)
        value_clipped = th.clamp(y_instance_pred, percentile_min, percentile_max)
        y_instance_pred = (value_clipped - percentile_min) / (percentile_max - percentile_min)

        return y_instance_pred

def confusion_matrix(
        TP,
        TN,
        FP,
        FN,
        misc_save_path,
        set,
):
    confusion_matrix = np.array([[TP, FN],
                              [FP, TN]])

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(confusion_matrix, cmap="Reds")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted Positive", "Predicted Negative"])
    ax.set_yticklabels(["Actual Positive", "Actual Negative"])
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(confusion_matrix[i, j]),
                    ha="center", va="center", color="black", fontsize=20)
    plt.tight_layout()
    plt.savefig(f"{misc_save_path}/{set}_confusion_matrix.png")


def create_metadata(
        normal_train,
        micro_train,
        macro_train, 
        normal_val,
        micro_val,
        macro_val,
        test_lst,
        slide_metadata,
        case_metadata,
        path_to_save,
        
):
    
    train_data = pd.DataFrame({
        'slide_id': normal_train + micro_train + macro_train,
        'split': ['train'] * (len(normal_train) + len(micro_train) + len(macro_train)),
        #'class': ['normal'] * len(normal_train) + ['micro'] * len(micro_train) + ['macro'] * len(macro_train)
    })

    val_data = pd.DataFrame({
        'slide_id': normal_val + micro_val + macro_val,
        'split': ['val'] * (len(normal_val) + len(micro_val) + len(macro_val)),
        #'class': ['normal'] * len(normal_val) + ['micro'] * len(micro_val) + ['macro'] * len(macro_val)
    })

    test_data = pd.DataFrame({
        'slide_id': test_lst,
        'split': ['test'] * len(test_lst),
    })
    
    combined_data = pd.concat([train_data, val_data, test_data], ignore_index=True)

    final_df = combined_data.merge(slide_metadata[['slide_id', 'case_id']], on='slide_id', how='left')

    final_df = final_df.merge(case_metadata[['case_id', 'class', 'type']], on='case_id', how='left')

    final_df = final_df.rename(columns={'type': 'label'})

    final_df = final_df.set_index('slide_id').reindex(slide_metadata['slide_id']).reset_index()

    final_df.to_csv(path_to_save, index=False)
    
    print(f"Split metadata saved to {path_to_save}")