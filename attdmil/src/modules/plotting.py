import os
import numpy as np
from PIL import Image
import wandb
import random
import torch as th
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from src.modules.utils import get_tumor_annotation, logits_to_label, cut_off
from matplotlib.colors import LogNorm
from tqdm import tqdm

def visualize_gtbags(
    bags: np.ndarray,
    labels: np.ndarray,
    idx: int, 
    positive_num: int, 
    show: bool,
    misc_save_path: str
):
    """
    Visualizes a bag of images and labels for multiple-instance learning (MIL).

    Displays images from a selected bag with individual labels, 
    indicating the overall bag status (positive or negative) based on the target label.

    Args:
        bags (np.ndarray): Array containing images in the bag; shape should be (1, num_images, height, width).
        labels (np.ndarray): Array with bag and instance labels; labels[0] is the bag label, labels[1] holds instance labels.
        idx (int): Index of the bag, used for file naming.
        positive_num (int): Target label considered positive for the bag.
        show (bool): If True, displays the plot; otherwise, saves it to file.

    Saves:
        Plot of bag images with labeled status in "./logs/misc/data" as "bag_{idx}_{bag_status}.png".

    """
    num_images = bags.shape[1]
    num_columns = int(np.ceil(np.sqrt(num_images)))
    num_rows = int(np.ceil(num_images / num_columns))
    fig, axes = plt.subplots(num_rows, num_columns, figsize=(num_columns * 2, num_rows * 2))

    axes = axes.flatten()
    is_positive_bag = labels[0] == 1
    bag_status = "Positive" if is_positive_bag else "Negative"
    color = 'green' if is_positive_bag else 'red'
    fig.suptitle(f'Bag Status: {bag_status}', fontsize=14, color=color)
    # 0.92 for mu10, 0.94 for mu50, 0.96 for mu100
    fig.text(0.5, 0.92, f"positive label: {positive_num}", ha='center', fontsize=12, color='black')

    for i in range(num_images):
        ax = axes[i]
        ax.imshow(bags[0][i].squeeze().cpu(), cmap='gray')

        label_value = labels[1][0][i]
        title_color = 'green' if label_value else 'red'

        ax.set_title(f'Label: {label_value}', color=title_color)
        ax.axis('off')

    for j in range(num_images, num_rows * num_columns):
        axes[j].axis('off')

    plt.tight_layout()
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if show:
        plt.show()
    else:
        plot_path = os.path.join(misc_save_path, f"sample_bag_{idx}_{bag_status}.png")
        try:
            plt.savefig(plot_path, format='png')
            image = wandb.Image(plot_path, caption="sample bag visualization")
            wandb.log({"sample bag visualization": image})
        except Exception as e:
            print(f"Error saving plot: {e}")
        finally:
            plt.close(fig) 

def visualize_attMechanism(
        model, 
        batch: tuple,
        positive_num: int,
        global_step: int,
        show: bool,
        misc_save_path: str
):
    bag, label = batch[0], batch[1]
    y_bag_true = label[0].float()

    y_bag_pred, y_instance_pred = model(bag.squeeze(0))
    if y_instance_pred is None:
        return
    else:
        num_images = batch[0].shape[1]
        num_columns = int(np.ceil(np.sqrt(num_images)))
        num_rows = int(np.ceil(num_images / num_columns))
        fig, axes = plt.subplots(num_rows, num_columns, figsize=(num_columns * 2, num_rows * 2))

        axes = axes.flatten()
        is_positive_bag = label[0] == 1
        bag_status = "Positive" if is_positive_bag else "Negative"
        color = 'green' if is_positive_bag else 'red'
        fig.suptitle(f'Bag Status: {bag_status}', fontsize=14, color=color)
        # 0.92 for mu10, 0.94 for mu50, 0.96 for mu100
        fig.text(0.5, 0.92, f"positive label: {positive_num}", ha='center', fontsize=12, color='black')

        for i in range(num_images):
            ax = axes[i]
            ax.imshow(bag[0][i].squeeze().cpu(), cmap='gray')

            label_value = round(y_instance_pred[i].item(), 5)
            color_v = label[1][0][i]
            title_color = 'green' if color_v else 'red'
    
            ax.set_title(f'a{i} = {label_value}', color=title_color)
            ax.axis('off')

        for j in range(num_images, num_rows * num_columns):
            axes[j].axis('off')

        plt.tight_layout()
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        if show:
            plt.show()
        else:
            plot_path = os.path.join(misc_save_path, f"att_bag_{bag_status}_{global_step}.png")
            try:
                plt.savefig(plot_path)
                image = wandb.Image(plot_path, caption="attention mechanism visualization")
                wandb.log({"attention mechanism visualization": image})
            except Exception as e:
                print(f"Error saving plot: {e}")
            finally:
                plt.close(fig) 


def visualize_auc_results(
    mean_bag_size: int,
    var_bag_size: float,
    save_path: str,
    svg_flag: bool,
    local_gpu_flag: bool
):
    """
    Visualizes AUC validation metric on MNIST-bags for different approaches, like in Figures 1-3 of the paper.
    The approaches are: ['instance+MAX', 'instance+MEAN', 'embedding+MAX', 'embedding+MEAN', 'Attention', 'Gated-Attention']
    Assumes that the results files are complete (include both mean and std) and there is a file corresponding to each case.
    
    Args:
        mean_bag_size (int): The average number of instances per bag, e.g. 10, 50 or 100.
        var_bag_size (float): The variance of the number of instances per bag, e.g. 2, 10, 20.
        save_path (str): The path where the resulting plot will be saved, e.g. './logs/misc/results'.
        svg_flag (bool): Flag stating, whether to save a plot in a vectorized format (.svg) or not (.png).
        local_gpu_flag (bool): Flag indicating, if the results to be plotted should be the ones from local GPU or not.

    Saves:
        Plot comparing AUC for different approaches in the given 'save_path' as 'auc_comparison_{mean_bag_size}'.
    """
    approaches = ['instance_poolmax', 'instance_poolmean', 'embedding_poolmax', 'embedding_poolmean', 'embedding_poolattention', 'embedding_poolgated_attention']
    num_train_bags = [50, 100, 150, 200, 300, 400, 500]
    auc_results = {}
    local_gpu_path_adjustment = 'local_gpu/' if local_gpu_flag else ''

    for approach in approaches:
        approach_results = []
        for num_bags in num_train_bags:
            path_to_res = f"./logs/{local_gpu_path_adjustment}new_mu{mean_bag_size}/{approach}_mu{mean_bag_size}_var{var_bag_size}_num{num_bags}/misc/metric_5runs.txt"
            auc_mean = -1
            auc_std = -1
            try:
                with open(path_to_res, 'r') as file:
                    for line in file:
                        if 'Mean' in line:
                            parts = line.split(" ")
                            try:
                                auc_mean = float(parts[1].strip())
                                continue
                            except ValueError:
                                print(f"Error: Convertion of mean '{parts[1]} to float failed.'")
                        elif 'Std' in line:
                            parts = line.split(" ")
                            try:
                                auc_std = float(parts[1].strip())
                                continue
                            except ValueError:
                                print(f"Error: Convertion of std '{parts[1]} to float failed.'")
            except FileNotFoundError:
                print(f"Error: The AUC result file '{path_to_res}' was not found.")
            approach_results.append((auc_mean, auc_std))
        auc_results[approach] = approach_results
    
    colors = ['#1a661a', '#81c784', 'royalblue', 'skyblue', 'salmon', 'firebrick']
    markers = ['o', 's', '^', '*', 'D', 'v']
    labels = ['instance+MAX', 'instance+MEAN', 'embedding+MAX', 'embedding+MEAN', 'Attention', 'Gated-Attention']

    fig, ax = plt.subplots(figsize=(10, 8), dpi=600)

    for idx, (approach, data) in enumerate(auc_results.items()):
        means = [point[0] for point in data]
        stds = [point[1] for point in data]
        ax.errorbar(
            num_train_bags, means, yerr=stds, label=labels[idx], color=colors[idx],
            marker=markers[idx], linestyle='dashdot', capsize=5
        )

    ax.set_xlabel('Number of training bags')
    ax.set_ylabel('AUC')
    ax.set_ylim((0.55, 1.0))
    loc = 'center right' if mean_bag_size == 10 else 'lower right'
    ax.legend(loc=loc)
    ax.grid(which='major', color='gray', linestyle='-', linewidth=0.5)
    ax.minorticks_on()
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.2)
    file_format = 'svg' if svg_flag else 'png'
    fig.savefig(f"{save_path}/auc_results_{mean_bag_size}.{file_format}", bbox_inches='tight')


def visualize_histo_att(
        model, 
        batch: tuple,
        misc_save_path: str,
        global_step: int,
        mode: str,
        vis_mode: str,
):
    features, label, cls, dict = batch
    features = features.squeeze(0)

    y_bag_pred, y_instance_pred = model(features)
    if y_instance_pred is None:
        return
    else:
        y_instance_pred = cut_off(y_instance_pred, top_k=10, threshold=0.9, vis_mode=vis_mode)
        y_instance_pred = y_instance_pred.cpu().detach().numpy()
        positions = [dict[i][1] for i in range(len(dict)-2)]
        patch_size_abs = [dict[i][2] for i in range(len(dict)-2)]
        original_shape = dict['original_shape']
        
        figsize = ((original_shape[0].item() / 100), (original_shape[1].item() / 100))

        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor("black")

        ax.set_xlim(0, original_shape[0].item())
        ax.set_ylim(0, original_shape[1].item())
        ax.invert_yaxis()
        max_ak = max(y_instance_pred)
        min_ak = min(y_instance_pred)
        log_norm = LogNorm(vmin=y_instance_pred.min(), vmax=y_instance_pred.max())

        for i, position in enumerate(positions):
            x, y = position
            x = x.item()/16
            y = y.item()/16

            patch_width = patch_size_abs[i].item()/16
            patch_height = patch_size_abs[i].item()/16

            value = y_instance_pred[i]
            if vis_mode == "percentile" or vis_mode == "raw":
                value = (value - min_ak) / (max_ak - min_ak)
            if vis_mode == "log":
                value = log_norm(value)
            color = plt.cm.Reds(value)
            rect = patches.Rectangle((x, y), patch_width, patch_height, linewidth=0, edgecolor=None, facecolor=color)
            ax.add_patch(rect)

        ax.axis('off')
        if misc_save_path:
            batch_name = dict['case_name'][0]
            batch_dir = os.path.join(misc_save_path, batch_name)
            if not os.path.exists(batch_dir):
                os.makedirs(batch_dir)

            image_save_path = f"{batch_dir}/{mode}_aks_{logits_to_label(label)}_{global_step+1}_{vis_mode}.png"
            plt.savefig(image_save_path, dpi=100, bbox_inches='tight', pad_inches=0)

            img = Image.open(image_save_path)
            img = img.resize((original_shape[0].item(), original_shape[1].item()), Image.Resampling.LANCZOS)
            img.save(image_save_path)
        plt.close()

def visualize_histo_smoothgrad(
        model, 
        batch: tuple,
        misc_save_path: str,
        global_step: int,
        mode: str,
        vis_mode: str,
):
    features, label, cls, dict = batch
    features = features.squeeze(0)
    
    model.eval()
    contributions_sum = th.zeros_like(features) 
    for _ in tqdm(range(10)):
        noise = th.randn_like(features) * 0.1
        noisy_features = features + noise
        noisy_features.requires_grad = True

        y_bag_pred, y_instance_pred = model(noisy_features)
        gradients = th.autograd.grad(
                outputs=y_bag_pred,
                inputs=noisy_features,
                grad_outputs=th.ones_like(y_bag_pred),
                create_graph=True,
                retain_graph=True,
            )[0]
        contributions_sum += gradients * noisy_features
    contributions = contributions_sum / 10
    contributions = th.sum(contributions, dim=1)
    contributions = contributions.cpu().detach().numpy()

    plot_fn(dict, label, contributions, None, misc_save_path, global_step, mode, vis_mode, 'saliency')

def visualize_histo_shap(
        model,
        batch: tuple,
        misc_save_path: str,
        global_step: int,
        mode: str,
        vis_mode: str,
):
    features, label, cls, dict = batch
    features = features.squeeze(0)
    num_instances = features.shape[0]
    shapley_values = th.zeros(num_instances)

    if misc_save_path:
        batch_name = dict['case_name'][0]
        batch_dir = os.path.join(misc_save_path, batch_name)
        if not os.path.exists(batch_dir):
            os.makedirs(batch_dir)
        shapley_save_path = f"{batch_dir}/{mode}_shap_{logits_to_label(label)}.pt"

    for _ in tqdm(range(100)):
        subset_indices = random.sample(range(num_instances), random.randint(1, num_instances - 1))
        subset = features[subset_indices]

        with th.no_grad():
            y_bag_pred, y_instance_pred = model(subset)
        for i in tqdm(range(num_instances)):
            if i not in subset_indices:
                subset_i = th.cat([subset, features[i:i+1]], dim=0)
                with th.no_grad():
                    y_bag_pred_i, y_instance_pred_i = model(subset_i)
                marginal_contribution = y_bag_pred_i - y_bag_pred
                shapley_values[i] += marginal_contribution.item()
    th.save(shapley_values, shapley_save_path)

    if os.path.exists(shapley_save_path):
        print(f"Loading shapley values from {batch_name}")
        shapley_values = th.load(shapley_save_path, weights_only=True)
    max_shapley = th.max(shapley_values)
    min_shapley = th.min(shapley_values)
    shapley_values = 2 * (shapley_values - min_shapley) / (max_shapley - min_shapley) - 1
    shapley_values = shapley_values.cpu().detach().numpy()

    plot_fn(dict, label, shapley_values, None, misc_save_path, global_step, mode, vis_mode, 'shap')


def visualize_histo_lrp(
        model,
        batch: tuple,
        misc_save_path: str,
        global_step: int,
        mode: str,
        vis_mode: str,
):
    features, label, cls, dict = batch
    features = features.squeeze(0)

    model.eval()
    # do the forward activation
    print("LRP --- Doing the forward activation...")
    features_d = features.detach().requires_grad_(True)
    act_in_pool, act_in_pool_d = features, features_d # activations before pooling

    att_scores = model.pooling.attention(features_d)
    act_out_pool, _ = model.pooling(features_d)

    act_out_pool_d = act_out_pool.detach().requires_grad_(True)
    act_in_cls, act_in_cls_d = act_out_pool, act_out_pool_d # activations before classification

    logit = model.bag_classifier(act_in_cls_d)
    act_out_cls = logit # activations after classification

    # do the lrp backpropagation
    print("LRP --- Doing the LRP backpropagation...")
    relevance_logit = logit
    Relevance = {'out': relevance_logit}

    Relevance['cls'] = lrp(Relevance['out'], act_out_cls, act_in_cls_d)
    Relevance['pool'] = lrp(Relevance['cls'], act_out_pool, act_in_pool_d)
    Relevance['pool'] = Relevance['pool'].sum(dim=1, keepdim=True).cpu().detach().numpy() # sum over the feature dim

    att_scores = att_scores.cpu().detach().numpy()
    max_ak = max(att_scores)
    min_ak = min(att_scores)
    att_scores_sign = 2 * (att_scores - min_ak) / (max_ak - min_ak) - 1
    att_scores_nosign = (att_scores - min_ak) / (max_ak - min_ak)

    plot_fn(dict, label, Relevance['pool'], None, misc_save_path, global_step, mode, vis_mode, 'lrp')
    plot_fn(dict, label, Relevance['pool'], att_scores_sign, misc_save_path, global_step, mode, vis_mode, 'lrpMULsignatt')
    plot_fn(dict, label, Relevance['pool'], att_scores_nosign, misc_save_path, global_step, mode, vis_mode, 'lrpMULnosignatt')


def lrp(relevance, layer_out_act, layer_in_act):
    rel_in_graph = (layer_out_act * (relevance / (layer_out_act + 1e-6)).detach()).sum().backward()
    relevance = layer_in_act * layer_in_act.grad
    layer_in_act.grad.zero_()
    return relevance

def plot_fn(
        dict: dict,
        label: int,
        scores: th.Tensor,
        att_scores: th.Tensor,
        misc_save_path: str,
        global_step: int,
        mode: str,
        vis_mode: str,
        xai_mode: str 
):
    
    scores = cut_off(scores, top_k=10, threshold=0.9, vis_mode=vis_mode)
    positions = [dict[i][1] for i in range(len(dict)-2)]
    patch_size_abs = [dict[i][2] for i in range(len(dict)-2)]
    original_shape = dict['original_shape']
    
    figsize = ((original_shape[0].item() / 100), (original_shape[1].item() / 100))
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("black")
    ax.set_xlim(0, original_shape[0].item())
    ax.set_ylim(0, original_shape[1].item())
    ax.invert_yaxis()
    max_ak = max(scores)
    min_ak = min(scores)
    log_norm = LogNorm(vmin=scores.min(), vmax=scores.max())
    for i, position in enumerate(positions):
        x, y = position
        x = x.item()/16
        y = y.item()/16
        patch_width = patch_size_abs[i].item()/16
        patch_height = patch_size_abs[i].item()/16
        value = scores[i]
        if vis_mode == "percentile" or vis_mode == "raw":
            value = 2 * (value - min_ak) / (max_ak - min_ak) - 1
        if vis_mode == "log":
            value = log_norm(value)
        
        value = value + att_scores[i] if att_scores is not None else value
        
        if value >= 0:
            color = plt.cm.Reds(value)
        else:
            color = plt.cm.Blues(-value)
        rect = patches.Rectangle((x, y), patch_width, patch_height, linewidth=0, edgecolor=None, facecolor=color)
        ax.add_patch(rect)
    ax.axis('off')
    if misc_save_path:
        batch_name = dict['case_name'][0]
        batch_dir = os.path.join(misc_save_path, batch_name)
        if not os.path.exists(batch_dir):
            os.makedirs(batch_dir)
        image_save_path = f"{batch_dir}/{mode}_xai_{xai_mode}_{logits_to_label(label)}_{global_step+1}_{vis_mode}.png"
        plt.savefig(image_save_path, dpi=10, bbox_inches='tight', pad_inches=0) #should be 100
        img = Image.open(image_save_path)
        img = img.resize((original_shape[0].item(), original_shape[1].item()), Image.Resampling.LANCZOS)
        img.save(image_save_path)
    plt.close()
    

def visualize_histo_patches(
        model,
        batch: tuple,
        misc_save_path: str
):
    features, label, cls, dict = batch
    positions = [dict[i][1] for i in range(len(dict)-2)]
    patch_size_abs = [dict[i][2] for i in range(len(dict)-2)]
    original_shape = dict['original_shape']

    figsize = ((original_shape[0].item() / 100), (original_shape[1].item() / 100))

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("black")
    ax.set_xlim(0, original_shape[0].item())
    ax.set_ylim(0, original_shape[1].item())
    ax.invert_yaxis()

    for i, position in enumerate(positions):
        patch_path = os.path.join("/home/space/datasets/camelyon16/patches/20x", dict['case_name'][0], f"{dict[i][0].item()}.jpg")
        if os.path.exists(patch_path):
            patch = Image.open(patch_path)
            patch_array = np.asarray(patch)
            patch_array = np.flipud(patch_array) 
            x, y = position
            x = x.item()/16
            y = y.item()/16

            patch_width = patch_size_abs[i].item()/16
            patch_height = patch_size_abs[i].item()/16
            ax.imshow(patch_array, extent=(x, (x + patch_width), y, (y + patch_height)))
            
    
    ax.axis('off')
    if misc_save_path:
        batch_name = dict['case_name'][0]
        batch_dir = os.path.join(misc_save_path, batch_name)
        if not os.path.exists(batch_dir):
            os.makedirs(batch_dir)
        image_save_path = f"{batch_dir}/all_cell_patches.png"
        if not os.path.exists(image_save_path):
            plt.savefig(image_save_path, dpi=100, bbox_inches='tight', pad_inches=0)
            img = Image.open(image_save_path)
            img = img.resize((original_shape[0].item(), original_shape[1].item()), Image.ANTIALIAS)
            img.save(image_save_path)
    plt.close()


def visualize_histo_gt(
        model,
        batch: tuple,
        misc_save_path: str
):
    features, label, cls, dict = batch
    positions = [dict[i][1] for i in range(len(dict)-2)]
    patch_size_abs = [dict[i][2] for i in range(len(dict)-2)]
    original_shape = dict['original_shape']
    annotation_array = get_tumor_annotation(dict['case_name'][0])

    figsize = ((original_shape[0].item() / 100), (original_shape[1].item() / 100))

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("black")
    ax.set_xlim(0, original_shape[0].item())
    ax.set_ylim(0, original_shape[1].item())
    ax.invert_yaxis()

    for i, position in enumerate(positions):
        patch_path = os.path.join("/home/space/datasets/camelyon16/patches/20x", dict['case_name'][0], f"{dict[i][0].item()}.jpg")
        if os.path.exists(patch_path):
            patch = Image.open(patch_path)
            patch_array = np.asarray(patch)
            patch_array = np.flipud(patch_array) 
            x, y = position
            x = x.item()/16
            y = y.item()/16

            patch_width = patch_size_abs[i].item()/16
            patch_height = patch_size_abs[i].item()/16

            annotation_region = annotation_array[
                int(y):int(y + patch_height),
                int(x):int(x + patch_width),
                :
            ]
            is_red = np.any(
                (annotation_region[:, :, 0] == 255) & 
                (annotation_region[:, :, 1] == 0) & 
                (annotation_region[:, :, 2] == 0)   
            )
            if is_red:
                ax.imshow(patch_array, extent=(x, (x + patch_width), y, (y + patch_height)))


    ax.axis('off')
    if misc_save_path:
        batch_name = dict['case_name'][0]
        batch_dir = os.path.join(misc_save_path, batch_name)
        if not os.path.exists(batch_dir):
            os.makedirs(batch_dir)
        image_save_path = f"{batch_dir}/gt_tumor_patches.png"
        if not os.path.exists(image_save_path):
            plt.savefig(image_save_path, dpi=100, bbox_inches='tight', pad_inches=0)
            img = Image.open(image_save_path)
            img = img.resize((original_shape[0].item(), original_shape[1].item()), Image.ANTIALIAS)
            img.save(image_save_path)
    plt.close()
    
    


if __name__ == "__main__":
    print("Visualizing AUC results...")
    #visualize_auc_results(10, 2, "./logs", False, True)
    #visualize_auc_results(50, 10, "./logs", False, True)
    #visualize_auc_results(100, 20, "./logs", False, True)