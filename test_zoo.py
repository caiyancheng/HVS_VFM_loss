import numpy as np
import os
# os.environ["PATH"] += os.pathsep + r"D:\App_real\Graphviz\bin"
import torch
torch.hub.set_dir(r'../Torch_hub')
from display_encoding import display_encode
display_encode_tool = display_encode(400)
import math
from Gabor_test_stimulus_generator.generate_plot_gabor_functions_new import generate_gabor_patch
from Band_limit_noise_generator.generate_plot_band_lim_noise import generate_band_lim_noise, generate_band_lim_noise_fix_random_seed
from Sinusoidal_grating_generator.generate_plot_sinusoidal_grating import generate_sinusoidal_grating
from Contrast_masking_generator.generate_plot_contrast_masking import generate_contrast_masking
from Contrast_masking_generator.generate_plot_contrast_masking_gabor_on_noise import generate_contrast_masking_gabor_on_noise
from tqdm import tqdm
import torch.nn.functional as F
import json
# from scipy.optimize import minimize, brute, differential_evolution, root_scalar
# import gc
# from scipy.stats import pearsonr, spearmanr
# from torchviz import make_dot
import cv2
import matplotlib.pyplot as plt

def convert_numpy_to_python(data):
    if isinstance(data, dict):
        return {key: convert_numpy_to_python(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_numpy_to_python(item) for item in data]
    elif isinstance(data, np.ndarray):
        return data.tolist()  # 转换为列表
    elif isinstance(data, np.float32) or isinstance(data, np.float64):
        return float(data)  # 转换为 Python 浮点数
    elif isinstance(data, np.int32) or isinstance(data, np.int64):
        return int(data)  # 转换为 Python 整数
    else:
        return data

plot_ticks_label_dict = {
            'contrast_detection_SpF_Gabor_ach': [[0.5, 1, 2, 4, 8, 16, 32], [1, 10, 100, 1000], 'Spatial Frequency (cpd)', 'Sensitivity'],
            'contrast_detection_SpF_Noise_ach': [[0.5, 1, 2, 4, 8, 16, 32], [1, 10, 100, 1000], 'Spatial Frequency (cpd)', 'Sensitivity'],
            'contrast_detection_SpF_Gabor_RG': [[0.5, 1, 2, 4, 8, 16, 32], [round(1/0.12,2), 10, 100, 1000], 'Spatial Frequency (cpd)', 'Sensitivity'],
            'contrast_detection_SpF_Gabor_YV': [[0.5, 1, 2, 4, 8, 16, 32], [round(1/0.8,2), 10, 100, 1000], 'Spatial Frequency (cpd)', 'Sensitivity'],
            'contrast_detection_luminance': [[0.1, 1, 10, 100], [1, 10, 100, 1000], 'Luminance (cd/m$^2$)', 'Sensitivity'],
            'contrast_detection_area': [[0.1, 1], [1, 10, 100, 1000], 'Area (degree$^2$)', 'Sensitivity'],
            'contrast_masking_phase_coherent_masking': [[0.01, 0.1], [0.01, 0.1], 'Mask Contrast', 'Test Contrast'],
            'contrast_masking_phase_incoherent_masking': [[0.01, 0.1], [0.01, 0.1], 'Mask Contrast', 'Test Contrast'],
        }
class Contrast_Detection_Area:
    def __init__(self, sample_num):
        self.W = 224
        self.H = 224
        self.sample_num = sample_num
        R_min = 0.1
        R_max = 1
        self.Area_list = np.logspace(np.log10(math.pi * R_min ** 2), np.log10(math.pi * R_max ** 2), self.sample_num)
        self.R_list = (self.Area_list / math.pi) ** 0.5
        self.rho = 2 #8
        self.contrast_list = np.logspace(np.log10(0.001), np.log10(1), self.sample_num)
        self.O = 0
        self.L_b = 100
        self.ppd = 60
        self.test_type = 'Contrast Detection - Area'
        self.test_short_name = 'contrast_detection_area'
        self.multiplier_list = np.logspace(np.log10(0.5), np.log10(2), 5)
        castleCSF_result_json = rf'Matlab_CSF_plot/castleCSF_area_sensitivity_{self.rho}cpd.json'
        with open(castleCSF_result_json, 'r') as fp:
            castleCSF_result_data = json.load(fp)
        self.castleCSF_result_area_list = castleCSF_result_data['area_list']
        self.castleCSF_result_sensitivity_list = castleCSF_result_data['sensitivity_list']

    def test_models(self, model, resolution=None):
        # Compute the Spearman Correlation
        Spearman_matrix_score = torch.zeros(len(self.Area_list), len(self.multiplier_list), device='cuda')
        for Area_index, Area_value in enumerate(self.Area_list):
            S_gt = np.interp(Area_value, self.castleCSF_result_area_list, self.castleCSF_result_sensitivity_list)
            for multiplier_index, multiplier_value in enumerate(self.multiplier_list):
                S_test = multiplier_value * S_gt
                T_L_array, R_L_array = generate_gabor_patch(W=self.W, H=self.H, R=(Area_value / math.pi) ** 0.5,
                                                            rho=self.rho, O=self.O, L_b=self.L_b,
                                                            contrast=1 / S_test, ppd=self.ppd,
                                                            color_direction='ach')
                if resolution is not None:
                    T_L_array = cv2.resize(T_L_array, (resolution[0], resolution[1]), interpolation=cv2.INTER_LINEAR)
                    R_L_array = cv2.resize(R_L_array, (resolution[0], resolution[1]), interpolation=cv2.INTER_LINEAR)
                T_C_array = display_encode_tool.L2C_sRGB(T_L_array)
                T_C_tensor = torch.tensor(T_C_array, dtype=torch.float32).permute(2, 0, 1)[None, ...].cuda()
                R_C_array = display_encode_tool.L2C_sRGB(R_L_array)
                R_C_tensor = torch.tensor(R_C_array, dtype=torch.float32).permute(2, 0, 1)[None, ...].cuda()
                test_feature = model(T_C_tensor)
                reference_feature = model(R_C_tensor)
                cos_similarity = F.cosine_similarity(
                    test_feature.reshape(1, -1),
                    reference_feature.reshape(1, -1)
                )
                cos_similarity = torch.clamp(cos_similarity, -1 + 1e-6, 1 - 1e-6)
                Spearman_score = torch.arccos(cos_similarity) / math.pi  # shape: [1]
                Spearman_matrix_score[Area_index, multiplier_index] = Spearman_score
        multiplier_tensor = torch.tensor(self.multiplier_list, dtype=torch.float32, device='cuda')
        X = multiplier_tensor.repeat(len(self.Area_list))  # shape: [N * M]
        Y = Spearman_matrix_score.flatten()
        X_mean = torch.mean(X)
        Y_mean = torch.mean(Y)
        X_centered = X - X_mean
        Y_centered = Y - Y_mean
        numerator = torch.sum(X_centered * Y_centered)
        denominator = torch.sqrt(torch.sum(X_centered ** 2) * torch.sum(Y_centered ** 2) + 1e-8)
        correlation_pearson = - numerator / denominator
        loss = 1 - correlation_pearson
        # make_dot(loss, params=dict(model.named_parameters())).render("computation_graph", format="pdf")
        return loss

    def test_models_plot_contours(self, model_name, model, suffix, epoch, resolution=None):
        with torch.no_grad():
            radius_matrix = np.zeros([len(self.R_list), len(self.contrast_list)])
            area_matrix = np.zeros([len(self.R_list), len(self.contrast_list)])
            contrast_matrix = np.zeros([len(self.R_list), len(self.contrast_list)])
            arccos_cos_similarity_matrix = np.zeros([len(self.R_list), len(self.contrast_list)])

            for R_index in range(len(self.R_list)):
                R_value = self.R_list[R_index]
                A_value = self.Area_list[R_index]
                for contrast_index in range(len(self.contrast_list)):
                    contrast_value = self.contrast_list[contrast_index]
                    T_L_array, R_L_array = generate_gabor_patch(W=self.W, H=self.H, R=R_value, rho=self.rho,
                                                                O=self.O,
                                                                L_b=self.L_b, contrast=contrast_value, ppd=self.ppd,
                                                                color_direction='ach')
                    radius_matrix[R_index, contrast_index] = R_value
                    area_matrix[R_index, contrast_index] = A_value
                    contrast_matrix[R_index, contrast_index] = contrast_value
                    if resolution is not None:
                        T_L_array = cv2.resize(T_L_array, (resolution[0], resolution[1]),
                                               interpolation=cv2.INTER_LINEAR)
                        R_L_array = cv2.resize(R_L_array, (resolution[0], resolution[1]),
                                               interpolation=cv2.INTER_LINEAR)
                    T_C_array = display_encode_tool.L2C_sRGB(T_L_array)
                    T_C_tensor = torch.tensor(T_C_array, dtype=torch.float32).permute(2, 0, 1)[None, ...].cuda()
                    R_C_array = display_encode_tool.L2C_sRGB(R_L_array)
                    R_C_tensor = torch.tensor(R_C_array, dtype=torch.float32).permute(2, 0, 1)[None, ...].cuda()
                    test_feature = model(T_C_tensor)
                    reference_feature = model(R_C_tensor)
                    cos_similarity = F.cosine_similarity(
                        test_feature.reshape(1, -1),
                        reference_feature.reshape(1, -1)
                    )
                    cos_similarity = torch.clamp(cos_similarity, -1 + 1e-6, 1 - 1e-6)
                    arccos_cos_similarity_matrix[R_index, contrast_index] = np.arccos(np.array(cos_similarity.cpu())) / np.arccos(-1)

        plot_ticks_label_list = plot_ticks_label_dict[self.test_short_name]
        X_ticks = plot_ticks_label_list[0]
        Y_ticks = plot_ticks_label_list[1]
        X_label = plot_ticks_label_list[2]
        Y_label = plot_ticks_label_list[3]
        plot_X_matrix = area_matrix
        plot_Y_matrix = 1 / contrast_matrix
        plot_score_matrix = arccos_cos_similarity_matrix
        gt_result_X_list = self.castleCSF_result_area_list
        gt_result_Y_list = self.castleCSF_result_sensitivity_list
        plot_figure_name = f'{model_name}_arccos_epoch{epoch}'
        if resolution is not None:
            plot_figure_name += f'_{resolution[0]}_{resolution[1]}'
        plt.figure(figsize=(4.3,3), dpi=300)
        levels = np.linspace(0, 1, 200)
        plt.contourf(plot_X_matrix, plot_Y_matrix, plot_score_matrix,
                     levels=levels, cmap='rainbow', alpha=0.3)
        plt.contour(plot_X_matrix, plot_Y_matrix, plot_score_matrix,
                    levels=levels, cmap='rainbow', linewidths=1)
        plt.plot(gt_result_X_list, gt_result_Y_list, 'k', linestyle='--', linewidth=2,
                 label='castleCSF prediction')
        plt.xlim([plot_X_matrix.min(), plot_X_matrix.max()])
        plt.ylim([plot_Y_matrix.min(), plot_Y_matrix.max()])
        plt.xlabel(X_label, fontsize=12)
        plt.ylabel(Y_label, fontsize=12)
        plt.xscale('log')
        plt.yscale('log')
        plt.xticks(X_ticks, X_ticks)
        plt.yticks(Y_ticks, Y_ticks)
        plt.tight_layout()
        plt.legend(loc='lower right')
        save_figure_dir = os.path.join(f'plot_contours/test_{suffix}/test_{self.test_short_name}', model_name)
        os.makedirs(save_figure_dir, exist_ok=True)
        plt.savefig(os.path.join(save_figure_dir, plot_figure_name + '.png'), dpi=300,
                    bbox_inches='tight', pad_inches=0.02)
        plt.close()



class Contrast_Detection_Luminance:
    def __init__(self, sample_num):
        self.W = 224
        self.H = 224
        self.sample_num = sample_num
        self.R = 1
        self.rho = 1 #2
        self.contrast_list = np.logspace(np.log10(0.001), np.log10(1), self.sample_num)
        self.O = 0
        self.L_b_list = np.logspace(np.log10(0.1), np.log10(200), self.sample_num)
        self.ppd = 60
        self.test_type = 'Contrast Detection - Luminance'
        self.test_short_name = 'contrast_detection_luminance'
        self.multiplier_list = np.logspace(np.log10(0.5), np.log10(2), 5)
        castleCSF_result_json = rf'Matlab_CSF_plot/castleCSF_luminance_sensitivity_{self.rho}cpd.json'
        with open(castleCSF_result_json, 'r') as fp:
            castleCSF_result_data = json.load(fp)
        self.castleCSF_result_L_list = castleCSF_result_data['luminance_list']
        self.castleCSF_result_sensitivity_list = castleCSF_result_data['sensitivity_list']

    def test_models(self, model, resolution=None):
        # Compute the Spearman Correlation
        Spearman_matrix_score = torch.zeros(len(self.L_b_list), len(self.multiplier_list), device='cuda')
        for L_b_index, L_b_value in enumerate(self.L_b_list):
            S_gt = np.interp(L_b_value, self.castleCSF_result_L_list, self.castleCSF_result_sensitivity_list)
            for multiplier_index, multiplier_value in enumerate(self.multiplier_list):
                S_test = multiplier_value * S_gt
                T_L_array, R_L_array = generate_gabor_patch(W=self.W, H=self.H, R=self.R,
                                                            rho=self.rho, O=self.O, L_b=L_b_value,
                                                            contrast=1 / S_test, ppd=self.ppd,
                                                            color_direction='ach')
                if resolution is not None:
                    T_L_array = cv2.resize(T_L_array, (resolution[0], resolution[1]), interpolation=cv2.INTER_LINEAR)
                    R_L_array = cv2.resize(R_L_array, (resolution[0], resolution[1]), interpolation=cv2.INTER_LINEAR)
                T_C_array = display_encode_tool.L2C_sRGB(T_L_array)
                T_C_tensor = torch.tensor(T_C_array, dtype=torch.float32).permute(2, 0, 1)[None, ...].cuda()
                R_C_array = display_encode_tool.L2C_sRGB(R_L_array)
                R_C_tensor = torch.tensor(R_C_array, dtype=torch.float32).permute(2, 0, 1)[None, ...].cuda()
                test_feature = model(T_C_tensor)
                reference_feature = model(R_C_tensor)
                cos_similarity = F.cosine_similarity(
                    test_feature.reshape(1, -1),
                    reference_feature.reshape(1, -1)
                )
                cos_similarity = torch.clamp(cos_similarity, -1 + 1e-6, 1 - 1e-6)
                Spearman_score = torch.arccos(cos_similarity) / math.pi  # shape: [1]
                Spearman_matrix_score[L_b_index, multiplier_index] = Spearman_score
        multiplier_tensor = torch.tensor(self.multiplier_list, dtype=torch.float32, device='cuda')
        X = multiplier_tensor.repeat(len(self.L_b_list))  # shape: [N * M]
        Y = Spearman_matrix_score.flatten()
        X_mean = torch.mean(X)
        Y_mean = torch.mean(Y)
        X_centered = X - X_mean
        Y_centered = Y - Y_mean
        numerator = torch.sum(X_centered * Y_centered)
        denominator = torch.sqrt(torch.sum(X_centered ** 2) * torch.sum(Y_centered ** 2) + 1e-8)
        correlation_pearson = - numerator / denominator
        loss = 1 - correlation_pearson
        # make_dot(loss, params=dict(model.named_parameters())).render("computation_graph", format="pdf")
        return loss

    def test_models_plot_contours(self, model_name, model, suffix, epoch, resolution=None):
        with torch.no_grad():
            L_b_matrix = np.zeros([len(self.L_b_list), len(self.contrast_list)])
            contrast_matrix = np.zeros([len(self.L_b_list), len(self.contrast_list)])
            arccos_cos_similarity_matrix = np.zeros([len(self.L_b_list), len(self.contrast_list)])

            for L_b_index in range(len(self.L_b_list)):
                L_b_value = self.L_b_list[L_b_index]
                for contrast_index in range(len(self.contrast_list)):
                    contrast_value = self.contrast_list[contrast_index]
                    T_L_array, R_L_array = generate_gabor_patch(W=self.W, H=self.H, R=self.R, rho=self.rho,
                                                                O=self.O,
                                                                L_b=L_b_value, contrast=contrast_value, ppd=self.ppd,
                                                                color_direction='ach')
                    L_b_matrix[L_b_index, contrast_index] = L_b_value
                    contrast_matrix[L_b_index, contrast_index] = contrast_value
                    if resolution is not None:
                        T_L_array = cv2.resize(T_L_array, (resolution[0], resolution[1]),
                                               interpolation=cv2.INTER_LINEAR)
                        R_L_array = cv2.resize(R_L_array, (resolution[0], resolution[1]),
                                               interpolation=cv2.INTER_LINEAR)
                    T_C_array = display_encode_tool.L2C_sRGB(T_L_array)
                    T_C_tensor = torch.tensor(T_C_array, dtype=torch.float32).permute(2, 0, 1)[None, ...].cuda()
                    R_C_array = display_encode_tool.L2C_sRGB(R_L_array)
                    R_C_tensor = torch.tensor(R_C_array, dtype=torch.float32).permute(2, 0, 1)[None, ...].cuda()
                    test_feature = model(T_C_tensor)
                    reference_feature = model(R_C_tensor)
                    cos_similarity = F.cosine_similarity(
                        test_feature.reshape(1, -1),
                        reference_feature.reshape(1, -1)
                    )
                    cos_similarity = torch.clamp(cos_similarity, -1 + 1e-6, 1 - 1e-6)
                    arccos_cos_similarity_matrix[L_b_index, contrast_index] = np.arccos(np.array(cos_similarity.cpu())) / np.arccos(-1)

        plot_ticks_label_list = plot_ticks_label_dict[self.test_short_name]
        X_ticks = plot_ticks_label_list[0]
        Y_ticks = plot_ticks_label_list[1]
        X_label = plot_ticks_label_list[2]
        Y_label = plot_ticks_label_list[3]
        plot_X_matrix = L_b_matrix
        plot_Y_matrix = 1 / contrast_matrix
        plot_score_matrix = arccos_cos_similarity_matrix
        gt_result_X_list = self.castleCSF_result_L_list
        gt_result_Y_list = self.castleCSF_result_sensitivity_list
        plot_figure_name = f'{model_name}_arccos_epoch{epoch}'
        if resolution is not None:
            plot_figure_name += f'_{resolution[0]}_{resolution[1]}'
        plt.figure(figsize=(4.3,3), dpi=300)
        levels = np.linspace(0, 1, 200)
        plt.contourf(plot_X_matrix, plot_Y_matrix, plot_score_matrix,
                     levels=levels, cmap='rainbow', alpha=0.3)
        plt.contour(plot_X_matrix, plot_Y_matrix, plot_score_matrix,
                    levels=levels, cmap='rainbow', linewidths=1)
        plt.plot(gt_result_X_list, gt_result_Y_list, 'k', linestyle='--', linewidth=2,
                 label='castleCSF prediction')
        plt.xlim([plot_X_matrix.min(), plot_X_matrix.max()])
        plt.ylim([plot_Y_matrix.min(), plot_Y_matrix.max()])
        plt.xlabel(X_label, fontsize=12)
        plt.ylabel(Y_label, fontsize=12)
        plt.xscale('log')
        plt.yscale('log')
        plt.xticks(X_ticks, X_ticks)
        plt.yticks(Y_ticks, Y_ticks)
        plt.tight_layout()
        plt.legend(loc='lower right')
        save_figure_dir = os.path.join(f'plot_contours/test_{suffix}/test_{self.test_short_name}', model_name)
        os.makedirs(save_figure_dir, exist_ok=True)
        plt.savefig(os.path.join(save_figure_dir, plot_figure_name + '.png'), dpi=300,
                    bbox_inches='tight', pad_inches=0.02)
        plt.close()

class Contrast_Detection_SpF_Gabor_Ach:
    def __init__(self, sample_num):
        self.W = 224
        self.H = 224
        self.sample_num = sample_num
        self.R = 1
        self.rho_list = np.logspace(np.log10(0.5), np.log10(8), self.sample_num)
        self.contrast_list = np.logspace(np.log10(0.001), np.log10(1), self.sample_num)
        self.O = 0
        self.L_b = 100
        self.ppd = 60
        self.test_type = 'Contrast Detection - SpF_Gabor_ach'
        self.test_short_name = 'contrast_detection_SpF_Gabor_ach'
        self.multiplier_list = np.logspace(np.log10(0.5), np.log10(2), 10)
        castleCSF_result_json = rf'Matlab_CSF_plot/castleCSF_rho_sensitivity_data.json'
        with open(castleCSF_result_json, 'r') as fp:
            castleCSF_result_data = json.load(fp)
        self.castleCSF_result_rho_list = castleCSF_result_data['rho_list']
        self.castleCSF_result_sensitivity_list = castleCSF_result_data['sensitivity_list']

    def test_models(self, model, resolution=None):
        Spearman_matrix_score = torch.zeros(len(self.rho_list), len(self.multiplier_list), device='cuda')
        for rho_index, rho_value in enumerate(self.rho_list):
            S_gt = np.interp(rho_value, self.castleCSF_result_rho_list, self.castleCSF_result_sensitivity_list)
            for multiplier_index, multiplier_value in enumerate(self.multiplier_list):
                S_test = multiplier_value * S_gt
                T_L_array, R_L_array = generate_gabor_patch(W=self.W, H=self.H, R=self.R,
                                                            rho=rho_value, O=self.O, L_b=self.L_b,
                                                            contrast=1 / S_test, ppd=self.ppd,
                                                            color_direction='ach')
                if resolution is not None:
                    T_L_array = cv2.resize(T_L_array, (resolution[0], resolution[1]), interpolation=cv2.INTER_LINEAR)
                    R_L_array = cv2.resize(R_L_array, (resolution[0], resolution[1]), interpolation=cv2.INTER_LINEAR)
                T_C_array = display_encode_tool.L2C_sRGB(T_L_array)
                T_C_tensor = torch.tensor(T_C_array, dtype=torch.float32).permute(2, 0, 1)[None, ...].cuda()
                R_C_array = display_encode_tool.L2C_sRGB(R_L_array)
                R_C_tensor = torch.tensor(R_C_array, dtype=torch.float32).permute(2, 0, 1)[None, ...].cuda()
                test_feature = model(T_C_tensor)
                reference_feature = model(R_C_tensor)
                cos_similarity = F.cosine_similarity(
                    test_feature.reshape(1, -1),
                    reference_feature.reshape(1, -1)
                )
                cos_similarity = torch.clamp(cos_similarity, -1 + 1e-6, 1 - 1e-6)
                Spearman_score = torch.arccos(cos_similarity) / math.pi  # shape: [1]
                Spearman_matrix_score[rho_index, multiplier_index] = Spearman_score
        multiplier_tensor = torch.tensor(self.multiplier_list, dtype=torch.float32, device='cuda')
        X = multiplier_tensor.repeat(len(self.rho_list))  # shape: [N * M]
        Y = Spearman_matrix_score.flatten()
        X_mean = torch.mean(X)
        Y_mean = torch.mean(Y)
        X_centered = X - X_mean
        Y_centered = Y - Y_mean
        numerator = torch.sum(X_centered * Y_centered)
        denominator = torch.sqrt(torch.sum(X_centered ** 2) * torch.sum(Y_centered ** 2) + 1e-8)
        correlation_pearson = - numerator / denominator
        loss = 1 - correlation_pearson
        # make_dot(loss, params=dict(model.named_parameters())).render("computation_graph", format="pdf")
        return loss

    def test_models_plot_contours(self, model_name, model, suffix, epoch, resolution=None):
        with torch.no_grad():
            rho_matrix = np.zeros([len(self.rho_list), len(self.contrast_list)])
            contrast_matrix = np.zeros([len(self.rho_list), len(self.contrast_list)])
            arccos_cos_similarity_matrix = np.zeros([len(self.rho_list), len(self.contrast_list)])

            for rho_index in range(len(self.rho_list)):
                rho_value = self.rho_list[rho_index]
                for contrast_index in range(len(self.contrast_list)):
                    contrast_value = self.contrast_list[contrast_index]
                    T_L_array, R_L_array = generate_gabor_patch(W=self.W, H=self.H, R=self.R, rho=rho_value,
                                                                O=self.O,
                                                                L_b=self.L_b, contrast=contrast_value, ppd=self.ppd,
                                                                color_direction='ach')
                    rho_matrix[rho_index, contrast_index] = rho_value
                    contrast_matrix[rho_index, contrast_index] = contrast_value
                    if resolution is not None:
                        T_L_array = cv2.resize(T_L_array, (resolution[0], resolution[1]),
                                               interpolation=cv2.INTER_LINEAR)
                        R_L_array = cv2.resize(R_L_array, (resolution[0], resolution[1]),
                                               interpolation=cv2.INTER_LINEAR)
                    T_C_array = display_encode_tool.L2C_sRGB(T_L_array)
                    T_C_tensor = torch.tensor(T_C_array, dtype=torch.float32).permute(2, 0, 1)[None, ...].cuda()
                    R_C_array = display_encode_tool.L2C_sRGB(R_L_array)
                    R_C_tensor = torch.tensor(R_C_array, dtype=torch.float32).permute(2, 0, 1)[None, ...].cuda()
                    test_feature = model(T_C_tensor)
                    reference_feature = model(R_C_tensor)
                    cos_similarity = F.cosine_similarity(
                        test_feature.reshape(1, -1),
                        reference_feature.reshape(1, -1)
                    )
                    cos_similarity = torch.clamp(cos_similarity, -1 + 1e-6, 1 - 1e-6)
                    arccos_cos_similarity_matrix[rho_index, contrast_index] = np.arccos(np.array(cos_similarity.cpu())) / np.arccos(-1)

        plot_ticks_label_list = plot_ticks_label_dict[self.test_short_name]
        X_ticks = plot_ticks_label_list[0]
        Y_ticks = plot_ticks_label_list[1]
        X_label = plot_ticks_label_list[2]
        Y_label = plot_ticks_label_list[3]
        plot_X_matrix = rho_matrix
        plot_Y_matrix = 1 / contrast_matrix
        plot_score_matrix = arccos_cos_similarity_matrix
        gt_result_X_list = self.castleCSF_result_rho_list
        gt_result_Y_list = self.castleCSF_result_sensitivity_list
        plot_figure_name = f'{model_name}_arccos_epoch{epoch}'
        if resolution is not None:
            plot_figure_name += f'_{resolution[0]}_{resolution[1]}'
        plt.figure(figsize=(4.3,3), dpi=300)
        levels = np.linspace(0, 1, 200)
        plt.contourf(plot_X_matrix, plot_Y_matrix, plot_score_matrix,
                     levels=levels, cmap='rainbow', alpha=0.3)
        plt.contour(plot_X_matrix, plot_Y_matrix, plot_score_matrix,
                    levels=levels, cmap='rainbow', linewidths=1)
        plt.plot(gt_result_X_list, gt_result_Y_list, 'k', linestyle='--', linewidth=2,
                 label='castleCSF prediction')
        plt.xlim([plot_X_matrix.min(), plot_X_matrix.max()])
        plt.ylim([plot_Y_matrix.min(), plot_Y_matrix.max()])
        plt.xlabel(X_label, fontsize=12)
        plt.ylabel(Y_label, fontsize=12)
        plt.xscale('log')
        plt.yscale('log')
        plt.xticks(X_ticks, X_ticks)
        plt.yticks(Y_ticks, Y_ticks)
        plt.tight_layout()
        plt.legend(loc='lower right')
        save_figure_dir = os.path.join(f'plot_contours/test_{suffix}/test_{self.test_short_name}', model_name)
        os.makedirs(save_figure_dir, exist_ok=True)
        plt.savefig(os.path.join(save_figure_dir, plot_figure_name + '.png'), dpi=300,
                    bbox_inches='tight', pad_inches=0.02)
        plt.close()

class Contrast_Masking_Phase_Coherent:
    def __init__(self, sample_num):
        self.W = 224
        self.H = 224
        self.sample_num = sample_num
        self.rho = 2
        self.O = 0
        self.L_b = 32
        self.contrast_mask_list = np.logspace(np.log10(0.005), np.log10(0.5), self.sample_num)
        self.contrast_test_list = np.logspace(np.log10(0.01), np.log10(0.5), self.sample_num)
        self.ppd = 60
        self.R = 0.5
        self.test_type = 'Contrast Masking - Phase-Coherent Masking'
        self.test_short_name = 'contrast_masking_phase_coherent_masking'
        self.multiplier_list = np.logspace(np.log10(0.5), np.log10(2), 5)
        foley_result_json = r'Matlab_CSF_plot/foley_contrast_masking_data_gabor.json'
        with open(foley_result_json, 'r') as fp:
            foley_result_data = json.load(fp)
        foley_result_x_mask_contrast_list = np.array(foley_result_data['mask_contrast_list'])
        foley_result_y_test_contrast_list = np.array(foley_result_data['test_contrast_list'])
        valid_gt_indices = [index for index, value in enumerate(foley_result_x_mask_contrast_list) if
                            value > self.contrast_mask_list.min() and value < 0.25]
        self.gt_x_mask_C = foley_result_x_mask_contrast_list[valid_gt_indices]
        self.gt_y_test_C = foley_result_y_test_contrast_list[valid_gt_indices]

    def test_models(self, model, resolution=None):
        Spearman_matrix_score = torch.zeros(len(self.gt_x_mask_C), len(self.multiplier_list), device='cuda')
        for contrast_mask_index in range(len(self.gt_x_mask_C)):
            contrast_mask_value = self.gt_x_mask_C[contrast_mask_index]
            contrast_test_value = self.gt_y_test_C[contrast_mask_index]
            for multiplier_index, multiplier_value in enumerate(self.multiplier_list):
                C_test = contrast_test_value * multiplier_value
                T_L_array, R_L_array = generate_contrast_masking(W=self.W, H=self.H, rho=self.rho, O=self.O,
                                                                 L_b=self.L_b, contrast_mask=contrast_mask_value,
                                                                 contrast_test=C_test, ppd=self.ppd,
                                                                 gabor_radius=self.R)
                T_L_array = np.stack([T_L_array] * 3, axis=-1)
                R_L_array = np.stack([R_L_array] * 3, axis=-1)
                if resolution is not None:
                    T_L_array = cv2.resize(T_L_array, (resolution[0], resolution[1]), interpolation=cv2.INTER_LINEAR)
                    R_L_array = cv2.resize(R_L_array, (resolution[0], resolution[1]), interpolation=cv2.INTER_LINEAR)
                T_C_array = display_encode_tool.L2C_sRGB(T_L_array)
                T_C_tensor = torch.tensor(T_C_array, dtype=torch.float32).permute(2, 0, 1)[None, ...].cuda()
                R_C_array = display_encode_tool.L2C_sRGB(R_L_array)
                R_C_tensor = torch.tensor(R_C_array, dtype=torch.float32).permute(2, 0, 1)[None, ...].cuda()
                test_feature = model(T_C_tensor)
                reference_feature = model(R_C_tensor)
                cos_similarity = F.cosine_similarity(
                    test_feature.reshape(1, -1),
                    reference_feature.reshape(1, -1)
                )
                cos_similarity = torch.clamp(cos_similarity, -1 + 1e-6, 1 - 1e-6)
                Spearman_score = torch.arccos(cos_similarity) / math.pi  # shape: [1]
                Spearman_matrix_score[contrast_mask_index, multiplier_index] = Spearman_score
        multiplier_tensor = torch.tensor(self.multiplier_list, dtype=torch.float32, device='cuda')
        X = multiplier_tensor.repeat(len(self.gt_x_mask_C))  # shape: [N * M]
        Y = Spearman_matrix_score.flatten()
        X_mean = torch.mean(X)
        Y_mean = torch.mean(Y)
        X_centered = X - X_mean
        Y_centered = Y - Y_mean
        numerator = torch.sum(X_centered * Y_centered)
        denominator = torch.sqrt(torch.sum(X_centered ** 2) * torch.sum(Y_centered ** 2) + 1e-8)
        correlation_pearson = numerator / denominator
        loss = 1 - correlation_pearson
        # make_dot(loss, params=dict(model.named_parameters())).render("computation_graph", format="pdf")
        return loss

    def test_models_plot_contours(self, model_name, model, suffix, epoch, resolution=None):
        with torch.no_grad():
            contrast_mask_matrix = np.zeros([len(self.contrast_mask_list), len(self.contrast_test_list)])
            contrast_test_matrix = np.zeros([len(self.contrast_mask_list), len(self.contrast_test_list)])
            arccos_cos_similarity_matrix = np.zeros([len(self.contrast_mask_list), len(self.contrast_test_list)])

            for contrast_mask_index in range(len(self.contrast_mask_list)):
                contrast_mask_value = self.contrast_mask_list[contrast_mask_index]
                for contrast_test_index in range(len(self.contrast_test_list)):
                    contrast_test_value = self.contrast_test_list[contrast_test_index]
                    T_L_array, R_L_array = generate_contrast_masking(W=self.W, H=self.H, rho=self.rho, O=self.O,
                                                                     L_b=self.L_b, contrast_mask=contrast_mask_value,
                                                                     contrast_test=contrast_test_value, ppd=self.ppd,
                                                                     gabor_radius=self.R)
                    T_L_array = np.stack([T_L_array] * 3, axis=-1)
                    R_L_array = np.stack([R_L_array] * 3, axis=-1)
                    contrast_mask_matrix[contrast_mask_index, contrast_test_index] = contrast_mask_value
                    contrast_test_matrix[contrast_mask_index, contrast_test_index] = contrast_test_value
                    if resolution is not None:
                        T_L_array = cv2.resize(T_L_array, (resolution[0], resolution[1]),
                                               interpolation=cv2.INTER_LINEAR)
                        R_L_array = cv2.resize(R_L_array, (resolution[0], resolution[1]),
                                               interpolation=cv2.INTER_LINEAR)
                    T_C_array = display_encode_tool.L2C_sRGB(T_L_array)
                    T_C_tensor = torch.tensor(T_C_array, dtype=torch.float32).permute(2, 0, 1)[None, ...].cuda()
                    R_C_array = display_encode_tool.L2C_sRGB(R_L_array)
                    R_C_tensor = torch.tensor(R_C_array, dtype=torch.float32).permute(2, 0, 1)[None, ...].cuda()
                    test_feature = model(T_C_tensor)
                    reference_feature = model(R_C_tensor)
                    cos_similarity = F.cosine_similarity(
                        test_feature.reshape(1, -1),
                        reference_feature.reshape(1, -1)
                    )
                    cos_similarity = torch.clamp(cos_similarity, -1 + 1e-6, 1 - 1e-6)
                    arccos_cos_similarity_matrix[contrast_mask_index, contrast_test_index] = np.arccos(np.array(cos_similarity.cpu())) / np.arccos(-1)

        plot_ticks_label_list = plot_ticks_label_dict[self.test_short_name]
        X_ticks = plot_ticks_label_list[0]
        Y_ticks = plot_ticks_label_list[1]
        X_label = plot_ticks_label_list[2]
        Y_label = plot_ticks_label_list[3]
        plot_X_matrix = contrast_mask_matrix
        plot_Y_matrix = contrast_test_matrix
        plot_score_matrix = arccos_cos_similarity_matrix
        gt_result_X_list = self.gt_x_mask_C
        gt_result_Y_list = self.gt_y_test_C
        plot_figure_name = f'{model_name}_arccos_epoch{epoch}'
        if resolution is not None:
            plot_figure_name += f'_{resolution[0]}_{resolution[1]}'
        plt.figure(figsize=(4.3,3), dpi=300)
        levels = np.linspace(0, 1, 200)
        plt.contourf(plot_X_matrix, plot_Y_matrix, plot_score_matrix,
                     levels=levels, cmap='rainbow', alpha=0.3)
        plt.contour(plot_X_matrix, plot_Y_matrix, plot_score_matrix,
                    levels=levels, cmap='rainbow', linewidths=1)
        plt.plot(gt_result_X_list, gt_result_Y_list, 'k', linestyle='--', linewidth=2,
                 label='Human Results', marker='o')
        plt.xlim([plot_X_matrix.min(), plot_X_matrix.max()])
        plt.ylim([plot_Y_matrix.min(), plot_Y_matrix.max()])
        plt.xlabel(X_label, fontsize=12)
        plt.ylabel(Y_label, fontsize=12)
        plt.xscale('log')
        plt.yscale('log')
        plt.xticks(X_ticks, X_ticks)
        plt.yticks(Y_ticks, Y_ticks)
        plt.tight_layout()
        plt.legend(loc='lower right')
        save_figure_dir = os.path.join(f'plot_contours/test_{suffix}/test_{self.test_short_name}', model_name)
        os.makedirs(save_figure_dir, exist_ok=True)
        plt.savefig(os.path.join(save_figure_dir, plot_figure_name + '.png'), dpi=300,
                    bbox_inches='tight', pad_inches=0.02)
        plt.close()

class Contrast_Masking_Phase_Incoherent:
    def __init__(self, sample_num):
        self.W = 224
        self.H = 224
        self.sample_num = sample_num
        self.Mask_upper_frequency = 12
        self.L_b = 37
        self.contrast_mask_list = np.logspace(np.log10(0.005), np.log10(0.5), self.sample_num)
        self.contrast_test_list = np.logspace(np.log10(0.01), np.log10(0.5), self.sample_num)
        self.ppd = 60
        self.R = 0.8
        self.rho_test = 1.2
        self.test_type = 'Contrast Masking - Phase-Incoherent Masking'
        self.test_short_name = 'contrast_masking_phase_incoherent_masking'
        self.multiplier_list = np.logspace(np.log10(0.5), np.log10(2), 10)
        foley_result_json = r'Matlab_CSF_plot/contrast_masking_data_gabor_on_noise.json'
        with open(foley_result_json, 'r') as fp:
            foley_result_data = json.load(fp)
        foley_result_x_mask_contrast_list = np.array(foley_result_data['mask_contrast_list'])
        foley_result_y_test_contrast_list = np.array(foley_result_data['test_contrast_list'])
        valid_gt_indices = [index for index, value in enumerate(foley_result_x_mask_contrast_list) if
                            value > self.contrast_mask_list.min() and value < 0.25]
        self.gt_x_mask_C = foley_result_x_mask_contrast_list[valid_gt_indices]
        self.gt_y_test_C = foley_result_y_test_contrast_list[valid_gt_indices]

    def test_models(self, model, resolution=None):
        Spearman_matrix_score = torch.zeros(len(self.gt_x_mask_C), len(self.multiplier_list), device='cuda')
        for contrast_mask_index in range(len(self.gt_x_mask_C)):
            contrast_mask_value = self.gt_x_mask_C[contrast_mask_index]
            contrast_test_value = self.gt_y_test_C[contrast_mask_index]
            for multiplier_index, multiplier_value in enumerate(self.multiplier_list):
                C_test = contrast_test_value * multiplier_value
                T_L_array, R_L_array = generate_contrast_masking_gabor_on_noise(W=self.W, H=self.H,
                                                                                sigma=self.R,
                                                                                rho=self.rho_test,
                                                                                Mask_upper_frequency=self.Mask_upper_frequency,
                                                                                L_b=self.L_b,
                                                                                contrast_mask=contrast_mask_value,
                                                                                contrast_test=C_test,
                                                                                ppd=self.ppd)
                T_L_array = np.stack([T_L_array] * 3, axis=-1)
                R_L_array = np.stack([R_L_array] * 3, axis=-1)
                if resolution is not None:
                    T_L_array = cv2.resize(T_L_array, (resolution[0], resolution[1]), interpolation=cv2.INTER_LINEAR)
                    R_L_array = cv2.resize(R_L_array, (resolution[0], resolution[1]), interpolation=cv2.INTER_LINEAR)
                T_C_array = display_encode_tool.L2C_sRGB(T_L_array)
                T_C_tensor = torch.tensor(T_C_array, dtype=torch.float32).permute(2, 0, 1)[None, ...].cuda()
                R_C_array = display_encode_tool.L2C_sRGB(R_L_array)
                R_C_tensor = torch.tensor(R_C_array, dtype=torch.float32).permute(2, 0, 1)[None, ...].cuda()
                test_feature = model(T_C_tensor)
                reference_feature = model(R_C_tensor)
                cos_similarity = F.cosine_similarity(
                    test_feature.reshape(1, -1),
                    reference_feature.reshape(1, -1)
                )
                cos_similarity = torch.clamp(cos_similarity, -1 + 1e-6, 1 - 1e-6)
                Spearman_score = torch.arccos(cos_similarity) / math.pi  # shape: [1]
                Spearman_matrix_score[contrast_mask_index, multiplier_index] = Spearman_score
        multiplier_tensor = torch.tensor(self.multiplier_list, dtype=torch.float32, device='cuda')
        X = multiplier_tensor.repeat(len(self.gt_x_mask_C))  # shape: [N * M]
        Y = Spearman_matrix_score.flatten()
        X_mean = torch.mean(X)
        Y_mean = torch.mean(Y)
        X_centered = X - X_mean
        Y_centered = Y - Y_mean
        numerator = torch.sum(X_centered * Y_centered)
        denominator = torch.sqrt(torch.sum(X_centered ** 2) * torch.sum(Y_centered ** 2) + 1e-8)
        correlation_pearson = numerator / denominator
        loss = 1 - correlation_pearson
        # make_dot(loss, params=dict(model.named_parameters())).render("computation_graph", format="pdf")
        return loss

    def test_models_plot_contours(self, model_name, model, suffix, epoch, resolution=None):
        with torch.no_grad():
            contrast_mask_matrix = np.zeros([len(self.contrast_mask_list), len(self.contrast_test_list)])
            contrast_test_matrix = np.zeros([len(self.contrast_mask_list), len(self.contrast_test_list)])
            arccos_cos_similarity_matrix = np.zeros([len(self.contrast_mask_list), len(self.contrast_test_list)])

            for contrast_mask_index in range(len(self.contrast_mask_list)):
                contrast_mask_value = self.contrast_mask_list[contrast_mask_index]
                for contrast_test_index in range(len(self.contrast_test_list)):
                    contrast_test_value = self.contrast_test_list[contrast_test_index]
                    T_L_array, R_L_array = generate_contrast_masking_gabor_on_noise(W=self.W, H=self.H,
                                                                                    sigma=self.R,
                                                                                    rho=self.rho_test,
                                                                                    Mask_upper_frequency=self.Mask_upper_frequency,
                                                                                    L_b=self.L_b,
                                                                                    contrast_mask=contrast_mask_value,
                                                                                    contrast_test=contrast_test_value,
                                                                                    ppd=self.ppd)
                    T_L_array = np.stack([T_L_array] * 3, axis=-1)
                    R_L_array = np.stack([R_L_array] * 3, axis=-1)
                    contrast_mask_matrix[contrast_mask_index, contrast_test_index] = contrast_mask_value
                    contrast_test_matrix[contrast_mask_index, contrast_test_index] = contrast_test_value
                    if resolution is not None:
                        T_L_array = cv2.resize(T_L_array, (resolution[0], resolution[1]),
                                               interpolation=cv2.INTER_LINEAR)
                        R_L_array = cv2.resize(R_L_array, (resolution[0], resolution[1]),
                                               interpolation=cv2.INTER_LINEAR)
                    T_C_array = display_encode_tool.L2C_sRGB(T_L_array)
                    T_C_tensor = torch.tensor(T_C_array, dtype=torch.float32).permute(2, 0, 1)[None, ...].cuda()
                    R_C_array = display_encode_tool.L2C_sRGB(R_L_array)
                    R_C_tensor = torch.tensor(R_C_array, dtype=torch.float32).permute(2, 0, 1)[None, ...].cuda()
                    test_feature = model(T_C_tensor)
                    reference_feature = model(R_C_tensor)
                    cos_similarity = F.cosine_similarity(
                        test_feature.reshape(1, -1),
                        reference_feature.reshape(1, -1)
                    )
                    cos_similarity = torch.clamp(cos_similarity, -1 + 1e-6, 1 - 1e-6)
                    arccos_cos_similarity_matrix[contrast_mask_index, contrast_test_index] = np.arccos(np.array(cos_similarity.cpu())) / np.arccos(-1)

        plot_ticks_label_list = plot_ticks_label_dict[self.test_short_name]
        X_ticks = plot_ticks_label_list[0]
        Y_ticks = plot_ticks_label_list[1]
        X_label = plot_ticks_label_list[2]
        Y_label = plot_ticks_label_list[3]
        plot_X_matrix = contrast_mask_matrix
        plot_Y_matrix = contrast_test_matrix
        plot_score_matrix = arccos_cos_similarity_matrix
        gt_result_X_list = self.gt_x_mask_C
        gt_result_Y_list = self.gt_y_test_C
        plot_figure_name = f'{model_name}_arccos_epoch{epoch}'
        if resolution is not None:
            plot_figure_name += f'_{resolution[0]}_{resolution[1]}'
        plt.figure(figsize=(4.3,3), dpi=300)
        levels = np.linspace(0, 1, 200)
        plt.contourf(plot_X_matrix, plot_Y_matrix, plot_score_matrix,
                     levels=levels, cmap='rainbow', alpha=0.3)
        plt.contour(plot_X_matrix, plot_Y_matrix, plot_score_matrix,
                    levels=levels, cmap='rainbow', linewidths=1)
        plt.plot(gt_result_X_list, gt_result_Y_list, 'k', linestyle='--', linewidth=2,
                 label='Human Results', marker='o')
        plt.xlim([plot_X_matrix.min(), plot_X_matrix.max()])
        plt.ylim([plot_Y_matrix.min(), plot_Y_matrix.max()])
        plt.xlabel(X_label, fontsize=12)
        plt.ylabel(Y_label, fontsize=12)
        plt.xscale('log')
        plt.yscale('log')
        plt.xticks(X_ticks, X_ticks)
        plt.yticks(Y_ticks, Y_ticks)
        plt.tight_layout()
        plt.legend(loc='lower right')
        save_figure_dir = os.path.join(f'plot_contours/test_{suffix}/test_{self.test_short_name}', model_name)
        os.makedirs(save_figure_dir, exist_ok=True)
        plt.savefig(os.path.join(save_figure_dir, plot_figure_name + '.png'), dpi=300,
                    bbox_inches='tight', pad_inches=0.02)
        plt.close()