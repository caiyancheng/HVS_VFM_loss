#!/bin/bash

# 激活conda环境（假设环境名为myenv）
# 注意：激活conda环境要用source并指定conda.sh路径
source ~/anaconda3/etc/profile.d/conda.sh
conda activate hvs_2
cd /auto/homes/yc613/Py_codes/HVS_VFM_loss/

#python train_main.py
#python train_main.py --test_classes Contrast_Masking_Phase_Coherent
#python train_main.py --test_classes Contrast_Masking_Phase_Incoherent
#python train_main.py --test_classes Contrast_Detection_SpF_Gabor_Ach
#python train_main.py --test_classes Contrast_Detection_Area
#python train_main.py --test_classes Contrast_Detection_Luminance
#python train_main.py --test_classes Contrast_Detection_SpF_Gabor_Ach Contrast_Detection_Area Contrast_Detection_Luminance Contrast_Masking_Phase_Coherent Contrast_Masking_Phase_Incoherent

#python test_main_2.py
#python test_main_2.py --test_classes Contrast_Detection_SpF_Gabor_Ach
#python test_main_2.py --test_classes Contrast_Detection_Area
#python test_main_2.py --test_classes Contrast_Detection_Luminance
#python test_main_2.py --test_classes Contrast_Masking_Phase_Coherent
#python test_main_2.py --test_classes Contrast_Masking_Phase_Incoherent
#python test_main_2.py --test_classes Contrast_Detection_SpF_Gabor_Ach Contrast_Detection_Area Contrast_Detection_Luminance Contrast_Masking_Phase_Coherent Contrast_Masking_Phase_Incoherent

python test_main_cross_domain.py
python test_main_cross_domain.py --test_classes Contrast_Detection_SpF_Gabor_Ach
python test_main_cross_domain.py --test_classes Contrast_Detection_Area
python test_main_cross_domain.py --test_classes Contrast_Detection_Luminance
python test_main_cross_domain.py --test_classes Contrast_Masking_Phase_Coherent
python test_main_cross_domain.py --test_classes Contrast_Masking_Phase_Incoherent
python test_main_cross_domain.py --test_classes Contrast_Detection_SpF_Gabor_Ach Contrast_Detection_Area Contrast_Detection_Luminance Contrast_Masking_Phase_Coherent Contrast_Masking_Phase_Incoherent
