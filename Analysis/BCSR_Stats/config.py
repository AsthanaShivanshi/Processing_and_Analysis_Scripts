import os
BASE_DIR = os.environ.get("BASE_DIR", "/work/FAC/FGSE/IDYST/tbeucler/downscaling")
DATASETS_TRAINING_DIR= os.environ.get("DATASETS_TRAINING_DIR", f"{BASE_DIR}/sasthana/Downscaling/Downscaling_Models/Dataset_Setup_I_Chronological_12km")
TARGET_DIR= os.environ.get("TARGET_DIR", f"{BASE_DIR}/sasthana/Downscaling/Processing_and_Analysis_Scripts/data_1971_2023/HR_files_full")   
BIAS_CORRECTED_DIR_SSP370= os.environ.get("BIAS_CORRECTED_DIR_RCP85", f"{BASE_DIR}/sasthana/Downscaling/GCM_pipeline/ALP-FINEv1.0/BC")
CH2025_DIR= os.environ.get("CH2025_DIR", f"{BASE_DIR}/sasthana/Downscaling/GCM_pipeline/ALP-FINEv1.0/BC/EQM")
BC_COARSE_EQM_DIR= os.environ.get("BC_COARSE_EQM_DIR", f"{BASE_DIR}/sasthana/Downscaling/GCM_pipeline/ALP-FINEv1.0/BC/EQM_C")
BC_COARSE_CDFT_DIR= os.environ.get("BC_COARSE_CDFT_DIR", f"{BASE_DIR}/sasthana/Downscaling/GCM_pipeline/ALP-FINEv1.0/BC/CDFT-t")


BC_COARSE_UNCORRECTED_DIR= os.environ.get("BC_COARSE_UNCORRECTED_DIR", f"{BASE_DIR}/sasthana/Downscaling/GCM_pipeline/ALP-FINEv1.0/Swiss")
BC_COARSE_dOTC_DIR= os.environ.get("BC_COARSE_dOTC_DIR", f"{BASE_DIR}/sasthana/Downscaling/GCM_pipeline/ALP-FINEv1.0/BC/dOTC")
BCSR_EQM_Bilinear_DIR= os.environ.get("BCSR_EQM_Bilinear_DIR", f"{BASE_DIR}/sasthana/Downscaling/GCM_pipeline/ALP-FINEv1.0/BC+SR/Bilinear/EQM_C")
BCSR_CDFT_Bilinear_DIR= os.environ.get("BCSR_CDFT_Bilinear_DIR", f"{BASE_DIR}/sasthana/Downscaling/GCM_pipeline/ALP-FINEv1.0/BC+SR/Bilinear/CDFT-t")
BCSR_dOTC_Bilinear_DIR= os.environ.get("BCSR_dOTC_Bilinear_DIR", f"{BASE_DIR}/sasthana/Downscaling/GCM_pipeline/ALP-FINEv1.0/BC+SR/Bilinear/dOTC")


BCSR_EQM_Bilinear_UNet_DIR= os.environ.get("BCSR_EQM_Bilinear_UNet_DIR", f"{BASE_DIR}/sasthana/Downscaling/GCM_pipeline/ALP-FINEv1.0/BC+SR/Bilinear_UNet/EQM_C")
BCSR_CDFT_Bilinear_UNet_DIR= os.environ.get("BCSR_CDFT_Bilinear_UNet_DIR", f"{BASE_DIR}/sasthana/Downscaling/GCM_pipeline/ALP-FINEv1.0/BC+SR/Bilinear_UNet/CDFT-t")
BCSR_dOTC_Bilinear_UNet_DIR= os.environ.get("BCSR_dOTC_Bilinear_UNet_DIR", f"{BASE_DIR}/sasthana/Downscaling/GCM_pipeline/ALP-FINEv1.0/BC+SR/Bilinear_UNet/dOTC")

BCSR_EQM_Bilinear_UNet_DDIM_DIR= os.environ.get("BCSR_EQM_Bilinear_UNet_DDIM_DIR", f"{BASE_DIR}/sasthana/Downscaling/GCM_pipeline/ALP-FINEv1.0/BC+SR/Bilinear_UNet_DDIM/EQM_C")
BCSR_CDFT_Bilinear_UNet_DDIM_DIR= os.environ.get("BCSR_CDFT_Bilinear_UNet_DDIM_DIR", f"{BASE_DIR}/sasthana/Downscaling/GCM_pipeline/ALP-FINEv1.0/BC+SR/Bilinear_UNet_DDIM/CDF-t")
BCSR_dOTC_Bilinear_UNet_DDIM_DIR= os.environ.get("BCSR_dOTC_Bilinear_UNet_DDIM_DIR", f"{BASE_DIR}/sasthana/Downscaling/GCM_pipeline/ALP-FINEv1.0/BC+SR/Bilinear_UNet_DDIM/dOTC")