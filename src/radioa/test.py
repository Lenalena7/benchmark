import numpy as np
import nibabel as nib

# 替換成你數據集中的一個 label 路徑
label_path = "/data/gas/radioactive_data/datasets/Dataset201_MS_Flair_instances/labelsTr/1_Flair.nii.gz"
lb = nib.load(label_path).get_fdata()

print(f"標籤矩陣形狀: {lb.shape}") # 應該是 (320, 280, 20)

# 檢查每個軸向上有標籤（非 0）的切片數量
slices_with_data_axis0 = np.any(lb, axis=(1, 2)).sum()
slices_with_data_axis1 = np.any(lb, axis=(0, 2)).sum()
slices_with_data_axis2 = np.any(lb, axis=(0, 1)).sum()

print(f"Axis 0 (320) 方向有資料的切片數: {slices_with_data_axis0}")
print(f"Axis 1 (280) 方向有資料的切片數: {slices_with_data_axis1}")
print(f"Axis 2 (20)  方向有資料的切片數: {slices_with_data_axis2}")