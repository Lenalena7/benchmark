import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import sys
import argparse

def view_nii(file_path, save_path=None):
    try:
        img = nib.load(file_path)
        data = img.get_fdata()
        print(f"影像載入成功！維度: {data.shape}")
        
        # 取三個軸向的中間層
        mid_x = data.shape[0] // 2
        mid_y = data.shape[1] // 2
        mid_z = data.shape[2] // 2

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Axial, Sagittal, Coronal 視角
        axes[0].imshow(np.rot90(data[:, :, mid_z]), cmap='viridis')
        axes[0].set_title(f"Axial (Slice {mid_z})")
        
        axes[1].imshow(np.rot90(data[mid_x, :, :]), cmap='viridis')
        axes[1].set_title(f"Sagittal (Slice {mid_x})")
        
        axes[2].imshow(np.rot90(data[:, mid_y, :]), cmap='viridis')
        axes[2].set_title(f"Coronal (Slice {mid_y})")

        for ax in axes:
            ax.axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            print(f"預覽圖已儲存至: {save_path}")
        else:
            plt.show()
            
    except Exception as e:
        print(f"發生錯誤: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="NIfTI file path")
    parser.add_argument("--save", help="Save the preview image path")
    parser.add_argument("--all_axes", action="store_true", help="Show 3 axes")
    args = parser.parse_args()

    view_nii(args.file, args.save)