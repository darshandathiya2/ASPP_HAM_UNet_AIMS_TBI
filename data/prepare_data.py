
import os
import nibabel as nib
import numpy as np
import imageio
from tqdm import tqdm

def extract_lesion_slices(filtered_mri_files,
                          filtered_lesion_files,
                          output_img_dir,
                          output_mask_dir):

    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)

    total_slices = 0

    for i in tqdm(range(len(filtered_mri_files))):

        file_id = i

        mri_path = filtered_mri_files[file_id]
        mask_path = filtered_lesion_files[file_id]

        mri = nib.load(mri_path).get_fdata()
        mask = nib.load(mask_path).get_fdata()

        for slice_idx in range(mask.shape[2]):

            mask_slice = mask[:, :, slice_idx]

            if np.any(mask_slice):

                mri_slice = mri[:, :, slice_idx]

                # Normalize MRI slice
                mri_slice = (
                    (mri_slice - mri_slice.min()) /
                    (mri_slice.ptp() + 1e-8) * 255
                ).astype(np.uint8)

                # Binary mask
                mask_slice = (mask_slice > 0).astype(np.uint8) * 255

                imageio.imwrite(
                    f"{output_img_dir}/{file_id}_slice_{slice_idx}.png",
                    mri_slice
                )

                imageio.imwrite(
                    f"{output_mask_dir}/{file_id}_slice_{slice_idx}.png",
                    mask_slice
                )

                total_slices += 1

    print(f"Total lesion-containing slices extracted: {total_slices}")
