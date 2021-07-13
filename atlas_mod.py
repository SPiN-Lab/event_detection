import nibabel as nib
import numpy as np
import os
import subprocess


def transform(atlas_orig, data_tlrc, temp_dir):
    if np.any(nib.load(atlas_orig).affine != nib.load(data_tlrc).affine):
        atlas_mat = nib.load(atlas_orig).get_fdata()
        tmp_image = nib.Nifti1Image(
            atlas_mat, nib.load(data_tlrc).affine, nib.load(data_tlrc).header
        )
        TMP_name = "atlas.nii.gz"

        nib.save(tmp_image, os.path.join(temp_dir, TMP_name))
        subprocess.run(
            f"3drefit -space TLRC -view tlrc {os.path.join(temp_dir, TMP_name)}",
            shell=True,
        )
    else:
        os.system(f"cp {atlas_orig} {os.path.join(temp_dir, TMP_name)}")
    return os.path.join(temp_dir, TMP_name)


def inverse_transform(data_tlrc, atlas_orig):
    subprocess.run(
            f"3drefit -space ORIG -view orig {data_tlrc}",
            shell=True,
        )
    tmp_data = nib.Nifti1Image(nib.load(data_tlrc).get_fdata(),
                               nib.load(atlas_orig).affine,
                               nib.load(atlas_orig).header
                               )
    nib.save(tmp_data, data_tlrc)
