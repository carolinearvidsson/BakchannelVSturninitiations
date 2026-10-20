"""
Adapted from a script written by Ekaterina Torubarova, which was used in 
the Torubarova et al. (2026) study on conversational production and comprehension.
"""

import os
import numpy as np
import nibabel as nib
import csv
from nilearn import datasets, surface
from scipy.io import loadmat

# CONFIGURATION

BASE_DIR = '/Users/caroline/Desktop/Dissertation/TI_Backchannels/'

FIRSTLEVEL_DIR = os.path.join(
    BASE_DIR,
    'ROIanalysis',
    'first_level'
)

MASKS_DIR = os.path.join(
    BASE_DIR,
    'ROIanalysis',
    'ROIs'
)

OUTPUT_CSV = os.path.join(
    BASE_DIR,
    'ROIanalysis',
    'mshbm_TI_BI_runlevel_surface_betas.csv'
)

TARGET_MESH = 'fsaverage5'

NETWORKS = [
    'DU_DN_A',
    'DU_DN_B',
    'DU_FPN_B'
]

INCLUDED_SUBS = range(1, 34)

# LOAD INDIVIDUAL MSHBM MASK

def load_mask(sub_str, network):

    network_dir = os.path.join(
        MASKS_DIR,
        network
    )

    lh_file = os.path.join(
        network_dir,
        f'{sub_str}_hemi-L_space-{TARGET_MESH}_{network}.gii'
    )

    rh_file = os.path.join(
        network_dir,
        f'{sub_str}_hemi-R_space-{TARGET_MESH}_{network}.gii'
    )

    if not os.path.exists(lh_file):
        print(f'    WARNING: Missing LH mask: {lh_file}')
        return None, None

    if not os.path.exists(rh_file):
        print(f'    WARNING: Missing RH mask: {rh_file}')
        return None, None

    lh_mask = nib.load(lh_file).darrays[0].data > 0
    rh_mask = nib.load(rh_file).darrays[0].data > 0

    return lh_mask, rh_mask

# GET REGRESSOR NAMES FROM SPM.MAT

def get_spm_regressors(spm_path):

    mat = loadmat(
        spm_path,
        struct_as_record=False,
        squeeze_me=True
    )

    SPM = mat['SPM']

    names = SPM.xX.name

    # Ensure iterable
    names = np.atleast_1d(names)

    regressors = []

    for name in names:
        regressors.append(str(name))

    return regressors

# IDENTIFY TI / BI REGRESSORS

def identify_condition_regressors(regressors):

    found = []

    for i, name in enumerate(regressors):

        lower = name.lower()

        # Only task regressors convolved with bf(1)

        if '*bf(1)' not in lower:
            continue

        # Identify condition

        condition = None
        if ')' in lower:
            regressor_part = lower.split(')', 1)[1].strip()
        else:
            regressor_part = lower.strip()

        if regressor_part.startswith('ti*bf(1)'):
            condition = 'TI'

        elif regressor_part.startswith('bi*bf(1)'):
            condition = 'BI'

        else:
            continue

        run = None

        if lower.startswith('sn('):

            try:
                run_text = lower.split('sn(')[1].split(')')[0]
                run = int(run_text)

            except (IndexError, ValueError):
                pass

        if run is None:
            print(
                f'  WARNING: Could not determine run for: {name}'
            )
            continue

        beta_index = i + 1

        beta_file = f'beta_{beta_index:04d}.nii'

        found.append({
            'Run': run,
            'Condition': condition,
            'BetaIndex': beta_index,
            'BetaFile': beta_file,
            'Regressor': name
        })

    return found

# EXTRACT NETWORK MEAN

def extract_network_mean(
    lh_surface,
    rh_surface,
    lh_mask,
    rh_mask
):

    # Check dimensions
    if len(lh_surface) != len(lh_mask):
        raise ValueError(
            f'LH mismatch: surface={len(lh_surface)}, '
            f'mask={len(lh_mask)}'
        )

    if len(rh_surface) != len(rh_mask):
        raise ValueError(
            f'RH mismatch: surface={len(rh_surface)}, '
            f'mask={len(rh_mask)}'
        )

    lh_values = lh_surface[lh_mask]
    rh_values = rh_surface[rh_mask]

    all_values = np.concatenate([
        lh_values,
        rh_values
    ])

    all_values = all_values[
        np.isfinite(all_values)
    ]

    if len(all_values) == 0:
        return np.nan

    return np.mean(all_values)

# MAIN

def main():

    print('MSHBM surface ROI extraction')
    print('RUN-LEVEL TI AND BI BETAS')
 

    print(f'\nFirst-level directory: {FIRSTLEVEL_DIR}')
    print(f'Mask directory:        {MASKS_DIR}')
    print(f'Output:                {OUTPUT_CSV}')

    # fsaverage5

    print('\nFetching fsaverage5 templates...')

    fsaverage = datasets.fetch_surf_fsaverage(
        TARGET_MESH
    )

    results = []

    # SUBJECT LOOP

    for sub_num in INCLUDED_SUBS:

        sub_str = f'sub-{sub_num:02d}'

        subject_dir = os.path.join(
            FIRSTLEVEL_DIR,
            sub_str
        )

        if not os.path.exists(subject_dir):

            print(
                f'\n[{sub_str}] WARNING: '
                f'Missing first-level directory. Skipping.'
            )

            continue

        print(f'Processing {sub_str}')

        spm_path = os.path.join(
            subject_dir,
            'SPM.mat'
        )

        if not os.path.exists(spm_path):

            print('  WARNING: SPM.mat missing. Skipping.')
            continue

        regressors = get_spm_regressors(
            spm_path
        )

        condition_regressors = identify_condition_regressors(
            regressors
        )

        print('\n  Identified regressors:')

        for reg in condition_regressors:

            print(
                f"    Run {reg['Run']} | "
                f"{reg['Condition']} | "
                f"{reg['BetaFile']} | "
                f"{reg['Regressor']}"
            )

        if len(condition_regressors) == 0:

            print(
                '  WARNING: No TI/BI regressors found. '
                'Skipping subject.'
            )

            continue

        # Check condition/run combinations

        print('\n  Condition counts:')

        for run in sorted(
            set(x['Run'] for x in condition_regressors)
        ):

            for condition in ['BI', 'TI']:

                n = sum(
                    x['Run'] == run and
                    x['Condition'] == condition
                    for x in condition_regressors
                )

                print(
                    f'    Run {run}, {condition}: {n}'
                )

        # LOAD NETWORK MASKS ONCE PER SUBJECT

        masks = {}

        for network in NETWORKS:

            lh_mask, rh_mask = load_mask(
                sub_str,
                network
            )

            if lh_mask is None:
                continue

            masks[network] = {
                'lh': lh_mask,
                'rh': rh_mask
            }
        # LOOP OVER RUN-LEVEL TI / BI BETAS

        for reg in condition_regressors:

            run = reg['Run']
            condition = reg['Condition']
            beta_file = reg['BetaFile']

            beta_path = os.path.join(
                subject_dir,
                beta_file
            )

            if not os.path.exists(beta_path):

                print(
                    f'  WARNING: Missing {beta_file}. '
                    f'Skipping this regressor.'
                )

                continue

            print(
                f'\n  Projecting Run {run} '
                f'{condition}: {beta_file}'
            )

            # Project beta to fsaverage5

            lh_surface = surface.vol_to_surf(
                beta_path,
                fsaverage.pial_left,
                inner_mesh=fsaverage.white_left
            )

            rh_surface = surface.vol_to_surf(
                beta_path,
                fsaverage.pial_right,
                inner_mesh=fsaverage.white_right
            )

            # EXTRACT EACH NETWORK

            for network in NETWORKS:

                if network not in masks:
                    continue

                lh_mask = masks[network]['lh']
                rh_mask = masks[network]['rh']

                n_lh = int(np.sum(lh_mask))
                n_rh = int(np.sum(rh_mask))
                n_total = n_lh + n_rh

                try:

                    mean_beta = extract_network_mean(
                        lh_surface,
                        rh_surface,
                        lh_mask,
                        rh_mask
                    )

                except ValueError as error:

                    print(
                        f'    ERROR {network}: {error}'
                    )

                    continue

                print(
                    f'    {network}: '
                    f'beta = {mean_beta:.6f}'
                )

                results.append({
                    'Subject': sub_str,
                    'Run': run,
                    'ROI': network,
                    'Condition': condition,
                    'Beta': mean_beta,
                    'LH_vertices': n_lh,
                    'RH_vertices': n_rh,
                    'Total_vertices': n_total,
                    'BetaFile': beta_file,
                    'Regressor': reg['Regressor']
                })

    # SAVE CSV

    if results:

        # Sort output
        results = sorted(
            results,
            key=lambda x: (
                x['Subject'],
                x['Run'],
                x['ROI'],
                x['Condition']
            )
        )

        fieldnames = [
            'Subject',
            'Run',
            'ROI',
            'Condition',
            'Beta',
            'LH_vertices',
            'RH_vertices',
            'Total_vertices',
            'BetaFile',
            'Regressor'
        ]

        with open(
            OUTPUT_CSV,
            'w',
            newline=''
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames
            )

            writer.writeheader()
            writer.writerows(results)

        print('DONE')

        print(
            f'Extracted {len(results)} '
            f'run-level observations.'
        )

        print(
            f'\nSaved to:\n{OUTPUT_CSV}'
        )

    else:

        print('\nNo valid data were extracted.')

# RUN

if __name__ == '__main__':
    main()