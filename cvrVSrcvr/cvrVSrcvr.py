# -*- coding: utf-8 -*-
"""
cvrVSrcvr main code

Tool to compare CVR and rCVR maps.
"""

# info

# You must download the data from openneuro to run this script. You need ds004604 and ds005418. The expected location is /data/ds004604 and /data/ds005418.
# You also need to install the following python packages: numpry, os, itertools, nilearn, pathlib, pybids, pandas, xml, json and warnings
# Make also sure you have the following files installed (see fsl installation to get then):
# '/opt/fsl/data/atlases/HarvardOxford-Cortical.xml'
# '/opt/fsl/data/atlases/HarvardOxford/HarvardOxford-cort-prob-1mm.nii.gz'
# To run the script, execute `python cvrVSrcvr.py'

# preamble

import numpy as np
from os.path import join
from itertools import combinations
from nilearn.plotting import plot_glass_brain
from nilearn.reporting import get_clusters_table
from pathlib import Path
import warnings

# functions

def load_layout_and_derivatives(bids_dir, fmriprep_dir, derivative):
    from bids import BIDSLayout as bidslayout
    layout = bidslayout(bids_dir, validate=False)
    layout.add_derivatives(fmriprep_dir, validate=False)
    layout.add_derivatives(derivative, validate=False)
    return layout

def check_number_of_subjects(layout1, layout2, first_task='gas', second_task='gas', key='cvrmap'):
    subjects1 = layout1.derivatives[key].get_subjects(task=first_task)
    subjects2 = layout2.derivatives[key].get_subjects(task=second_task)
    n1 = len(subjects1)
    n2 = len(subjects2)

    test = n1 == n2
    
    if test:
        print('Number of subjects for both derivatives (%s): %s' % (key, n1))
    else:
        print('Error: mismatch number of subjects! (%s and %s)' % (n1, n2))
        for s in subjects1:
            if not s in subjects2:
                print("%s is not in both sets" % s)
        for s in subjects2:
            if not s in subjects1:
                print("%s is not in both sets" % s)
        

    return test

def get_data_of_interest(layout, space='MNI152NLin2009cAsym', key='cvrmap', task='gas'):
    from nilearn.image import load_img
    filenames = []
    for item in layout.derivatives[key].get(space=space, return_type='filename', extension='.nii.gz', suffix=['cvr', 'rcvr'], task=task):
        filenames.append(load_img(item))
    return filenames

def harmonize_affines(filenames, ref,  interpolation='continuous'):
    from nilearn.image import resample_to_img
    imgs = []
    for fn in filenames:
        imgs.append(resample_to_img(fn, ref, interpolation = interpolation))
    return imgs

def get_masks(layout, space = 'MNI152NLin2009cAsym', task = 'gas'):
    from nilearn.image import load_img
    brainmasks = []
    for item in layout.derivatives['fMRIPrep'].get(space=space, return_type='filename', extension='.nii.gz',
                                               desc='brain', task=task, suffix='mask'):
        brainmasks.append(load_img(item))
    return brainmasks

def get_std(img):
    return np.nanstd(img.get_fdata())

def get_wholebrain_mean(img):
    return np.nanmean(img.get_fdata())

def get_vessel_mean(img):
    from nilearn.image import binarize_img, resample_to_img
    from nilearn.masking import apply_mask
    import numpy as np

    vs_atlas_path = '/home/arovai/git/ln2t/cvrmap/cvrmap/data/VesselDensityLR.nii.gz'
    # vs_atlas_path = '/home/arovai/git/cvrmap/cvrmap/data/VesselDensityLR.nii.gz'
    vs_atlas_resampled = resample_to_img(source_img=vs_atlas_path, target_img=img)
    vs_mask = binarize_img(img=vs_atlas_resampled, threshold='99.5%')
    masked_data = apply_mask(imgs=img, mask_img=vs_mask)
    return np.nanmean(masked_data.flatten())

def rescale_imgs(imgs, scaling):
    from nilearn.image import math_img
    imgs_rescaled = []   

    for img in imgs:
        #_normalized = image.math_img('(img - np.nanmean(img))/np.nanstd(img)', img=img)

        if scaling == 'std':
            rescaling_factor = get_std(img)
        if scaling == 'wholebrain':
            rescaling_factor = get_wholebrain_mean(img)
        if scaling == 'vesselmask':
            rescaling_factor = get_vessel_mean(img)

        if not rescaling_factor == 0:
            rescaled = math_img('img/%s' % rescaling_factor, img=img)
        else:
            print('Rescaling factor is zero, something is wrong?')
        
        imgs_rescaled.append(rescaled)

    return imgs_rescaled

def create_pairwise_design_matrix(n_subjects, label):

    import pandas as pd
    import numpy as np
    
    condition_effect = np.hstack(([1] * n_subjects, [-1] * n_subjects))
    
    subject_effect = np.vstack((np.eye(n_subjects), np.eye(n_subjects)))
    subjects = [f"sub-{i:03d}" for i in range(1, n_subjects + 1)]
    
    paired_design_matrix = pd.DataFrame(
        np.hstack((condition_effect[:, np.newaxis], subject_effect)),
        columns=[label] + subjects,
    )
    
    return paired_design_matrix

def pairwise_group_compare_derivatives(bids_dir, fmriprep_dir, 
                                       first_derivative, second_derivative,
                                       first_task = 'gas',
                                       second_task = 'gas',
                                       scaling='std', effect_label = "Derivatives1VersusDerivatives2", 
                                       plot_title = "Comparison of derivatives 1 versus derivatives 2"):


    first_layout = load_layout_and_derivatives(bids_dir, fmriprep_dir, first_derivative)
    second_layout = load_layout_and_derivatives(bids_dir, fmriprep_dir, second_derivative)

    check_number_of_subjects(first_layout, second_layout, first_task=first_task, second_task=second_task)

    first_filenames = get_data_of_interest(first_layout, task=first_task)
    second_filenames = get_data_of_interest(second_layout, task=second_task)

    first_imgs = harmonize_affines(first_filenames, first_filenames[0])
    second_imgs = harmonize_affines(second_filenames, first_filenames[0])

    first_imgs_rescaled = rescale_imgs(first_imgs, scaling=scaling)
    second_imgs_rescaled = rescale_imgs(second_imgs, scaling=scaling)
    
    n_subjects = len(first_filenames)
    paired_design_matrix = create_pairwise_design_matrix(n_subjects, label=effect_label)
    
    second_level_input = first_imgs_rescaled + second_imgs_rescaled
    
    from nilearn.glm.second_level import SecondLevelModel
    second_level_model_paired = SecondLevelModel().fit(second_level_input, design_matrix=paired_design_matrix)
    stat_maps = second_level_model_paired.compute_contrast(effect_label, output_type="all")

    from nilearn.glm import threshold_stats_img
    thresholded_stat_map, threshold = threshold_stats_img(stat_maps['z_score'], 
                                                        cluster_threshold=0, 
                                                        height_control='fdr')

    from nilearn.plotting import plot_glass_brain
    plot_glass_brain(stat_maps['z_score'], 
                     threshold = threshold, 
                     title = plot_title, 
                     plot_abs=False, 
                     colorbar=True)
    
    return second_level_model_paired, stat_maps, threshold

def mni_to_voxel(x, y, z, affine):
    import numpy as np
    mni = np.array([[x], [y], [z], [1]])
    voxel = np.linalg.inv(affine).dot(mni)
    return tuple(np.round(voxel[:3]).astype(int))

def read_xml(xml_file):
    import xml.etree.ElementTree as ET
    labels = {}
    tree = ET.parse(xml_file)
    root = tree.getroot()
    for label in root.findall('.//label'):
        index = int(label.get('index'))
        name = label.text.strip()
        labels[index] = name
    return labels

def get_location_harvard_oxford_old(row, data, affine, xml_file, prob_threshold=0):

    x = row['X']
    y = row['Y']
    z = row['Z']

    # Convert MNI coordinates to voxel coordinates
    voxel_x, voxel_y, voxel_z = mni_to_voxel(x, y, z, affine)

    # Get volumes at voxel coordinates
    volumes = data[voxel_x, voxel_y, voxel_z, :].flatten()

    # Read XML file to get label names
    labels = read_xml(xml_file)

    # Sort volumes by probability
    sorted_volumes = sorted(enumerate(volumes), key=lambda x: x[1], reverse=True)

    # Print selected volumes with corresponding label names
    output = []

    for index, prob in sorted_volumes:
        label_name = labels.get(index, "Unknown")
        if prob > prob_threshold:
            _str = f"{label_name} ({prob} %)"
            output.append(_str)

    return ' and '.join(output)

def get_location_harvard_oxford_on_df_old(df):

    import nibabel as nib
    xml_file = '/opt/fsl/data/atlases/HarvardOxford-Cortical.xml'
    nifti_file = '/opt/fsl/data/atlases/HarvardOxford/HarvardOxford-cort-prob-1mm.nii.gz'
    
    # Load NIfTI file
    nifti = nib.load(nifti_file)
    data = nifti.get_fdata()
    # Get affine transformation matrix
    affine = nifti.affine

    df['Location (Harvard-Oxford)'] = df.apply(lambda row: get_location_harvard_oxford_old(row, data, affine, xml_file), axis=1)
    return df

def compute_mask_size(img):
    voxels = np.count_nonzero(img.get_fdata())
    voxel_size = np.prod(img.header.get_zooms())
    return voxels * voxel_size

def get_clusters_location_harvard_oxford(label_maps, stat_img, prob_threshold=0.0):
    """
        prob_threshold: str or float. If string, must be of the form '5%' or '17%'. If float, must be a p-value, e.g. '0.05' of '0.17'.
    """
    import pandas as pd
    from nilearn.image import load_img, resample_to_img, binarize_img, math_img
    from nilearn.masking import apply_mask
    from os.path import isfile
    cluster_table = pd.DataFrame()
    if type(label_maps) is list:
        positive_label_map = label_maps[0]
        negative_label_map = label_maps[1]
        positive_cluster_table = get_clusters_location_harvard_oxford(positive_label_map,
                                                                      stat_img,
                                                                      prob_threshold=prob_threshold)
        negative_cluster_table = get_clusters_location_harvard_oxford(negative_label_map,
                                                                      stat_img,
                                                                      prob_threshold=prob_threshold)

        n_pos_clusters = len(positive_cluster_table)
        
        negative_cluster_table["Cluster id"] = negative_cluster_table.apply(lambda row: row["Cluster id"] + n_pos_clusters, axis=1)
        
        cluster_table = pd.concat([positive_cluster_table, negative_cluster_table], axis=0)
    else:
        levels = set(label_maps.get_fdata().flatten())
        if 0 in levels:
            levels.remove(0)
            
        atlas = '/opt/fsl/data/atlases/HarvardOxford/HarvardOxford-cort-prob-1mm.nii.gz'
        xml_file = '/opt/fsl/data/atlases/HarvardOxford-Cortical.xml'
        
        if not isfile(atlas):
            atlas = '/data/atlases/HarvardOxford/HarvardOxford-cort-prob-1mm.nii.gz'
        if not isfile(xml_file):
            xml_file = '/data/atlases/HarvardOxford-Cortical.xml'
        
        atlas = load_img(atlas)
        atlas_labels = read_xml(xml_file)

        cluster_table = pd.DataFrame(columns=['Cluster id', 'Cluster z-score mean', 'Cluster size (mm3)', 'Location (Harvard-Oxford)'])
        
        if type(prob_threshold) is str:
            # Convert string to float percentage
            prob_threshold = float(prob_threshold.split(sep='%')[0])
        else:
            # Convert p-value to percentage
            prob_threshold *= 100
        
        for lvl in levels:
            single_cluster_map = math_img('img == %s' % lvl, img=label_maps)
            size = np.round(compute_mask_size(single_cluster_map))
            single_cluster_map = resample_to_img(single_cluster_map, atlas, interpolation='continuous')
            single_cluster_map = binarize_img(single_cluster_map, threshold="50%")
            data = apply_mask(atlas, single_cluster_map)
            
            probabilities = []
            for i in np.arange(atlas.shape[-1]):
                prop = np.round(np.mean(data[i, :]), decimals=2)
                probabilities.append(prop)
            
            sorted_probabilities = sorted(enumerate(probabilities), key=lambda x: x[1], reverse=True)

            output = []
            
            for index, prob in sorted_probabilities:
                label_name = atlas_labels.get(index, "Unknown")
                if prob > prob_threshold:
                    _str = f"{label_name} ({str(prob)} %)"
                    output.append(_str)

            if len(output) == 0:
                string = "Non-cortical location"
            else:
                string = ' and '.join(output)

            stat_img = harmonize_affines([stat_img], ref=single_cluster_map)[0]
            mean_score = np.mean(apply_mask(stat_img, mask_img=single_cluster_map))

            cluster_table.loc[len(cluster_table)] = [int(lvl), mean_score, int(size), string]
    return cluster_table

def make_summarized_cluster_table(table):
    summary = table
    # table.loc[table['Cluster Size (mm3)'] > 0]
    return summary

def save_threshold_to_json(threshold, filename):
    import json
    data = {}
    data['z_threshold'] = threshold
    with open(filename, 'w') as f:
        json.dump(data, f)

def perform_dataset_analysis(bids_dir,
                             inputs,
                             fmriprep_dir=None,
                             output_dir=None,
                             task_to_select=None,
                             scaling_strategies=['wholebrain']):
    
    # deal with default args
    
    if fmriprep_dir is None:
        fmriprep_dir = join(bids_dir, 'derivatives', 'fmriprep_v21.0.4')
    if output_dir is None:
        output_dir = join(bids_dir, 'derivatives', 'comparisons')
    if task_to_select is None:
        task_to_select = {}
        for key in inputs.keys():
            task_to_select[key] = 'gas'
    
    # perform the group analysis (paired t-tests)

    results = {}

    for maps1, maps2 in combinations(list(inputs.keys()), 2):
        results[(maps1, maps2)] = {}
        print('Running for %s versus %s' % (maps1, maps2))
        for scaling in scaling_strategies:
            print('      - Running scaling strategy %s' % scaling)
            model, maps, threshold = second_level_model_paired, stat_maps, threshold = pairwise_group_compare_derivatives(
                                                            bids_dir, fmriprep_dir, 
                                                            inputs[maps1],
                                                            inputs[maps2],
                                                            first_task = task_to_select[maps1],
                                                            second_task = task_to_select[maps2],
                                                            scaling=scaling,
                                                            effect_label = "Derivatives1VersusDerivatives2",
                                                            plot_title = '%s versus %s, scaling: %s' % 
                                                                            (maps1, maps2, scaling))
            results[(maps1, maps2)][scaling] = {}
            results[(maps1, maps2)][scaling]['model'] = model
            results[(maps1, maps2)][scaling]['maps'] = maps
            results[(maps1, maps2)][scaling]['threshold'] = threshold
            
    # find cluster tables

    clusters = {}
    label_maps = {}
    summarized_clusters = {}

    for maps1, maps2 in combinations(list(inputs.keys()), 2):
        clusters[(maps1, maps2)] = {}
        label_maps[(maps1, maps2)] = {}
        summarized_clusters[(maps1, maps2)] = {}
        for scaling in scaling_strategies:
            stat_img = results[(maps1, maps2)][scaling]['maps']['z_score']
            threshold = results[(maps1, maps2)][scaling]['threshold']
            table, label_map = get_clusters_table(stat_img = stat_img,
                              stat_threshold = threshold,
                              two_sided = True,
                              cluster_threshold = 20,
                              return_label_maps=True)
            if not table.empty:
                #table = get_location_harvard_oxford_on_df(table)
                table_with_labeled_clusters = get_clusters_location_harvard_oxford(label_map,
                                                                                   stat_img,
                                                                                   prob_threshold=0.05)
            
            clusters[(maps1, maps2)][scaling] = table_with_labeled_clusters
            label_maps[(maps1, maps2)][scaling] = label_map
            summarized_clusters[(maps1, maps2)][scaling] = make_summarized_cluster_table(table_with_labeled_clusters)
            
    # save the results (z_score maps, figures, and cluster tables)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for maps1, maps2 in combinations(list(inputs.keys()), 2):
        for scaling in scaling_strategies:
            zmap_filename = join(output_dir,
                                   '%s_versus_%s_scaling-%s_z_score.nii.gz'  % (maps1, maps2, scaling))
            json_filename = join(output_dir,
                                   '%s_versus_%s_scaling-%s_z_score.json'  % (maps1, maps2, scaling))
            
            print('Saving z-score map to %s' % zmap_filename)
            results[(maps1, maps2)][scaling]['maps']['z_score'].to_filename(zmap_filename)
            
            print('Saving z-threshold to %s' % json_filename)
            save_threshold_to_json(results[(maps1, maps2)][scaling]['threshold'], json_filename)

            cluster_table_filename = join(output_dir,
                                   '%s_versus_%s_scaling-%s_z_score.csv'  % (maps1, maps2, scaling))
            print('Saving cluster table to %s' % cluster_table_filename)
            clusters[(maps1, maps2)][scaling].to_csv(cluster_table_filename, sep='\t', index=False)
            
            #summarized_cluster_table_filename = join(output_dir,
            #                       '%s_versus_%s_scaling-%s_z_score_summary.csv'  % (maps1, maps2, scaling))
            #print('Saving summarized cluster table to %s' % summarized_cluster_table_filename)
            #summarized_clusters[(maps1, maps2)][scaling].to_csv(summarized_cluster_table_filename, sep='\t', index=False)
            
            latex_cluster_table_filename = join(output_dir,
                                   '%s_versus_%s_scaling-%s_z_score.tex'  % (maps1, maps2, scaling))
            print('Saving latex cluster table to %s' % latex_cluster_table_filename)
            with open(latex_cluster_table_filename, 'w') as f:
                f.write(clusters[(maps1, maps2)][scaling].to_latex(escape=True,
                                                                   index=False,
                                                                   column_format='cccl'))
            
            
    for maps1, maps2 in combinations(list(inputs.keys()), 2):
        for scaling in ['wholebrain']:
            plot_filename = join(output_dir,
                                   '%s_versus_%s_scaling-%s_z_score.svg'  % (maps1, maps2, scaling))
            plot_title = '%s versus %s' % (maps1, maps2)
            plot_title = None
            model = results[(maps1, maps2)][scaling]['model']
            maps = results[(maps1, maps2)][scaling]['maps']
            threshold = results[(maps1, maps2)][scaling]['threshold']
            
            print('Saving figure to %s' % plot_filename)
            plot_glass_brain(maps['z_score'],
                             threshold=threshold,
                             plot_abs=False,
                             colorbar=True,
                             title=plot_title,
                             output_file=plot_filename)
            
    return results, clusters

# main script

warnings.filterwarnings('ignore')

# ds004604 - 50 subjets, all with CO2 inhalation breathing challenge together with physiological monitoring.

bids_dir_ds004604 = '/data/ds004604'

inputs_ds004604 = {}
inputs_ds004604['true-CVR']          = join(bids_dir_ds004604, 'derivatives', 'cvrmap_2.0.25')
inputs_ds004604['globalsignal-rCVR'] = join(bids_dir_ds004604, 'derivatives', 'cvrmap_2.0.25_gs')
inputs_ds004604['vesselsignal-rCVR'] = join(bids_dir_ds004604, 'derivatives', 'cvrmap_2.0.25_vs')

print('Starting analysis for dataset ds004604')
perform_dataset_analysis(bids_dir_ds004604, inputs_ds004604)

# ds005418 - 35 subjects, all with both CO2 inhalation breathing challenge together with physiological monitoring and a resting-state session.

bids_dir_ds005418 = '/data/ds005418'

inputs_ds005418 = {}
inputs_ds005418['true-CVR']             = join(bids_dir_ds005418, 'derivatives', 'cvrmap_2.0.25')
inputs_ds005418['globalsignal-rs-rCVR'] = join(bids_dir_ds005418, 'derivatives', 'cvrmap_2.0.25_rs_gs')
inputs_ds005418['vesselsignal-rs-rCVR'] = join(bids_dir_ds005418, 'derivatives', 'cvrmap_2.0.25_rs_vs')

task_to_select_ds005418 = {}
task_to_select_ds005418['true-CVR'] = 'gas'
task_to_select_ds005418['globalsignal-rs-rCVR'] = 'restingstate'
task_to_select_ds005418['vesselsignal-rs-rCVR'] = 'restingstate'

print('Starting analysis for dataset ds005418')
perform_dataset_analysis(bids_dir_ds005418, inputs_ds005418, task_to_select=task_to_select_ds005418)