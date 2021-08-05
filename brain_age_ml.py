#######################################################
# Python program name	: 
#Description	: brain_biological_chronological_age
#Args           :                                                                                  
#Author       	: Jaime Gomez-Ramirez                                               
#Email         	: jd.gomezramirez@gmail.com 
#REMEMBER to Activate Keras source ~/github/code/tensorflow/bin/activate
#pyenv install 3.7.0
#pyenv local 3.7.0
#python3 -V
# To use ipython3 debes unset esta var pq contien old version
#PYTHONPATH=/usr/local/lib/python2.7/site-packages
#unset PYTHONPATH
# $ipython3
# To use ipython2 /usr/local/bin/ipython2
#/Library/Frameworks/Python.framework/Versions/3.7/bin/ipython3
#pip install rfpimp. (only for python 3)
#######################################################
# -*- coding: utf-8 -*-
import os, sys, pdb, operator
import time
import numpy as np
import pandas as pd
import importlib
#importlib.reload(module)
import sys
from collections import Counter
import statsmodels.api as sm
import time
import importlib
import random
import pickle
from scipy import stats
from sklearn.model_selection import train_test_split
import shap
import re
#import rfpimp
#from rfpimp import *
#sys.path.append('/Users/jaime/github/code/tensorflow/production')
#import descriptive_stats as pv
sys.path.append('/Users/borri/github/bilateralBrain/code')
import thebilateral_brain_v2
import warnings
#from subprocess import check_output
#import area_under_curve 
import matplotlib
matplotlib.use('Agg')
#from matplotlib import pyplot
import matplotlib.pyplot as plt
import seaborn as sns
#import nibabel as nib

np.random.seed(11)
# Set figures dir. Use csv used in bilateral brain
if sys.platform == 'linux':
	from numpy.random import default_rng
	rng = default_rng(42)  # Create a random number generator
	figures_dir = "/mnt/c/Users/borri/github/bilateralBrain/figures"
	reports_dir  = "/mnt/c/Users/borri/github/bilateralBrain/reports"
	csv_path = "/mnt/c/Users/borri/github/BBDD/Vallecas_Index-Vols1234567-Siena-Free-27ONovember2019_segmentation.csv"
elif sys.platform == 'darwin':
	figures_dir = "/Users/borri/github/brain_age_estimation/figures"
	reports_dir  = "/Users/borri/github/brain_age_estimation/reports"
	csv_path = "/Users/borri/github/BBDD/Vallecas_Index-Vols1234567-Siena-Free-27ONovember2019_segmentation.csv"
else: 
	figures_dir = "C:\\Users\\borri\\github\\bilateralBrain\\figures"
	reports_dir = "C:\\Users\\borri\\github\\bilateralBrain\\reports"
	csv_path = "C:\\Users\\borri\\github\\BBDD\\Vallecas_Index-Vols1234567-Siena-Free-27ONovember2019_segmentation.csv"


def plot_boxandcorr_cortical(df, label=None):
	"""Plot boxplot and correlations for pandas df
	""" 
	## TODO

	prefix='free_'
	df.columns = df.columns.str.lstrip(prefix)
	df.rename(columns={"R_Thal": "rTh", "L_Thal": "lTh","R_Puta": "rPu", "L_Puta": "lPu", "R_Amyg": "rAm", "L_Amyg": "lAm","R_Pall": "rPa", "L_Pall": "lPa","R_Caud": "rCa", "L_Caud": "lCa","R_Hipp": "rHc", "L_Hipp": "lHc", "R_Accu": "rNAc", "L_Accu": "lNAc" }, inplace=True)
	subcortical = df.columns.to_list()

	print('Median volumes of %s \n'% label)
	sorted_nb = df[subcortical].median().sort_values()
	print(sorted_nb)

	f = plt.figure(figsize=(19, 15))
	fontsize=24
	# Plot volumes increasing order
	ax = df.boxplot(column=sorted_nb.index.values.tolist(), rot=45, fontsize=fontsize)
	ax.set_title('Subcortical volume estimates ', fontsize=fontsize)
	ax.set_ylabel(r'Volume in $mm^3$', fontsize=fontsize) #ax.set_xlabel(' ')
	fig_name = os.path.join(figures_dir, label + '_boxplot_long_Axes.png')
	plt.savefig(fig_name)
	# Plot correlations
	f = plt.figure(figsize=(19, 15))
	plt.matshow(df.corr(method='pearson'), fignum=f.number)
	plt.xticks(range(df.shape[1]), df.columns, fontsize=fontsize, rotation=45)
	plt.yticks(range(df.shape[1]), df.columns, fontsize=fontsize)
	cb = plt.colorbar()
	cb.ax.tick_params(labelsize=fontsize)
	fig_name = os.path.join(figures_dir, label + '_corr_long_Axes.png')
	plt.savefig(fig_name)

def plot_removed_foriso_cortical(indexes, free_counts, label=None):
	""" Plot stacked bar #removed/remain based on isoforest contamination param
	"""
	df = pd.DataFrame({'contamination': indexes, 'free removed':free_counts}, index=indexes)
	ax = df.plot.bar(rot=0)
	ax.set_xlabel('IsoForest Contamination')
	ax.set_ylabel('$\\%$ removed cases')
	fig_name = os.path.join(figures_dir, 'PCremovedIsoforest_' + str(label) + '.png')
	plt.savefig(fig_name)
	return df

def outlier_detection_isoforest_cortical(X, contamination, y=None):
	"""https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html
	"""
	from sklearn.ensemble import IsolationForest
	
	#X.dropna(axis=0,how='any',inplace=True)
	iso = IsolationForest(random_state=10, contamination=contamination, bootstrap=True,max_features=100, n_estimators=10, warm_start=True)
	print(iso)
	yhat = iso.fit_predict(X)
	# select all rows that are not outliers
	mask = yhat != -1
	print('Number of outliers (rows) removed = %.d / %.d' %(sum(mask==False), yhat.shape[0]))
	if y is None:
		return  X[mask]
	else:
		X_train, y_train = X[mask], y[mask]
		return X_train, y_train

def convertdf_into_longitudinal_cortical(df, parcels, thorsf=None):
	""" Get df with measuremennt 1x N per years and transforms it longitudinal Nx1 
	thorsf = 'thick or surf'
	"""

	# Freesurfer
	# added brain volume patt = [s for s in dataframe_orig.columns if "Brain" in s]
	
	frame_free1_parc = df[parcels[0]];frame_free2_parc = df[parcels[1]];frame_free3_parc = df[parcels[2]];frame_free4_parc = df[parcels[3]];frame_free5_parc = df[parcels[4]];frame_free6_parc = df[parcels[5]]
	

	df['nvisita1'] = df['fr_Right_Caudate_y1'].copy(); df['nvisita1'].loc[~df['nvisita1'].isnull()] = 1 ;df['nvisita2'] = df['fr_Right_Caudate_y2'].copy(); df['nvisita2'].loc[~df['nvisita2'].isnull()] = 2 ;df['nvisita3'] = df['fr_Right_Caudate_y3'].copy(); df['nvisita3'].loc[~df['nvisita3'].isnull()] = 3 
	df['nvisita4'] = df['fr_Right_Caudate_y4'].copy(); df['nvisita4'].loc[~df['nvisita4'].isnull()] = 4 ;df['nvisita5'] = df['fr_Right_Caudate_y5'].copy(); df['nvisita5'].loc[~df['nvisita5'].isnull()] = 5 ;df['nvisita6'] = df['fr_Right_Caudate_y6'].copy(); df['nvisita6'].loc[~df['nvisita6'].isnull()] = 6 

	df['nvisita1'] =1; df['nvisita2']=2;df['nvisita3']=3;df['nvisita4']=4;df['nvisita5']=5;df['nvisita6']=6
	if thorsf == 'thick':
		
		#frame_free1 = {'handlat': df['lat_manual'],'apoe': df['apoe'], 'y_last':df['ultimavisita'], 'dx_last': df['ultimodx'], 'dx_visita': df['dx_corto_visita1'], 'age_lastdx': df['edad_ultimodx'], 'age': df['edad_visita1'], 'sex': df['sexo'], 'free_R_Thal': df['fr_Right_Thalamus_Proper_y1'], 'free_L_Thal': df['fr_Left_Thalamus_Proper_y1'],'free_R_Puta': df['fr_Right_Putamen_y1'], 'free_L_Puta': df['fr_Left_Putamen_y1'],'free_R_Amyg': df['fr_Right_Amygdala_y1'], 'free_L_Amyg': df['fr_Left_Amygdala_y1'],'free_R_Pall': df['fr_Right_Pallidum_y1'], 'free_L_Pall': df['fr_Left_Pallidum_y1'],'free_R_Caud': df['fr_Right_Caudate_y1'], 'free_L_Caud': df['fr_Left_Caudate_y1'],'free_R_Hipp': df['fr_Right_Hippocampus_y1'], 'free_L_Hipp': df['fr_Left_Hippocampus_y1'],'free_R_Accu': df['fr_Right_Accumbens_area_y1'], 'free_L_Accu': df['fr_Left_Accumbens_area_y1'], 'fr_BrainSegVol': df['fr_BrainSegVol_y1'], 'fr_BrainSegVolNotVent':df['fr_BrainSegVolNotVent_y1'], 'fr_BrainSegVol_to_eTIV': df['fr_BrainSegVol_to_eTIV_y1']}
		frame_free1 = {'education': df['nivel_educativo'],'handlat': df['lat_manual'],'apoe': df['apoe'],'nvisita':df['nvisita1'] ,'y_last':df['ultimavisita'], 'dx_last': df['ultimodx'], 'dx_visita': df['dx_corto_visita1'], 'age_lastdx': df['edad_ultimodx'], 'age': df['edad_visita1'], 'sex': df['sexo'], 'free_R_Thal': df['fr_Right_Thalamus_Proper_y1'], 'free_L_Thal': df['fr_Left_Thalamus_Proper_y1'],'free_R_Puta': df['fr_Right_Putamen_y1'], 'free_L_Puta': df['fr_Left_Putamen_y1'],'free_R_Amyg': df['fr_Right_Amygdala_y1'], 'free_L_Amyg': df['fr_Left_Amygdala_y1'],'free_R_Pall': df['fr_Right_Pallidum_y1'], 'free_L_Pall': df['fr_Left_Pallidum_y1'],'free_R_Caud': df['fr_Right_Caudate_y1'], 'free_L_Caud': df['fr_Left_Caudate_y1'],'free_R_Hipp': df['fr_Right_Hippocampus_y1'], 'free_L_Hipp': df['fr_Left_Hippocampus_y1'],'free_R_Accu': df['fr_Right_Accumbens_area_y1'], 'free_L_Accu': df['fr_Left_Accumbens_area_y1'], 'fr_BrainSegVol': df['fr_BrainSegVol_y1'], 'fr_BrainSegVolNotVent':df['fr_BrainSegVolNotVent_y1'], 'fr_BrainSegVol_to_eTIV': df['fr_BrainSegVol_to_eTIV_y1'], 'fr_R_thick_G_Ins_lgandS_cent_ins': df['fr_R_thick_G_Ins_lgandS_cent_ins_y1'], 'fr_R_thick_G_cingul_Post_ventral': df['fr_R_thick_G_cingul_Post_ventral_y1'], 'fr_R_thick_G_cuneus': df['fr_R_thick_G_cuneus_y1'], 'fr_R_thick_G_front_inf_Opercular': df['fr_R_thick_G_front_inf_Opercular_y1'], 'fr_R_thick_G_front_inf_Orbital': df['fr_R_thick_G_front_inf_Orbital_y1'], 'fr_R_thick_G_front_inf_Triangul': df['fr_R_thick_G_front_inf_Triangul_y1'], 'fr_R_thick_G_front_middle': df['fr_R_thick_G_front_middle_y1'], 'fr_R_thick_G_front_sup': df['fr_R_thick_G_front_sup_y1'], 'fr_R_thick_G_insular_short': df['fr_R_thick_G_insular_short_y1'], 'fr_R_thick_G_oc_temp_lat_fusifor': df['fr_R_thick_G_oc_temp_lat_fusifor_y1'], 'fr_R_thick_G_oc_temp_med_Lingual': df['fr_R_thick_G_oc_temp_med_Lingual_y1'], 'fr_R_thick_G_oc_temp_med_Parahip': df['fr_R_thick_G_oc_temp_med_Parahip_y1'], 'fr_R_thick_G_occipital_middle': df['fr_R_thick_G_occipital_middle_y1'], 'fr_R_thick_G_occipital_sup': df['fr_R_thick_G_occipital_sup_y1'], 'fr_R_thick_G_orbital': df['fr_R_thick_G_orbital_y1'], 'fr_R_thick_G_pariet_inf_Angular': df['fr_R_thick_G_pariet_inf_Angular_y1'], 'fr_R_thick_G_pariet_inf_Supramar': df['fr_R_thick_G_pariet_inf_Supramar_y1'], 'fr_R_thick_G_parietal_sup': df['fr_R_thick_G_parietal_sup_y1'], 'fr_R_thick_G_postcentral': df['fr_R_thick_G_postcentral_y1'], 'fr_R_thick_G_precentral': df['fr_R_thick_G_precentral_y1'], 'fr_R_thick_G_precuneus': df['fr_R_thick_G_precuneus_y1'], 'fr_R_thick_G_rectus': df['fr_R_thick_G_rectus_y1'], 'fr_R_thick_G_temp_sup_G_T_transv': df['fr_R_thick_G_temp_sup_G_T_transv_y1'], 'fr_R_thick_G_temp_sup_Lateral': df['fr_R_thick_G_temp_sup_Lateral_y1'], 'fr_R_thick_G_temp_sup_Plan_polar': df['fr_R_thick_G_temp_sup_Plan_polar_y1'], 'fr_R_thick_G_temp_sup_Plan_tempo': df['fr_R_thick_G_temp_sup_Plan_tempo_y1'], 'fr_R_thick_G_temporal_inf': df['fr_R_thick_G_temporal_inf_y1'], 'fr_R_thick_G_temporal_middle': df['fr_R_thick_G_temporal_middle_y1'], 'fr_R_thick_GandS_cingul_Ant': df['fr_R_thick_GandS_cingul_Ant_y1'], 'fr_R_thick_GandS_cingul_Mid_Post': df['fr_R_thick_GandS_cingul_Mid_Post_y1'], 'fr_R_thick_GandS_frontomargin': df['fr_R_thick_GandS_frontomargin_y1'], 'fr_R_thick_GandS_occipital_inf': df['fr_R_thick_GandS_occipital_inf_y1'], 'fr_R_thick_GandS_paracentral': df['fr_R_thick_GandS_paracentral_y1'], 'fr_R_thick_GandS_subcentral': df['fr_R_thick_GandS_subcentral_y1'], 'fr_R_thick_GandS_transv_frontopol': df['fr_R_thick_GandS_transv_frontopol_y1'], 'fr_R_thick_Lat_Fis_ant_Horizont': df['fr_R_thick_Lat_Fis_ant_Horizont_y1'], 'fr_R_thick_Lat_Fis_post': df['fr_R_thick_Lat_Fis_post_y1'], 'fr_R_thick_Pole_occipital': df['fr_R_thick_Pole_occipital_y1'], 'fr_R_thick_Pole_temporal': df['fr_R_thick_Pole_temporal_y1'], 'fr_R_thick_S_calcarine': df['fr_R_thick_S_calcarine_y1'], 'fr_R_thick_S_central': df['fr_R_thick_S_central_y1'], 'fr_R_thick_S_cingul_Marginalis': df['fr_R_thick_S_cingul_Marginalis_y1'], 'fr_R_thick_S_circular_insula_ant': df['fr_R_thick_S_circular_insula_ant_y1'], 'fr_R_thick_S_circular_insula_inf': df['fr_R_thick_S_circular_insula_inf_y1'], 'fr_R_thick_S_circular_insula_sup': df['fr_R_thick_S_circular_insula_sup_y1'], 'fr_R_thick_S_collat_transv_ant': df['fr_R_thick_S_collat_transv_ant_y1'], 'fr_R_thick_S_collat_transv_post': df['fr_R_thick_S_collat_transv_post_y1'], 'fr_R_thick_S_front_inf': df['fr_R_thick_S_front_inf_y1'], 'fr_R_thick_S_front_middle': df['fr_R_thick_S_front_middle_y1'], 'fr_R_thick_S_front_sup': df['fr_R_thick_S_front_sup_y1'], 'fr_R_thick_S_interm_prim_Jensen': df['fr_R_thick_S_interm_prim_Jensen_y1'], 'fr_R_thick_S_intraparietandP_trans': df['fr_R_thick_S_intraparietandP_trans_y1'], 'fr_R_thick_S_oc_middleandLunatus': df['fr_R_thick_S_oc_middleandLunatus_y1'], 'fr_R_thick_S_oc_temp_lat': df['fr_R_thick_S_oc_temp_lat_y1'], 'fr_R_thick_S_oc_temp_medandLingual': df['fr_R_thick_S_oc_temp_medandLingual_y1'], 'fr_R_thick_S_orbital_H_Shaped': df['fr_R_thick_S_orbital_H_Shaped_y1'], 'fr_R_thick_S_orbital_lateral': df['fr_R_thick_S_orbital_lateral_y1'], 'fr_R_thick_S_orbital_med_olfact': df['fr_R_thick_S_orbital_med_olfact_y1'], 'fr_R_thick_S_parieto_occipital': df['fr_R_thick_S_parieto_occipital_y1'], 'fr_R_thick_S_pericallosal': df['fr_R_thick_S_pericallosal_y1'], 'fr_R_thick_S_postcentral': df['fr_R_thick_S_postcentral_y1'], 'fr_R_thick_S_precentral_inf_part': df['fr_R_thick_S_precentral_inf_part_y1'], 'fr_R_thick_S_precentral_sup_part': df['fr_R_thick_S_precentral_sup_part_y1'], 'fr_R_thick_S_suborbital': df['fr_R_thick_S_suborbital_y1'], 'fr_R_thick_S_subparietal': df['fr_R_thick_S_subparietal_y1'], 'fr_R_thick_S_temporal_inf': df['fr_R_thick_S_temporal_inf_y1'], 'fr_R_thick_S_temporal_sup': df['fr_R_thick_S_temporal_sup_y1'], 'fr_R_thick_S_temporal_transverse': df['fr_R_thick_S_temporal_transverse_y1'],'fr_L_thick_G_Ins_lgandS_cent_ins': df['fr_L_thick_G_Ins_lgandS_cent_ins_y1'], 'fr_L_thick_G_cingul_Post_ventral': df['fr_L_thick_G_cingul_Post_ventral_y1'], 'fr_L_thick_G_cuneus': df['fr_L_thick_G_cuneus_y1'], 'fr_L_thick_G_front_inf_Opercular': df['fr_L_thick_G_front_inf_Opercular_y1'], 'fr_L_thick_G_front_inf_Orbital': df['fr_L_thick_G_front_inf_Orbital_y1'], 'fr_L_thick_G_front_inf_Triangul': df['fr_L_thick_G_front_inf_Triangul_y1'], 'fr_L_thick_G_front_middle': df['fr_L_thick_G_front_middle_y1'], 'fr_L_thick_G_front_sup': df['fr_L_thick_G_front_sup_y1'], 'fr_L_thick_G_insular_short': df['fr_L_thick_G_insular_short_y1'], 'fr_L_thick_G_oc_temp_lat_fusifor': df['fr_L_thick_G_oc_temp_lat_fusifor_y1'], 'fr_L_thick_G_oc_temp_med_Lingual': df['fr_L_thick_G_oc_temp_med_Lingual_y1'], 'fr_L_thick_G_oc_temp_med_Parahip': df['fr_L_thick_G_oc_temp_med_Parahip_y1'], 'fr_L_thick_G_occipital_middle': df['fr_L_thick_G_occipital_middle_y1'], 'fr_L_thick_G_occipital_sup': df['fr_L_thick_G_occipital_sup_y1'], 'fr_L_thick_G_orbital': df['fr_L_thick_G_orbital_y1'], 'fr_L_thick_G_pariet_inf_Angular': df['fr_L_thick_G_pariet_inf_Angular_y1'], 'fr_L_thick_G_pariet_inf_Supramar': df['fr_L_thick_G_pariet_inf_Supramar_y1'], 'fr_L_thick_G_parietal_sup': df['fr_L_thick_G_parietal_sup_y1'], 'fr_L_thick_G_postcentral': df['fr_L_thick_G_postcentral_y1'], 'fr_L_thick_G_precentral': df['fr_L_thick_G_precentral_y1'], 'fr_L_thick_G_precuneus': df['fr_L_thick_G_precuneus_y1'], 'fr_L_thick_G_rectus': df['fr_L_thick_G_rectus_y1'], 'fr_L_thick_G_temp_sup_G_T_transv': df['fr_L_thick_G_temp_sup_G_T_transv_y1'], 'fr_L_thick_G_temp_sup_Lateral': df['fr_L_thick_G_temp_sup_Lateral_y1'], 'fr_L_thick_G_temp_sup_Plan_polar': df['fr_L_thick_G_temp_sup_Plan_polar_y1'], 'fr_L_thick_G_temp_sup_Plan_tempo': df['fr_L_thick_G_temp_sup_Plan_tempo_y1'], 'fr_L_thick_G_temporal_inf': df['fr_L_thick_G_temporal_inf_y1'], 'fr_L_thick_G_temporal_middle': df['fr_L_thick_G_temporal_middle_y1'], 'fr_L_thick_GandS_cingul_Ant': df['fr_L_thick_GandS_cingul_Ant_y1'], 'fr_L_thick_GandS_cingul_Mid_Post': df['fr_L_thick_GandS_cingul_Mid_Post_y1'], 'fr_L_thick_GandS_frontomargin': df['fr_L_thick_GandS_frontomargin_y1'], 'fr_L_thick_GandS_occipital_inf': df['fr_L_thick_GandS_occipital_inf_y1'], 'fr_L_thick_GandS_paracentral': df['fr_L_thick_GandS_paracentral_y1'], 'fr_L_thick_GandS_subcentral': df['fr_L_thick_GandS_subcentral_y1'], 'fr_L_thick_GandS_transv_frontopol': df['fr_L_thick_GandS_transv_frontopol_y1'], 'fr_L_thick_Lat_Fis_ant_Horizont': df['fr_L_thick_Lat_Fis_ant_Horizont_y1'], 'fr_L_thick_Lat_Fis_post': df['fr_L_thick_Lat_Fis_post_y1'], 'fr_L_thick_Pole_occipital': df['fr_L_thick_Pole_occipital_y1'], 'fr_L_thick_Pole_temporal': df['fr_L_thick_Pole_temporal_y1'], 'fr_L_thick_S_calcarine': df['fr_L_thick_S_calcarine_y1'], 'fr_L_thick_S_central': df['fr_L_thick_S_central_y1'], 'fr_L_thick_S_cingul_Marginalis': df['fr_L_thick_S_cingul_Marginalis_y1'], 'fr_L_thick_S_circular_insula_ant': df['fr_L_thick_S_circular_insula_ant_y1'], 'fr_L_thick_S_circular_insula_inf': df['fr_L_thick_S_circular_insula_inf_y1'], 'fr_L_thick_S_circular_insula_sup': df['fr_L_thick_S_circular_insula_sup_y1'], 'fr_L_thick_S_collat_transv_ant': df['fr_L_thick_S_collat_transv_ant_y1'], 'fr_L_thick_S_collat_transv_post': df['fr_L_thick_S_collat_transv_post_y1'], 'fr_L_thick_S_front_inf': df['fr_L_thick_S_front_inf_y1'], 'fr_L_thick_S_front_middle': df['fr_L_thick_S_front_middle_y1'], 'fr_L_thick_S_front_sup': df['fr_L_thick_S_front_sup_y1'],  'fr_L_thick_S_intraparietandP_trans': df['fr_L_thick_S_intraparietandP_trans_y1'], 'fr_L_thick_S_oc_middleandLunatus': df['fr_L_thick_S_oc_middleandLunatus_y1'], 'fr_L_thick_S_oc_temp_lat': df['fr_L_thick_S_oc_temp_lat_y1'], 'fr_L_thick_S_oc_temp_medandLingual': df['fr_L_thick_S_oc_temp_medandLingual_y1'], 'fr_L_thick_S_orbital_H_Shaped': df['fr_L_thick_S_orbital_H_Shaped_y1'], 'fr_L_thick_S_orbital_lateral': df['fr_L_thick_S_orbital_lateral_y1'], 'fr_L_thick_S_orbital_med_olfact': df['fr_L_thick_S_orbital_med_olfact_y1'], 'fr_L_thick_S_parieto_occipital': df['fr_L_thick_S_parieto_occipital_y1'], 'fr_L_thick_S_postcentral': df['fr_L_thick_S_postcentral_y1'], 'fr_L_thick_S_precentral_inf_part': df['fr_L_thick_S_precentral_inf_part_y1'], 'fr_L_thick_S_precentral_sup_part': df['fr_L_thick_S_precentral_sup_part_y1'], 'fr_L_thick_S_suborbital': df['fr_L_thick_S_suborbital_y1'], 'fr_L_thick_S_subparietal': df['fr_L_thick_S_subparietal_y1'], 'fr_L_thick_S_temporal_inf': df['fr_L_thick_S_temporal_inf_y1'], 'fr_L_thick_S_temporal_sup': df['fr_L_thick_S_temporal_sup_y1']}
		#frame_free2 = {'handlat': df['lat_manual'],'apoe': df['apoe'], 'y_last':df['ultimavisita'], 'dx_last': df['ultimodx'], 'dx_visita': df['dx_corto_visita2'], 'age_lastdx': df['edad_ultimodx'], 'age': df['edad_visita2'], 'sex': df['sexo'], 'free_R_Thal': df['fr_Right_Thalamus_Proper_y2'], 'free_L_Thal': df['fr_Left_Thalamus_Proper_y2'],'free_R_Puta': df['fr_Right_Putamen_y2'], 'free_L_Puta': df['fr_Left_Putamen_y2'],'free_R_Amyg': df['fr_Right_Amygdala_y2'], 'free_L_Amyg': df['fr_Left_Amygdala_y2'],'free_R_Pall': df['fr_Right_Pallidum_y2'], 'free_L_Pall': df['fr_Left_Pallidum_y2'],'free_R_Caud': df['fr_Right_Caudate_y2'], 'free_L_Caud': df['fr_Left_Caudate_y2'],'free_R_Hipp': df['fr_Right_Hippocampus_y2'], 'free_L_Hipp': df['fr_Left_Hippocampus_y2'],'free_R_Accu': df['fr_Right_Accumbens_area_y2'], 'free_L_Accu': df['fr_Left_Accumbens_area_y2'], 'fr_BrainSegVol': df['fr_BrainSegVol_y2'], 'fr_BrainSegVolNotVent':df['fr_BrainSegVolNotVent_y2'], 'fr_BrainSegVol_to_eTIV': df['fr_BrainSegVol_to_eTIV_y2']}
		frame_free2 = {'education': df['nivel_educativo'],'handlat': df['lat_manual'],'apoe': df['apoe'], 'nvisita':df['nvisita2'], 'y_last':df['ultimavisita'], 'dx_last': df['ultimodx'], 'dx_visita': df['dx_corto_visita2'], 'age_lastdx': df['edad_ultimodx'], 'age': df['edad_visita2'], 'sex': df['sexo'], 'free_R_Thal': df['fr_Right_Thalamus_Proper_y2'], 'free_L_Thal': df['fr_Left_Thalamus_Proper_y2'],'free_R_Puta': df['fr_Right_Putamen_y2'], 'free_L_Puta': df['fr_Left_Putamen_y2'],'free_R_Amyg': df['fr_Right_Amygdala_y2'], 'free_L_Amyg': df['fr_Left_Amygdala_y2'],'free_R_Pall': df['fr_Right_Pallidum_y2'], 'free_L_Pall': df['fr_Left_Pallidum_y2'],'free_R_Caud': df['fr_Right_Caudate_y2'], 'free_L_Caud': df['fr_Left_Caudate_y2'],'free_R_Hipp': df['fr_Right_Hippocampus_y2'], 'free_L_Hipp': df['fr_Left_Hippocampus_y2'],'free_R_Accu': df['fr_Right_Accumbens_area_y2'], 'free_L_Accu': df['fr_Left_Accumbens_area_y2'], 'fr_BrainSegVol': df['fr_BrainSegVol_y2'], 'fr_BrainSegVolNotVent':df['fr_BrainSegVolNotVent_y2'], 'fr_BrainSegVol_to_eTIV': df['fr_BrainSegVol_to_eTIV_y2'], 'fr_R_thick_G_Ins_lgandS_cent_ins': df['fr_R_thick_G_Ins_lgandS_cent_ins_y2'], 'fr_R_thick_G_cingul_Post_ventral': df['fr_R_thick_G_cingul_Post_ventral_y2'], 'fr_R_thick_G_cuneus': df['fr_R_thick_G_cuneus_y2'], 'fr_R_thick_G_front_inf_Opercular': df['fr_R_thick_G_front_inf_Opercular_y2'], 'fr_R_thick_G_front_inf_Orbital': df['fr_R_thick_G_front_inf_Orbital_y2'], 'fr_R_thick_G_front_inf_Triangul': df['fr_R_thick_G_front_inf_Triangul_y2'], 'fr_R_thick_G_front_middle': df['fr_R_thick_G_front_middle_y2'], 'fr_R_thick_G_front_sup': df['fr_R_thick_G_front_sup_y2'], 'fr_R_thick_G_insular_short': df['fr_R_thick_G_insular_short_y2'], 'fr_R_thick_G_oc_temp_lat_fusifor': df['fr_R_thick_G_oc_temp_lat_fusifor_y2'], 'fr_R_thick_G_oc_temp_med_Lingual': df['fr_R_thick_G_oc_temp_med_Lingual_y2'], 'fr_R_thick_G_oc_temp_med_Parahip': df['fr_R_thick_G_oc_temp_med_Parahip_y2'], 'fr_R_thick_G_occipital_middle': df['fr_R_thick_G_occipital_middle_y2'], 'fr_R_thick_G_occipital_sup': df['fr_R_thick_G_occipital_sup_y2'], 'fr_R_thick_G_orbital': df['fr_R_thick_G_orbital_y2'], 'fr_R_thick_G_pariet_inf_Angular': df['fr_R_thick_G_pariet_inf_Angular_y2'], 'fr_R_thick_G_pariet_inf_Supramar': df['fr_R_thick_G_pariet_inf_Supramar_y2'], 'fr_R_thick_G_parietal_sup': df['fr_R_thick_G_parietal_sup_y2'], 'fr_R_thick_G_postcentral': df['fr_R_thick_G_postcentral_y2'], 'fr_R_thick_G_precentral': df['fr_R_thick_G_precentral_y2'], 'fr_R_thick_G_precuneus': df['fr_R_thick_G_precuneus_y2'], 'fr_R_thick_G_rectus': df['fr_R_thick_G_rectus_y2'], 'fr_R_thick_G_temp_sup_G_T_transv': df['fr_R_thick_G_temp_sup_G_T_transv_y2'], 'fr_R_thick_G_temp_sup_Lateral': df['fr_R_thick_G_temp_sup_Lateral_y2'], 'fr_R_thick_G_temp_sup_Plan_polar': df['fr_R_thick_G_temp_sup_Plan_polar_y2'], 'fr_R_thick_G_temp_sup_Plan_tempo': df['fr_R_thick_G_temp_sup_Plan_tempo_y2'], 'fr_R_thick_G_temporal_inf': df['fr_R_thick_G_temporal_inf_y2'], 'fr_R_thick_G_temporal_middle': df['fr_R_thick_G_temporal_middle_y2'], 'fr_R_thick_GandS_cingul_Ant': df['fr_R_thick_GandS_cingul_Ant_y2'], 'fr_R_thick_GandS_cingul_Mid_Post': df['fr_R_thick_GandS_cingul_Mid_Post_y2'], 'fr_R_thick_GandS_frontomargin': df['fr_R_thick_GandS_frontomargin_y2'], 'fr_R_thick_GandS_occipital_inf': df['fr_R_thick_GandS_occipital_inf_y2'], 'fr_R_thick_GandS_paracentral': df['fr_R_thick_GandS_paracentral_y2'], 'fr_R_thick_GandS_subcentral': df['fr_R_thick_GandS_subcentral_y2'], 'fr_R_thick_GandS_transv_frontopol': df['fr_R_thick_GandS_transv_frontopol_y2'], 'fr_R_thick_Lat_Fis_ant_Horizont': df['fr_R_thick_Lat_Fis_ant_Horizont_y2'], 'fr_R_thick_Lat_Fis_post': df['fr_R_thick_Lat_Fis_post_y2'], 'fr_R_thick_Pole_occipital': df['fr_R_thick_Pole_occipital_y2'], 'fr_R_thick_Pole_temporal': df['fr_R_thick_Pole_temporal_y2'], 'fr_R_thick_S_calcarine': df['fr_R_thick_S_calcarine_y2'], 'fr_R_thick_S_central': df['fr_R_thick_S_central_y2'], 'fr_R_thick_S_cingul_Marginalis': df['fr_R_thick_S_cingul_Marginalis_y2'], 'fr_R_thick_S_circular_insula_ant': df['fr_R_thick_S_circular_insula_ant_y2'], 'fr_R_thick_S_circular_insula_inf': df['fr_R_thick_S_circular_insula_inf_y2'], 'fr_R_thick_S_circular_insula_sup': df['fr_R_thick_S_circular_insula_sup_y2'], 'fr_R_thick_S_collat_transv_ant': df['fr_R_thick_S_collat_transv_ant_y2'], 'fr_R_thick_S_collat_transv_post': df['fr_R_thick_S_collat_transv_post_y2'], 'fr_R_thick_S_front_inf': df['fr_R_thick_S_front_inf_y2'], 'fr_R_thick_S_front_middle': df['fr_R_thick_S_front_middle_y2'], 'fr_R_thick_S_front_sup': df['fr_R_thick_S_front_sup_y2'], 'fr_R_thick_S_interm_prim_Jensen': df['fr_R_thick_S_interm_prim_Jensen_y2'], 'fr_R_thick_S_intraparietandP_trans': df['fr_R_thick_S_intraparietandP_trans_y2'], 'fr_R_thick_S_oc_middleandLunatus': df['fr_R_thick_S_oc_middleandLunatus_y2'], 'fr_R_thick_S_oc_temp_lat': df['fr_R_thick_S_oc_temp_lat_y2'], 'fr_R_thick_S_oc_temp_medandLingual': df['fr_R_thick_S_oc_temp_medandLingual_y2'], 'fr_R_thick_S_orbital_H_Shaped': df['fr_R_thick_S_orbital_H_Shaped_y2'], 'fr_R_thick_S_orbital_lateral': df['fr_R_thick_S_orbital_lateral_y2'], 'fr_R_thick_S_orbital_med_olfact': df['fr_R_thick_S_orbital_med_olfact_y2'], 'fr_R_thick_S_parieto_occipital': df['fr_R_thick_S_parieto_occipital_y2'], 'fr_R_thick_S_pericallosal': df['fr_R_thick_S_pericallosal_y2'], 'fr_R_thick_S_postcentral': df['fr_R_thick_S_postcentral_y2'], 'fr_R_thick_S_precentral_inf_part': df['fr_R_thick_S_precentral_inf_part_y2'], 'fr_R_thick_S_precentral_sup_part': df['fr_R_thick_S_precentral_sup_part_y2'], 'fr_R_thick_S_suborbital': df['fr_R_thick_S_suborbital_y2'], 'fr_R_thick_S_subparietal': df['fr_R_thick_S_subparietal_y2'], 'fr_R_thick_S_temporal_inf': df['fr_R_thick_S_temporal_inf_y2'], 'fr_R_thick_S_temporal_sup': df['fr_R_thick_S_temporal_sup_y2'], 'fr_R_thick_S_temporal_transverse': df['fr_R_thick_S_temporal_transverse_y2'], 'fr_L_thick_G_Ins_lgandS_cent_ins': df['fr_L_thick_G_Ins_lgandS_cent_ins_y2'], 'fr_L_thick_G_cingul_Post_ventral': df['fr_L_thick_G_cingul_Post_ventral_y2'], 'fr_L_thick_G_cuneus': df['fr_L_thick_G_cuneus_y2'], 'fr_L_thick_G_front_inf_Opercular': df['fr_L_thick_G_front_inf_Opercular_y2'], 'fr_L_thick_G_front_inf_Orbital': df['fr_L_thick_G_front_inf_Orbital_y2'], 'fr_L_thick_G_front_inf_Triangul': df['fr_L_thick_G_front_inf_Triangul_y2'], 'fr_L_thick_G_front_middle': df['fr_L_thick_G_front_middle_y2'], 'fr_L_thick_G_front_sup': df['fr_L_thick_G_front_sup_y2'], 'fr_L_thick_G_insular_short': df['fr_L_thick_G_insular_short_y2'], 'fr_L_thick_G_oc_temp_lat_fusifor': df['fr_L_thick_G_oc_temp_lat_fusifor_y2'], 'fr_L_thick_G_oc_temp_med_Lingual': df['fr_L_thick_G_oc_temp_med_Lingual_y2'], 'fr_L_thick_G_oc_temp_med_Parahip': df['fr_L_thick_G_oc_temp_med_Parahip_y2'], 'fr_L_thick_G_occipital_middle': df['fr_L_thick_G_occipital_middle_y2'], 'fr_L_thick_G_occipital_sup': df['fr_L_thick_G_occipital_sup_y2'], 'fr_L_thick_G_orbital': df['fr_L_thick_G_orbital_y2'], 'fr_L_thick_G_pariet_inf_Angular': df['fr_L_thick_G_pariet_inf_Angular_y2'], 'fr_L_thick_G_pariet_inf_Supramar': df['fr_L_thick_G_pariet_inf_Supramar_y2'], 'fr_L_thick_G_parietal_sup': df['fr_L_thick_G_parietal_sup_y2'], 'fr_L_thick_G_postcentral': df['fr_L_thick_G_postcentral_y2'], 'fr_L_thick_G_precentral': df['fr_L_thick_G_precentral_y2'], 'fr_L_thick_G_precuneus': df['fr_L_thick_G_precuneus_y2'], 'fr_L_thick_G_rectus': df['fr_L_thick_G_rectus_y2'], 'fr_L_thick_G_temp_sup_G_T_transv': df['fr_L_thick_G_temp_sup_G_T_transv_y2'], 'fr_L_thick_G_temp_sup_Lateral': df['fr_L_thick_G_temp_sup_Lateral_y2'], 'fr_L_thick_G_temp_sup_Plan_polar': df['fr_L_thick_G_temp_sup_Plan_polar_y2'], 'fr_L_thick_G_temp_sup_Plan_tempo': df['fr_L_thick_G_temp_sup_Plan_tempo_y2'], 'fr_L_thick_G_temporal_inf': df['fr_L_thick_G_temporal_inf_y2'], 'fr_L_thick_G_temporal_middle': df['fr_L_thick_G_temporal_middle_y2'], 'fr_L_thick_GandS_cingul_Ant': df['fr_L_thick_GandS_cingul_Ant_y2'], 'fr_L_thick_GandS_cingul_Mid_Post': df['fr_L_thick_GandS_cingul_Mid_Post_y2'], 'fr_L_thick_GandS_frontomargin': df['fr_L_thick_GandS_frontomargin_y2'], 'fr_L_thick_GandS_occipital_inf': df['fr_L_thick_GandS_occipital_inf_y2'], 'fr_L_thick_GandS_paracentral': df['fr_L_thick_GandS_paracentral_y2'], 'fr_L_thick_GandS_subcentral': df['fr_L_thick_GandS_subcentral_y2'], 'fr_L_thick_GandS_transv_frontopol': df['fr_L_thick_GandS_transv_frontopol_y2'], 'fr_L_thick_Lat_Fis_ant_Horizont': df['fr_L_thick_Lat_Fis_ant_Horizont_y2'], 'fr_L_thick_Lat_Fis_post': df['fr_L_thick_Lat_Fis_post_y2'], 'fr_L_thick_Pole_occipital': df['fr_L_thick_Pole_occipital_y2'], 'fr_L_thick_Pole_temporal': df['fr_L_thick_Pole_temporal_y2'], 'fr_L_thick_S_calcarine': df['fr_L_thick_S_calcarine_y2'], 'fr_L_thick_S_central': df['fr_L_thick_S_central_y2'], 'fr_L_thick_S_cingul_Marginalis': df['fr_L_thick_S_cingul_Marginalis_y2'], 'fr_L_thick_S_circular_insula_ant': df['fr_L_thick_S_circular_insula_ant_y2'], 'fr_L_thick_S_circular_insula_inf': df['fr_L_thick_S_circular_insula_inf_y2'], 'fr_L_thick_S_circular_insula_sup': df['fr_L_thick_S_circular_insula_sup_y2'], 'fr_L_thick_S_collat_transv_ant': df['fr_L_thick_S_collat_transv_ant_y2'], 'fr_L_thick_S_collat_transv_post': df['fr_L_thick_S_collat_transv_post_y2'], 'fr_L_thick_S_front_inf': df['fr_L_thick_S_front_inf_y2'], 'fr_L_thick_S_front_middle': df['fr_L_thick_S_front_middle_y2'], 'fr_L_thick_S_front_sup': df['fr_L_thick_S_front_sup_y2'], 'fr_L_thick_S_intraparietandP_trans': df['fr_L_thick_S_intraparietandP_trans_y2'], 'fr_L_thick_S_oc_middleandLunatus': df['fr_L_thick_S_oc_middleandLunatus_y2'], 'fr_L_thick_S_oc_temp_lat': df['fr_L_thick_S_oc_temp_lat_y2'], 'fr_L_thick_S_oc_temp_medandLingual': df['fr_L_thick_S_oc_temp_medandLingual_y2'], 'fr_L_thick_S_orbital_H_Shaped': df['fr_L_thick_S_orbital_H_Shaped_y2'], 'fr_L_thick_S_orbital_lateral': df['fr_L_thick_S_orbital_lateral_y2'], 'fr_L_thick_S_orbital_med_olfact': df['fr_L_thick_S_orbital_med_olfact_y2'], 'fr_L_thick_S_parieto_occipital': df['fr_L_thick_S_parieto_occipital_y2'], 'fr_L_thick_S_postcentral': df['fr_L_thick_S_postcentral_y2'], 'fr_L_thick_S_precentral_inf_part': df['fr_L_thick_S_precentral_inf_part_y2'], 'fr_L_thick_S_precentral_sup_part': df['fr_L_thick_S_precentral_sup_part_y2'], 'fr_L_thick_S_suborbital': df['fr_L_thick_S_suborbital_y2'], 'fr_L_thick_S_subparietal': df['fr_L_thick_S_subparietal_y2'], 'fr_L_thick_S_temporal_inf': df['fr_L_thick_S_temporal_inf_y2'], 'fr_L_thick_S_temporal_sup': df['fr_L_thick_S_temporal_sup_y2']}
		#frame_free3 = {'handlat': df['lat_manual'],'apoe': df['apoe'], 'y_last':df['ultimavisita'], 'dx_last': df['ultimodx'], 'dx_visita': df['dx_corto_visita3'], 'age_lastdx': df['edad_ultimodx'], 'age': df['edad_visita3'], 'sex': df['sexo'], 'free_R_Thal': df['fr_Right_Thalamus_Proper_y3'], 'free_L_Thal': df['fr_Left_Thalamus_Proper_y3'],'free_R_Puta': df['fr_Right_Putamen_y3'], 'free_L_Puta': df['fr_Left_Putamen_y3'],'free_R_Amyg': df['fr_Right_Amygdala_y3'], 'free_L_Amyg': df['fr_Left_Amygdala_y3'],'free_R_Pall': df['fr_Right_Pallidum_y3'], 'free_L_Pall': df['fr_Left_Pallidum_y3'],'free_R_Caud': df['fr_Right_Caudate_y3'], 'free_L_Caud': df['fr_Left_Caudate_y3'],'free_R_Hipp': df['fr_Right_Hippocampus_y3'], 'free_L_Hipp': df['fr_Left_Hippocampus_y3'],'free_R_Accu': df['fr_Right_Accumbens_area_y3'], 'free_L_Accu': df['fr_Left_Accumbens_area_y3'], 'fr_BrainSegVol': df['fr_BrainSegVol_y3'], 'fr_BrainSegVolNotVent':df['fr_BrainSegVolNotVent_y3'], 'fr_BrainSegVol_to_eTIV': df['fr_BrainSegVol_to_eTIV_y3']}
		frame_free3 = {'education': df['nivel_educativo'],'handlat': df['lat_manual'],'apoe': df['apoe'], 'nvisita':df['nvisita3'] ,'y_last':df['ultimavisita'], 'dx_last': df['ultimodx'], 'dx_visita': df['dx_corto_visita3'], 'age_lastdx': df['edad_ultimodx'], 'age': df['edad_visita3'], 'sex': df['sexo'], 'free_R_Thal': df['fr_Right_Thalamus_Proper_y3'], 'free_L_Thal': df['fr_Left_Thalamus_Proper_y3'],'free_R_Puta': df['fr_Right_Putamen_y3'], 'free_L_Puta': df['fr_Left_Putamen_y3'],'free_R_Amyg': df['fr_Right_Amygdala_y3'], 'free_L_Amyg': df['fr_Left_Amygdala_y3'],'free_R_Pall': df['fr_Right_Pallidum_y3'], 'free_L_Pall': df['fr_Left_Pallidum_y3'],'free_R_Caud': df['fr_Right_Caudate_y3'], 'free_L_Caud': df['fr_Left_Caudate_y3'],'free_R_Hipp': df['fr_Right_Hippocampus_y3'], 'free_L_Hipp': df['fr_Left_Hippocampus_y3'],'free_R_Accu': df['fr_Right_Accumbens_area_y3'], 'free_L_Accu': df['fr_Left_Accumbens_area_y3'], 'fr_BrainSegVol': df['fr_BrainSegVol_y3'], 'fr_BrainSegVolNotVent':df['fr_BrainSegVolNotVent_y3'], 'fr_BrainSegVol_to_eTIV': df['fr_BrainSegVol_to_eTIV_y3'], 'fr_R_thick_G_Ins_lgandS_cent_ins': df['fr_R_thick_G_Ins_lgandS_cent_ins_y3'], 'fr_R_thick_G_cingul_Post_ventral': df['fr_R_thick_G_cingul_Post_ventral_y3'], 'fr_R_thick_G_cuneus': df['fr_R_thick_G_cuneus_y3'], 'fr_R_thick_G_front_inf_Opercular': df['fr_R_thick_G_front_inf_Opercular_y3'], 'fr_R_thick_G_front_inf_Orbital': df['fr_R_thick_G_front_inf_Orbital_y3'], 'fr_R_thick_G_front_inf_Triangul': df['fr_R_thick_G_front_inf_Triangul_y3'], 'fr_R_thick_G_front_middle': df['fr_R_thick_G_front_middle_y3'], 'fr_R_thick_G_front_sup': df['fr_R_thick_G_front_sup_y3'], 'fr_R_thick_G_insular_short': df['fr_R_thick_G_insular_short_y3'], 'fr_R_thick_G_oc_temp_lat_fusifor': df['fr_R_thick_G_oc_temp_lat_fusifor_y3'], 'fr_R_thick_G_oc_temp_med_Lingual': df['fr_R_thick_G_oc_temp_med_Lingual_y3'], 'fr_R_thick_G_oc_temp_med_Parahip': df['fr_R_thick_G_oc_temp_med_Parahip_y3'], 'fr_R_thick_G_occipital_middle': df['fr_R_thick_G_occipital_middle_y3'], 'fr_R_thick_G_occipital_sup': df['fr_R_thick_G_occipital_sup_y3'], 'fr_R_thick_G_orbital': df['fr_R_thick_G_orbital_y3'], 'fr_R_thick_G_pariet_inf_Angular': df['fr_R_thick_G_pariet_inf_Angular_y3'], 'fr_R_thick_G_pariet_inf_Supramar': df['fr_R_thick_G_pariet_inf_Supramar_y3'], 'fr_R_thick_G_parietal_sup': df['fr_R_thick_G_parietal_sup_y3'], 'fr_R_thick_G_postcentral': df['fr_R_thick_G_postcentral_y3'], 'fr_R_thick_G_precentral': df['fr_R_thick_G_precentral_y3'], 'fr_R_thick_G_precuneus': df['fr_R_thick_G_precuneus_y3'], 'fr_R_thick_G_rectus': df['fr_R_thick_G_rectus_y3'], 'fr_R_thick_G_temp_sup_G_T_transv': df['fr_R_thick_G_temp_sup_G_T_transv_y3'], 'fr_R_thick_G_temp_sup_Lateral': df['fr_R_thick_G_temp_sup_Lateral_y3'], 'fr_R_thick_G_temp_sup_Plan_polar': df['fr_R_thick_G_temp_sup_Plan_polar_y3'], 'fr_R_thick_G_temp_sup_Plan_tempo': df['fr_R_thick_G_temp_sup_Plan_tempo_y3'], 'fr_R_thick_G_temporal_inf': df['fr_R_thick_G_temporal_inf_y3'], 'fr_R_thick_G_temporal_middle': df['fr_R_thick_G_temporal_middle_y3'], 'fr_R_thick_GandS_cingul_Ant': df['fr_R_thick_GandS_cingul_Ant_y3'], 'fr_R_thick_GandS_cingul_Mid_Post': df['fr_R_thick_GandS_cingul_Mid_Post_y3'], 'fr_R_thick_GandS_frontomargin': df['fr_R_thick_GandS_frontomargin_y3'], 'fr_R_thick_GandS_occipital_inf': df['fr_R_thick_GandS_occipital_inf_y3'], 'fr_R_thick_GandS_paracentral': df['fr_R_thick_GandS_paracentral_y3'], 'fr_R_thick_GandS_subcentral': df['fr_R_thick_GandS_subcentral_y3'], 'fr_R_thick_GandS_transv_frontopol': df['fr_R_thick_GandS_transv_frontopol_y3'], 'fr_R_thick_Lat_Fis_ant_Horizont': df['fr_R_thick_Lat_Fis_ant_Horizont_y3'], 'fr_R_thick_Lat_Fis_post': df['fr_R_thick_Lat_Fis_post_y3'], 'fr_R_thick_Pole_occipital': df['fr_R_thick_Pole_occipital_y3'], 'fr_R_thick_Pole_temporal': df['fr_R_thick_Pole_temporal_y3'], 'fr_R_thick_S_calcarine': df['fr_R_thick_S_calcarine_y3'], 'fr_R_thick_S_central': df['fr_R_thick_S_central_y3'], 'fr_R_thick_S_cingul_Marginalis': df['fr_R_thick_S_cingul_Marginalis_y3'], 'fr_R_thick_S_circular_insula_ant': df['fr_R_thick_S_circular_insula_ant_y3'], 'fr_R_thick_S_circular_insula_inf': df['fr_R_thick_S_circular_insula_inf_y3'], 'fr_R_thick_S_circular_insula_sup': df['fr_R_thick_S_circular_insula_sup_y3'], 'fr_R_thick_S_collat_transv_ant': df['fr_R_thick_S_collat_transv_ant_y3'], 'fr_R_thick_S_collat_transv_post': df['fr_R_thick_S_collat_transv_post_y3'], 'fr_R_thick_S_front_inf': df['fr_R_thick_S_front_inf_y3'], 'fr_R_thick_S_front_middle': df['fr_R_thick_S_front_middle_y3'], 'fr_R_thick_S_front_sup': df['fr_R_thick_S_front_sup_y3'], 'fr_R_thick_S_interm_prim_Jensen': df['fr_R_thick_S_interm_prim_Jensen_y3'], 'fr_R_thick_S_intraparietandP_trans': df['fr_R_thick_S_intraparietandP_trans_y3'], 'fr_R_thick_S_oc_middleandLunatus': df['fr_R_thick_S_oc_middleandLunatus_y3'], 'fr_R_thick_S_oc_temp_lat': df['fr_R_thick_S_oc_temp_lat_y3'], 'fr_R_thick_S_oc_temp_medandLingual': df['fr_R_thick_S_oc_temp_medandLingual_y3'], 'fr_R_thick_S_orbital_H_Shaped': df['fr_R_thick_S_orbital_H_Shaped_y3'], 'fr_R_thick_S_orbital_lateral': df['fr_R_thick_S_orbital_lateral_y3'], 'fr_R_thick_S_orbital_med_olfact': df['fr_R_thick_S_orbital_med_olfact_y3'], 'fr_R_thick_S_parieto_occipital': df['fr_R_thick_S_parieto_occipital_y3'], 'fr_R_thick_S_pericallosal': df['fr_R_thick_S_pericallosal_y3'], 'fr_R_thick_S_postcentral': df['fr_R_thick_S_postcentral_y3'], 'fr_R_thick_S_precentral_inf_part': df['fr_R_thick_S_precentral_inf_part_y3'], 'fr_R_thick_S_precentral_sup_part': df['fr_R_thick_S_precentral_sup_part_y3'], 'fr_R_thick_S_suborbital': df['fr_R_thick_S_suborbital_y3'], 'fr_R_thick_S_subparietal': df['fr_R_thick_S_subparietal_y3'], 'fr_R_thick_S_temporal_inf': df['fr_R_thick_S_temporal_inf_y3'], 'fr_R_thick_S_temporal_sup': df['fr_R_thick_S_temporal_sup_y3'], 'fr_R_thick_S_temporal_transverse': df['fr_R_thick_S_temporal_transverse_y3'], 'fr_L_thick_G_Ins_lgandS_cent_ins': df['fr_L_thick_G_Ins_lgandS_cent_ins_y3'], 'fr_L_thick_G_cingul_Post_ventral': df['fr_L_thick_G_cingul_Post_ventral_y3'], 'fr_L_thick_G_cuneus': df['fr_L_thick_G_cuneus_y3'], 'fr_L_thick_G_front_inf_Opercular': df['fr_L_thick_G_front_inf_Opercular_y3'], 'fr_L_thick_G_front_inf_Orbital': df['fr_L_thick_G_front_inf_Orbital_y3'], 'fr_L_thick_G_front_inf_Triangul': df['fr_L_thick_G_front_inf_Triangul_y3'], 'fr_L_thick_G_front_middle': df['fr_L_thick_G_front_middle_y3'], 'fr_L_thick_G_front_sup': df['fr_L_thick_G_front_sup_y3'], 'fr_L_thick_G_insular_short': df['fr_L_thick_G_insular_short_y3'], 'fr_L_thick_G_oc_temp_lat_fusifor': df['fr_L_thick_G_oc_temp_lat_fusifor_y3'], 'fr_L_thick_G_oc_temp_med_Lingual': df['fr_L_thick_G_oc_temp_med_Lingual_y3'], 'fr_L_thick_G_oc_temp_med_Parahip': df['fr_L_thick_G_oc_temp_med_Parahip_y3'], 'fr_L_thick_G_occipital_middle': df['fr_L_thick_G_occipital_middle_y3'], 'fr_L_thick_G_occipital_sup': df['fr_L_thick_G_occipital_sup_y3'], 'fr_L_thick_G_orbital': df['fr_L_thick_G_orbital_y3'], 'fr_L_thick_G_pariet_inf_Angular': df['fr_L_thick_G_pariet_inf_Angular_y3'], 'fr_L_thick_G_pariet_inf_Supramar': df['fr_L_thick_G_pariet_inf_Supramar_y3'], 'fr_L_thick_G_parietal_sup': df['fr_L_thick_G_parietal_sup_y3'], 'fr_L_thick_G_postcentral': df['fr_L_thick_G_postcentral_y3'], 'fr_L_thick_G_precentral': df['fr_L_thick_G_precentral_y3'], 'fr_L_thick_G_precuneus': df['fr_L_thick_G_precuneus_y3'], 'fr_L_thick_G_rectus': df['fr_L_thick_G_rectus_y3'], 'fr_L_thick_G_temp_sup_G_T_transv': df['fr_L_thick_G_temp_sup_G_T_transv_y3'], 'fr_L_thick_G_temp_sup_Lateral': df['fr_L_thick_G_temp_sup_Lateral_y3'], 'fr_L_thick_G_temp_sup_Plan_polar': df['fr_L_thick_G_temp_sup_Plan_polar_y3'], 'fr_L_thick_G_temp_sup_Plan_tempo': df['fr_L_thick_G_temp_sup_Plan_tempo_y3'], 'fr_L_thick_G_temporal_inf': df['fr_L_thick_G_temporal_inf_y3'], 'fr_L_thick_G_temporal_middle': df['fr_L_thick_G_temporal_middle_y3'], 'fr_L_thick_GandS_cingul_Ant': df['fr_L_thick_GandS_cingul_Ant_y3'], 'fr_L_thick_GandS_cingul_Mid_Post': df['fr_L_thick_GandS_cingul_Mid_Post_y3'], 'fr_L_thick_GandS_frontomargin': df['fr_L_thick_GandS_frontomargin_y3'], 'fr_L_thick_GandS_occipital_inf': df['fr_L_thick_GandS_occipital_inf_y3'], 'fr_L_thick_GandS_paracentral': df['fr_L_thick_GandS_paracentral_y3'], 'fr_L_thick_GandS_subcentral': df['fr_L_thick_GandS_subcentral_y3'], 'fr_L_thick_GandS_transv_frontopol': df['fr_L_thick_GandS_transv_frontopol_y3'], 'fr_L_thick_Lat_Fis_ant_Horizont': df['fr_L_thick_Lat_Fis_ant_Horizont_y3'], 'fr_L_thick_Lat_Fis_post': df['fr_L_thick_Lat_Fis_post_y3'], 'fr_L_thick_Pole_occipital': df['fr_L_thick_Pole_occipital_y3'], 'fr_L_thick_Pole_temporal': df['fr_L_thick_Pole_temporal_y3'], 'fr_L_thick_S_calcarine': df['fr_L_thick_S_calcarine_y3'], 'fr_L_thick_S_central': df['fr_L_thick_S_central_y3'], 'fr_L_thick_S_cingul_Marginalis': df['fr_L_thick_S_cingul_Marginalis_y3'], 'fr_L_thick_S_circular_insula_ant': df['fr_L_thick_S_circular_insula_ant_y3'], 'fr_L_thick_S_circular_insula_inf': df['fr_L_thick_S_circular_insula_inf_y3'], 'fr_L_thick_S_circular_insula_sup': df['fr_L_thick_S_circular_insula_sup_y3'], 'fr_L_thick_S_collat_transv_ant': df['fr_L_thick_S_collat_transv_ant_y3'], 'fr_L_thick_S_collat_transv_post': df['fr_L_thick_S_collat_transv_post_y3'], 'fr_L_thick_S_front_inf': df['fr_L_thick_S_front_inf_y3'], 'fr_L_thick_S_front_middle': df['fr_L_thick_S_front_middle_y3'], 'fr_L_thick_S_front_sup': df['fr_L_thick_S_front_sup_y3'],  'fr_L_thick_S_intraparietandP_trans': df['fr_L_thick_S_intraparietandP_trans_y3'], 'fr_L_thick_S_oc_middleandLunatus': df['fr_L_thick_S_oc_middleandLunatus_y3'], 'fr_L_thick_S_oc_temp_lat': df['fr_L_thick_S_oc_temp_lat_y3'], 'fr_L_thick_S_oc_temp_medandLingual': df['fr_L_thick_S_oc_temp_medandLingual_y3'], 'fr_L_thick_S_orbital_H_Shaped': df['fr_L_thick_S_orbital_H_Shaped_y3'], 'fr_L_thick_S_orbital_lateral': df['fr_L_thick_S_orbital_lateral_y3'], 'fr_L_thick_S_orbital_med_olfact': df['fr_L_thick_S_orbital_med_olfact_y3'], 'fr_L_thick_S_parieto_occipital': df['fr_L_thick_S_parieto_occipital_y3'], 'fr_L_thick_S_postcentral': df['fr_L_thick_S_postcentral_y3'], 'fr_L_thick_S_precentral_inf_part': df['fr_L_thick_S_precentral_inf_part_y3'], 'fr_L_thick_S_precentral_sup_part': df['fr_L_thick_S_precentral_sup_part_y3'], 'fr_L_thick_S_suborbital': df['fr_L_thick_S_suborbital_y3'], 'fr_L_thick_S_subparietal': df['fr_L_thick_S_subparietal_y3'], 'fr_L_thick_S_temporal_inf': df['fr_L_thick_S_temporal_inf_y3'], 'fr_L_thick_S_temporal_sup': df['fr_L_thick_S_temporal_sup_y3']}
		#frame_free4 = {'handlat': df['lat_manual'],'apoe': df['apoe'], 'y_last':df['ultimavisita'], 'dx_last': df['ultimodx'], 'dx_visita': df['dx_corto_visita4'], 'age_lastdx': df['edad_ultimodx'], 'age': df['edad_visita4'], 'sex': df['sexo'], 'free_R_Thal': df['fr_Right_Thalamus_Proper_y4'], 'free_L_Thal': df['fr_Left_Thalamus_Proper_y4'],'free_R_Puta': df['fr_Right_Putamen_y4'], 'free_L_Puta': df['fr_Left_Putamen_y4'],'free_R_Amyg': df['fr_Right_Amygdala_y4'], 'free_L_Amyg': df['fr_Left_Amygdala_y4'],'free_R_Pall': df['fr_Right_Pallidum_y4'], 'free_L_Pall': df['fr_Left_Pallidum_y4'],'free_R_Caud': df['fr_Right_Caudate_y4'], 'free_L_Caud': df['fr_Left_Caudate_y4'],'free_R_Hipp': df['fr_Right_Hippocampus_y4'], 'free_L_Hipp': df['fr_Left_Hippocampus_y4'],'free_R_Accu': df['fr_Right_Accumbens_area_y4'], 'free_L_Accu': df['fr_Left_Accumbens_area_y4'], 'fr_BrainSegVol': df['fr_BrainSegVol_y4'], 'fr_BrainSegVolNotVent':df['fr_BrainSegVolNotVent_y4'], 'fr_BrainSegVol_to_eTIV': df['fr_BrainSegVol_to_eTIV_y4']}
		frame_free4 = {'education': df['nivel_educativo'],'handlat': df['lat_manual'],'apoe': df['apoe'], 'nvisita':df['nvisita4'], 'y_last':df['ultimavisita'], 'dx_last': df['ultimodx'], 'dx_visita': df['dx_corto_visita4'], 'age_lastdx': df['edad_ultimodx'], 'age': df['edad_visita4'], 'sex': df['sexo'], 'free_R_Thal': df['fr_Right_Thalamus_Proper_y4'], 'free_L_Thal': df['fr_Left_Thalamus_Proper_y4'],'free_R_Puta': df['fr_Right_Putamen_y4'], 'free_L_Puta': df['fr_Left_Putamen_y4'],'free_R_Amyg': df['fr_Right_Amygdala_y4'], 'free_L_Amyg': df['fr_Left_Amygdala_y4'],'free_R_Pall': df['fr_Right_Pallidum_y4'], 'free_L_Pall': df['fr_Left_Pallidum_y4'],'free_R_Caud': df['fr_Right_Caudate_y4'], 'free_L_Caud': df['fr_Left_Caudate_y4'],'free_R_Hipp': df['fr_Right_Hippocampus_y4'], 'free_L_Hipp': df['fr_Left_Hippocampus_y4'],'free_R_Accu': df['fr_Right_Accumbens_area_y4'], 'free_L_Accu': df['fr_Left_Accumbens_area_y4'], 'fr_BrainSegVol': df['fr_BrainSegVol_y4'], 'fr_BrainSegVolNotVent':df['fr_BrainSegVolNotVent_y4'], 'fr_BrainSegVol_to_eTIV': df['fr_BrainSegVol_to_eTIV_y4'], 'fr_R_thick_G_Ins_lgandS_cent_ins': df['fr_R_thick_G_Ins_lgandS_cent_ins_y4'], 'fr_R_thick_G_cingul_Post_ventral': df['fr_R_thick_G_cingul_Post_ventral_y4'], 'fr_R_thick_G_cuneus': df['fr_R_thick_G_cuneus_y4'], 'fr_R_thick_G_front_inf_Opercular': df['fr_R_thick_G_front_inf_Opercular_y4'], 'fr_R_thick_G_front_inf_Orbital': df['fr_R_thick_G_front_inf_Orbital_y4'], 'fr_R_thick_G_front_inf_Triangul': df['fr_R_thick_G_front_inf_Triangul_y4'], 'fr_R_thick_G_front_middle': df['fr_R_thick_G_front_middle_y4'], 'fr_R_thick_G_front_sup': df['fr_R_thick_G_front_sup_y4'], 'fr_R_thick_G_insular_short': df['fr_R_thick_G_insular_short_y4'], 'fr_R_thick_G_oc_temp_lat_fusifor': df['fr_R_thick_G_oc_temp_lat_fusifor_y4'], 'fr_R_thick_G_oc_temp_med_Lingual': df['fr_R_thick_G_oc_temp_med_Lingual_y4'], 'fr_R_thick_G_oc_temp_med_Parahip': df['fr_R_thick_G_oc_temp_med_Parahip_y4'], 'fr_R_thick_G_occipital_middle': df['fr_R_thick_G_occipital_middle_y4'], 'fr_R_thick_G_occipital_sup': df['fr_R_thick_G_occipital_sup_y4'], 'fr_R_thick_G_orbital': df['fr_R_thick_G_orbital_y4'], 'fr_R_thick_G_pariet_inf_Angular': df['fr_R_thick_G_pariet_inf_Angular_y4'], 'fr_R_thick_G_pariet_inf_Supramar': df['fr_R_thick_G_pariet_inf_Supramar_y4'], 'fr_R_thick_G_parietal_sup': df['fr_R_thick_G_parietal_sup_y4'], 'fr_R_thick_G_postcentral': df['fr_R_thick_G_postcentral_y4'], 'fr_R_thick_G_precentral': df['fr_R_thick_G_precentral_y4'], 'fr_R_thick_G_precuneus': df['fr_R_thick_G_precuneus_y4'], 'fr_R_thick_G_rectus': df['fr_R_thick_G_rectus_y4'], 'fr_R_thick_G_temp_sup_G_T_transv': df['fr_R_thick_G_temp_sup_G_T_transv_y4'], 'fr_R_thick_G_temp_sup_Lateral': df['fr_R_thick_G_temp_sup_Lateral_y4'], 'fr_R_thick_G_temp_sup_Plan_polar': df['fr_R_thick_G_temp_sup_Plan_polar_y4'], 'fr_R_thick_G_temp_sup_Plan_tempo': df['fr_R_thick_G_temp_sup_Plan_tempo_y4'], 'fr_R_thick_G_temporal_inf': df['fr_R_thick_G_temporal_inf_y4'], 'fr_R_thick_G_temporal_middle': df['fr_R_thick_G_temporal_middle_y4'], 'fr_R_thick_GandS_cingul_Ant': df['fr_R_thick_GandS_cingul_Ant_y4'], 'fr_R_thick_GandS_cingul_Mid_Post': df['fr_R_thick_GandS_cingul_Mid_Post_y4'], 'fr_R_thick_GandS_frontomargin': df['fr_R_thick_GandS_frontomargin_y4'], 'fr_R_thick_GandS_occipital_inf': df['fr_R_thick_GandS_occipital_inf_y4'], 'fr_R_thick_GandS_paracentral': df['fr_R_thick_GandS_paracentral_y4'], 'fr_R_thick_GandS_subcentral': df['fr_R_thick_GandS_subcentral_y4'], 'fr_R_thick_GandS_transv_frontopol': df['fr_R_thick_GandS_transv_frontopol_y4'], 'fr_R_thick_Lat_Fis_ant_Horizont': df['fr_R_thick_Lat_Fis_ant_Horizont_y4'], 'fr_R_thick_Lat_Fis_post': df['fr_R_thick_Lat_Fis_post_y4'], 'fr_R_thick_Pole_occipital': df['fr_R_thick_Pole_occipital_y4'], 'fr_R_thick_Pole_temporal': df['fr_R_thick_Pole_temporal_y4'], 'fr_R_thick_S_calcarine': df['fr_R_thick_S_calcarine_y4'], 'fr_R_thick_S_central': df['fr_R_thick_S_central_y4'], 'fr_R_thick_S_cingul_Marginalis': df['fr_R_thick_S_cingul_Marginalis_y4'], 'fr_R_thick_S_circular_insula_ant': df['fr_R_thick_S_circular_insula_ant_y4'], 'fr_R_thick_S_circular_insula_inf': df['fr_R_thick_S_circular_insula_inf_y4'], 'fr_R_thick_S_circular_insula_sup': df['fr_R_thick_S_circular_insula_sup_y4'], 'fr_R_thick_S_collat_transv_ant': df['fr_R_thick_S_collat_transv_ant_y4'], 'fr_R_thick_S_collat_transv_post': df['fr_R_thick_S_collat_transv_post_y4'], 'fr_R_thick_S_front_inf': df['fr_R_thick_S_front_inf_y4'], 'fr_R_thick_S_front_middle': df['fr_R_thick_S_front_middle_y4'], 'fr_R_thick_S_front_sup': df['fr_R_thick_S_front_sup_y4'], 'fr_R_thick_S_interm_prim_Jensen': df['fr_R_thick_S_interm_prim_Jensen_y4'], 'fr_R_thick_S_intraparietandP_trans': df['fr_R_thick_S_intraparietandP_trans_y4'], 'fr_R_thick_S_oc_middleandLunatus': df['fr_R_thick_S_oc_middleandLunatus_y4'], 'fr_R_thick_S_oc_temp_lat': df['fr_R_thick_S_oc_temp_lat_y4'], 'fr_R_thick_S_oc_temp_medandLingual': df['fr_R_thick_S_oc_temp_medandLingual_y4'], 'fr_R_thick_S_orbital_H_Shaped': df['fr_R_thick_S_orbital_H_Shaped_y4'], 'fr_R_thick_S_orbital_lateral': df['fr_R_thick_S_orbital_lateral_y4'], 'fr_R_thick_S_orbital_med_olfact': df['fr_R_thick_S_orbital_med_olfact_y4'], 'fr_R_thick_S_parieto_occipital': df['fr_R_thick_S_parieto_occipital_y4'], 'fr_R_thick_S_pericallosal': df['fr_R_thick_S_pericallosal_y4'], 'fr_R_thick_S_postcentral': df['fr_R_thick_S_postcentral_y4'], 'fr_R_thick_S_precentral_inf_part': df['fr_R_thick_S_precentral_inf_part_y4'], 'fr_R_thick_S_precentral_sup_part': df['fr_R_thick_S_precentral_sup_part_y4'], 'fr_R_thick_S_suborbital': df['fr_R_thick_S_suborbital_y4'], 'fr_R_thick_S_subparietal': df['fr_R_thick_S_subparietal_y4'], 'fr_R_thick_S_temporal_inf': df['fr_R_thick_S_temporal_inf_y4'], 'fr_R_thick_S_temporal_sup': df['fr_R_thick_S_temporal_sup_y4'], 'fr_R_thick_S_temporal_transverse': df['fr_R_thick_S_temporal_transverse_y4'],   'fr_L_thick_G_Ins_lgandS_cent_ins': df['fr_L_thick_G_Ins_lgandS_cent_ins_y4'], 'fr_L_thick_G_cingul_Post_ventral': df['fr_L_thick_G_cingul_Post_ventral_y4'], 'fr_L_thick_G_cuneus': df['fr_L_thick_G_cuneus_y4'], 'fr_L_thick_G_front_inf_Opercular': df['fr_L_thick_G_front_inf_Opercular_y4'], 'fr_L_thick_G_front_inf_Orbital': df['fr_L_thick_G_front_inf_Orbital_y4'], 'fr_L_thick_G_front_inf_Triangul': df['fr_L_thick_G_front_inf_Triangul_y4'], 'fr_L_thick_G_front_middle': df['fr_L_thick_G_front_middle_y4'], 'fr_L_thick_G_front_sup': df['fr_L_thick_G_front_sup_y4'], 'fr_L_thick_G_insular_short': df['fr_L_thick_G_insular_short_y4'], 'fr_L_thick_G_oc_temp_lat_fusifor': df['fr_L_thick_G_oc_temp_lat_fusifor_y4'], 'fr_L_thick_G_oc_temp_med_Lingual': df['fr_L_thick_G_oc_temp_med_Lingual_y4'], 'fr_L_thick_G_oc_temp_med_Parahip': df['fr_L_thick_G_oc_temp_med_Parahip_y4'], 'fr_L_thick_G_occipital_middle': df['fr_L_thick_G_occipital_middle_y4'], 'fr_L_thick_G_occipital_sup': df['fr_L_thick_G_occipital_sup_y4'], 'fr_L_thick_G_orbital': df['fr_L_thick_G_orbital_y4'], 'fr_L_thick_G_pariet_inf_Angular': df['fr_L_thick_G_pariet_inf_Angular_y4'], 'fr_L_thick_G_pariet_inf_Supramar': df['fr_L_thick_G_pariet_inf_Supramar_y4'], 'fr_L_thick_G_parietal_sup': df['fr_L_thick_G_parietal_sup_y4'], 'fr_L_thick_G_postcentral': df['fr_L_thick_G_postcentral_y4'], 'fr_L_thick_G_precentral': df['fr_L_thick_G_precentral_y4'], 'fr_L_thick_G_precuneus': df['fr_L_thick_G_precuneus_y4'], 'fr_L_thick_G_rectus': df['fr_L_thick_G_rectus_y4'], 'fr_L_thick_G_temp_sup_G_T_transv': df['fr_L_thick_G_temp_sup_G_T_transv_y4'], 'fr_L_thick_G_temp_sup_Lateral': df['fr_L_thick_G_temp_sup_Lateral_y4'], 'fr_L_thick_G_temp_sup_Plan_polar': df['fr_L_thick_G_temp_sup_Plan_polar_y4'], 'fr_L_thick_G_temp_sup_Plan_tempo': df['fr_L_thick_G_temp_sup_Plan_tempo_y4'], 'fr_L_thick_G_temporal_inf': df['fr_L_thick_G_temporal_inf_y4'], 'fr_L_thick_G_temporal_middle': df['fr_L_thick_G_temporal_middle_y4'], 'fr_L_thick_GandS_cingul_Ant': df['fr_L_thick_GandS_cingul_Ant_y4'], 'fr_L_thick_GandS_cingul_Mid_Post': df['fr_L_thick_GandS_cingul_Mid_Post_y4'], 'fr_L_thick_GandS_frontomargin': df['fr_L_thick_GandS_frontomargin_y4'], 'fr_L_thick_GandS_occipital_inf': df['fr_L_thick_GandS_occipital_inf_y4'], 'fr_L_thick_GandS_paracentral': df['fr_L_thick_GandS_paracentral_y4'], 'fr_L_thick_GandS_subcentral': df['fr_L_thick_GandS_subcentral_y4'], 'fr_L_thick_GandS_transv_frontopol': df['fr_L_thick_GandS_transv_frontopol_y4'], 'fr_L_thick_Lat_Fis_ant_Horizont': df['fr_L_thick_Lat_Fis_ant_Horizont_y4'], 'fr_L_thick_Lat_Fis_post': df['fr_L_thick_Lat_Fis_post_y4'], 'fr_L_thick_Pole_occipital': df['fr_L_thick_Pole_occipital_y4'], 'fr_L_thick_Pole_temporal': df['fr_L_thick_Pole_temporal_y4'], 'fr_L_thick_S_calcarine': df['fr_L_thick_S_calcarine_y4'], 'fr_L_thick_S_central': df['fr_L_thick_S_central_y4'], 'fr_L_thick_S_cingul_Marginalis': df['fr_L_thick_S_cingul_Marginalis_y4'], 'fr_L_thick_S_circular_insula_ant': df['fr_L_thick_S_circular_insula_ant_y4'], 'fr_L_thick_S_circular_insula_inf': df['fr_L_thick_S_circular_insula_inf_y4'], 'fr_L_thick_S_circular_insula_sup': df['fr_L_thick_S_circular_insula_sup_y4'], 'fr_L_thick_S_collat_transv_ant': df['fr_L_thick_S_collat_transv_ant_y4'], 'fr_L_thick_S_collat_transv_post': df['fr_L_thick_S_collat_transv_post_y4'], 'fr_L_thick_S_front_inf': df['fr_L_thick_S_front_inf_y4'], 'fr_L_thick_S_front_middle': df['fr_L_thick_S_front_middle_y4'], 'fr_L_thick_S_front_sup': df['fr_L_thick_S_front_sup_y4'], 'fr_L_thick_S_intraparietandP_trans': df['fr_L_thick_S_intraparietandP_trans_y4'], 'fr_L_thick_S_oc_middleandLunatus': df['fr_L_thick_S_oc_middleandLunatus_y4'], 'fr_L_thick_S_oc_temp_lat': df['fr_L_thick_S_oc_temp_lat_y4'], 'fr_L_thick_S_oc_temp_medandLingual': df['fr_L_thick_S_oc_temp_medandLingual_y4'], 'fr_L_thick_S_orbital_H_Shaped': df['fr_L_thick_S_orbital_H_Shaped_y4'], 'fr_L_thick_S_orbital_lateral': df['fr_L_thick_S_orbital_lateral_y4'], 'fr_L_thick_S_orbital_med_olfact': df['fr_L_thick_S_orbital_med_olfact_y4'], 'fr_L_thick_S_parieto_occipital': df['fr_L_thick_S_parieto_occipital_y4'], 'fr_L_thick_S_postcentral': df['fr_L_thick_S_postcentral_y4'], 'fr_L_thick_S_precentral_inf_part': df['fr_L_thick_S_precentral_inf_part_y4'], 'fr_L_thick_S_precentral_sup_part': df['fr_L_thick_S_precentral_sup_part_y4'], 'fr_L_thick_S_suborbital': df['fr_L_thick_S_suborbital_y4'], 'fr_L_thick_S_subparietal': df['fr_L_thick_S_subparietal_y4'], 'fr_L_thick_S_temporal_inf': df['fr_L_thick_S_temporal_inf_y4'], 'fr_L_thick_S_temporal_sup': df['fr_L_thick_S_temporal_sup_y4']}
		#frame_free5 = {'handlat': df['lat_manual'],'apoe': df['apoe'], 'y_last':df['ultimavisita'], 'dx_last': df['ultimodx'], 'dx_visita': df['dx_corto_visita5'], 'age_lastdx': df['edad_ultimodx'], 'age': df['edad_visita5'], 'sex': df['sexo'], 'free_R_Thal': df['fr_Right_Thalamus_Proper_y5'], 'free_L_Thal': df['fr_Left_Thalamus_Proper_y5'],'free_R_Puta': df['fr_Right_Putamen_y5'], 'free_L_Puta': df['fr_Left_Putamen_y5'],'free_R_Amyg': df['fr_Right_Amygdala_y5'], 'free_L_Amyg': df['fr_Left_Amygdala_y5'],'free_R_Pall': df['fr_Right_Pallidum_y5'], 'free_L_Pall': df['fr_Left_Pallidum_y5'],'free_R_Caud': df['fr_Right_Caudate_y5'], 'free_L_Caud': df['fr_Left_Caudate_y5'],'free_R_Hipp': df['fr_Right_Hippocampus_y5'], 'free_L_Hipp': df['fr_Left_Hippocampus_y5'],'free_R_Accu': df['fr_Right_Accumbens_area_y5'], 'free_L_Accu': df['fr_Left_Accumbens_area_y5'], 'fr_BrainSegVol': df['fr_BrainSegVol_y5'], 'fr_BrainSegVolNotVent':df['fr_BrainSegVolNotVent_y5'], 'fr_BrainSegVol_to_eTIV': df['fr_BrainSegVol_to_eTIV_y5']}
		frame_free5 = {'education': df['nivel_educativo'],'handlat': df['lat_manual'],'apoe': df['apoe'], 'nvisita':df['nvisita5'] ,'y_last':df['ultimavisita'], 'dx_last': df['ultimodx'], 'dx_visita': df['dx_corto_visita5'], 'age_lastdx': df['edad_ultimodx'], 'age': df['edad_visita5'], 'sex': df['sexo'], 'free_R_Thal': df['fr_Right_Thalamus_Proper_y5'], 'free_L_Thal': df['fr_Left_Thalamus_Proper_y5'],'free_R_Puta': df['fr_Right_Putamen_y5'], 'free_L_Puta': df['fr_Left_Putamen_y5'],'free_R_Amyg': df['fr_Right_Amygdala_y5'], 'free_L_Amyg': df['fr_Left_Amygdala_y5'],'free_R_Pall': df['fr_Right_Pallidum_y5'], 'free_L_Pall': df['fr_Left_Pallidum_y5'],'free_R_Caud': df['fr_Right_Caudate_y5'], 'free_L_Caud': df['fr_Left_Caudate_y5'],'free_R_Hipp': df['fr_Right_Hippocampus_y5'], 'free_L_Hipp': df['fr_Left_Hippocampus_y5'],'free_R_Accu': df['fr_Right_Accumbens_area_y5'], 'free_L_Accu': df['fr_Left_Accumbens_area_y5'], 'fr_BrainSegVol': df['fr_BrainSegVol_y5'], 'fr_BrainSegVolNotVent':df['fr_BrainSegVolNotVent_y5'], 'fr_BrainSegVol_to_eTIV': df['fr_BrainSegVol_to_eTIV_y5'], 'fr_R_thick_G_Ins_lgandS_cent_ins': df['fr_R_thick_G_Ins_lgandS_cent_ins_y5'], 'fr_R_thick_G_cingul_Post_ventral': df['fr_R_thick_G_cingul_Post_ventral_y5'], 'fr_R_thick_G_cuneus': df['fr_R_thick_G_cuneus_y5'], 'fr_R_thick_G_front_inf_Opercular': df['fr_R_thick_G_front_inf_Opercular_y5'], 'fr_R_thick_G_front_inf_Orbital': df['fr_R_thick_G_front_inf_Orbital_y5'], 'fr_R_thick_G_front_inf_Triangul': df['fr_R_thick_G_front_inf_Triangul_y5'], 'fr_R_thick_G_front_middle': df['fr_R_thick_G_front_middle_y5'], 'fr_R_thick_G_front_sup': df['fr_R_thick_G_front_sup_y5'], 'fr_R_thick_G_insular_short': df['fr_R_thick_G_insular_short_y5'], 'fr_R_thick_G_oc_temp_lat_fusifor': df['fr_R_thick_G_oc_temp_lat_fusifor_y5'], 'fr_R_thick_G_oc_temp_med_Lingual': df['fr_R_thick_G_oc_temp_med_Lingual_y5'], 'fr_R_thick_G_oc_temp_med_Parahip': df['fr_R_thick_G_oc_temp_med_Parahip_y5'], 'fr_R_thick_G_occipital_middle': df['fr_R_thick_G_occipital_middle_y5'], 'fr_R_thick_G_occipital_sup': df['fr_R_thick_G_occipital_sup_y5'], 'fr_R_thick_G_orbital': df['fr_R_thick_G_orbital_y5'], 'fr_R_thick_G_pariet_inf_Angular': df['fr_R_thick_G_pariet_inf_Angular_y5'], 'fr_R_thick_G_pariet_inf_Supramar': df['fr_R_thick_G_pariet_inf_Supramar_y5'], 'fr_R_thick_G_parietal_sup': df['fr_R_thick_G_parietal_sup_y5'], 'fr_R_thick_G_postcentral': df['fr_R_thick_G_postcentral_y5'], 'fr_R_thick_G_precentral': df['fr_R_thick_G_precentral_y5'], 'fr_R_thick_G_precuneus': df['fr_R_thick_G_precuneus_y5'], 'fr_R_thick_G_rectus': df['fr_R_thick_G_rectus_y5'], 'fr_R_thick_G_temp_sup_G_T_transv': df['fr_R_thick_G_temp_sup_G_T_transv_y5'], 'fr_R_thick_G_temp_sup_Lateral': df['fr_R_thick_G_temp_sup_Lateral_y5'], 'fr_R_thick_G_temp_sup_Plan_polar': df['fr_R_thick_G_temp_sup_Plan_polar_y5'], 'fr_R_thick_G_temp_sup_Plan_tempo': df['fr_R_thick_G_temp_sup_Plan_tempo_y5'], 'fr_R_thick_G_temporal_inf': df['fr_R_thick_G_temporal_inf_y5'], 'fr_R_thick_G_temporal_middle': df['fr_R_thick_G_temporal_middle_y5'], 'fr_R_thick_GandS_cingul_Ant': df['fr_R_thick_GandS_cingul_Ant_y5'], 'fr_R_thick_GandS_cingul_Mid_Post': df['fr_R_thick_GandS_cingul_Mid_Post_y5'], 'fr_R_thick_GandS_frontomargin': df['fr_R_thick_GandS_frontomargin_y5'], 'fr_R_thick_GandS_occipital_inf': df['fr_R_thick_GandS_occipital_inf_y5'], 'fr_R_thick_GandS_paracentral': df['fr_R_thick_GandS_paracentral_y5'], 'fr_R_thick_GandS_subcentral': df['fr_R_thick_GandS_subcentral_y5'], 'fr_R_thick_GandS_transv_frontopol': df['fr_R_thick_GandS_transv_frontopol_y5'], 'fr_R_thick_Lat_Fis_ant_Horizont': df['fr_R_thick_Lat_Fis_ant_Horizont_y5'], 'fr_R_thick_Lat_Fis_post': df['fr_R_thick_Lat_Fis_post_y5'], 'fr_R_thick_Pole_occipital': df['fr_R_thick_Pole_occipital_y5'], 'fr_R_thick_Pole_temporal': df['fr_R_thick_Pole_temporal_y5'], 'fr_R_thick_S_calcarine': df['fr_R_thick_S_calcarine_y5'], 'fr_R_thick_S_central': df['fr_R_thick_S_central_y5'], 'fr_R_thick_S_cingul_Marginalis': df['fr_R_thick_S_cingul_Marginalis_y5'], 'fr_R_thick_S_circular_insula_ant': df['fr_R_thick_S_circular_insula_ant_y5'], 'fr_R_thick_S_circular_insula_inf': df['fr_R_thick_S_circular_insula_inf_y5'], 'fr_R_thick_S_circular_insula_sup': df['fr_R_thick_S_circular_insula_sup_y5'], 'fr_R_thick_S_collat_transv_ant': df['fr_R_thick_S_collat_transv_ant_y5'], 'fr_R_thick_S_collat_transv_post': df['fr_R_thick_S_collat_transv_post_y5'], 'fr_R_thick_S_front_inf': df['fr_R_thick_S_front_inf_y5'], 'fr_R_thick_S_front_middle': df['fr_R_thick_S_front_middle_y5'], 'fr_R_thick_S_front_sup': df['fr_R_thick_S_front_sup_y5'], 'fr_R_thick_S_interm_prim_Jensen': df['fr_R_thick_S_interm_prim_Jensen_y5'], 'fr_R_thick_S_intraparietandP_trans': df['fr_R_thick_S_intraparietandP_trans_y5'], 'fr_R_thick_S_oc_middleandLunatus': df['fr_R_thick_S_oc_middleandLunatus_y5'], 'fr_R_thick_S_oc_temp_lat': df['fr_R_thick_S_oc_temp_lat_y5'], 'fr_R_thick_S_oc_temp_medandLingual': df['fr_R_thick_S_oc_temp_medandLingual_y5'], 'fr_R_thick_S_orbital_H_Shaped': df['fr_R_thick_S_orbital_H_Shaped_y5'], 'fr_R_thick_S_orbital_lateral': df['fr_R_thick_S_orbital_lateral_y5'], 'fr_R_thick_S_orbital_med_olfact': df['fr_R_thick_S_orbital_med_olfact_y5'], 'fr_R_thick_S_parieto_occipital': df['fr_R_thick_S_parieto_occipital_y5'], 'fr_R_thick_S_pericallosal': df['fr_R_thick_S_pericallosal_y5'], 'fr_R_thick_S_postcentral': df['fr_R_thick_S_postcentral_y5'], 'fr_R_thick_S_precentral_inf_part': df['fr_R_thick_S_precentral_inf_part_y5'], 'fr_R_thick_S_precentral_sup_part': df['fr_R_thick_S_precentral_sup_part_y5'], 'fr_R_thick_S_suborbital': df['fr_R_thick_S_suborbital_y5'], 'fr_R_thick_S_subparietal': df['fr_R_thick_S_subparietal_y5'], 'fr_R_thick_S_temporal_inf': df['fr_R_thick_S_temporal_inf_y5'], 'fr_R_thick_S_temporal_sup': df['fr_R_thick_S_temporal_sup_y5'], 'fr_R_thick_S_temporal_transverse': df['fr_R_thick_S_temporal_transverse_y5'],    'fr_L_thick_G_Ins_lgandS_cent_ins': df['fr_L_thick_G_Ins_lgandS_cent_ins_y5'], 'fr_L_thick_G_cingul_Post_ventral': df['fr_L_thick_G_cingul_Post_ventral_y5'], 'fr_L_thick_G_cuneus': df['fr_L_thick_G_cuneus_y5'], 'fr_L_thick_G_front_inf_Opercular': df['fr_L_thick_G_front_inf_Opercular_y5'], 'fr_L_thick_G_front_inf_Orbital': df['fr_L_thick_G_front_inf_Orbital_y5'], 'fr_L_thick_G_front_inf_Triangul': df['fr_L_thick_G_front_inf_Triangul_y5'], 'fr_L_thick_G_front_middle': df['fr_L_thick_G_front_middle_y5'], 'fr_L_thick_G_front_sup': df['fr_L_thick_G_front_sup_y5'], 'fr_L_thick_G_insular_short': df['fr_L_thick_G_insular_short_y5'], 'fr_L_thick_G_oc_temp_lat_fusifor': df['fr_L_thick_G_oc_temp_lat_fusifor_y5'], 'fr_L_thick_G_oc_temp_med_Lingual': df['fr_L_thick_G_oc_temp_med_Lingual_y5'], 'fr_L_thick_G_oc_temp_med_Parahip': df['fr_L_thick_G_oc_temp_med_Parahip_y5'], 'fr_L_thick_G_occipital_middle': df['fr_L_thick_G_occipital_middle_y5'], 'fr_L_thick_G_occipital_sup': df['fr_L_thick_G_occipital_sup_y5'], 'fr_L_thick_G_orbital': df['fr_L_thick_G_orbital_y5'], 'fr_L_thick_G_pariet_inf_Angular': df['fr_L_thick_G_pariet_inf_Angular_y5'], 'fr_L_thick_G_pariet_inf_Supramar': df['fr_L_thick_G_pariet_inf_Supramar_y5'], 'fr_L_thick_G_parietal_sup': df['fr_L_thick_G_parietal_sup_y5'], 'fr_L_thick_G_postcentral': df['fr_L_thick_G_postcentral_y5'], 'fr_L_thick_G_precentral': df['fr_L_thick_G_precentral_y5'], 'fr_L_thick_G_precuneus': df['fr_L_thick_G_precuneus_y5'], 'fr_L_thick_G_rectus': df['fr_L_thick_G_rectus_y5'], 'fr_L_thick_G_temp_sup_G_T_transv': df['fr_L_thick_G_temp_sup_G_T_transv_y5'], 'fr_L_thick_G_temp_sup_Lateral': df['fr_L_thick_G_temp_sup_Lateral_y5'], 'fr_L_thick_G_temp_sup_Plan_polar': df['fr_L_thick_G_temp_sup_Plan_polar_y5'], 'fr_L_thick_G_temp_sup_Plan_tempo': df['fr_L_thick_G_temp_sup_Plan_tempo_y5'], 'fr_L_thick_G_temporal_inf': df['fr_L_thick_G_temporal_inf_y5'], 'fr_L_thick_G_temporal_middle': df['fr_L_thick_G_temporal_middle_y5'], 'fr_L_thick_GandS_cingul_Ant': df['fr_L_thick_GandS_cingul_Ant_y5'], 'fr_L_thick_GandS_cingul_Mid_Post': df['fr_L_thick_GandS_cingul_Mid_Post_y5'], 'fr_L_thick_GandS_frontomargin': df['fr_L_thick_GandS_frontomargin_y5'], 'fr_L_thick_GandS_occipital_inf': df['fr_L_thick_GandS_occipital_inf_y5'], 'fr_L_thick_GandS_paracentral': df['fr_L_thick_GandS_paracentral_y5'], 'fr_L_thick_GandS_subcentral': df['fr_L_thick_GandS_subcentral_y5'], 'fr_L_thick_GandS_transv_frontopol': df['fr_L_thick_GandS_transv_frontopol_y5'], 'fr_L_thick_Lat_Fis_ant_Horizont': df['fr_L_thick_Lat_Fis_ant_Horizont_y5'], 'fr_L_thick_Lat_Fis_post': df['fr_L_thick_Lat_Fis_post_y5'], 'fr_L_thick_Pole_occipital': df['fr_L_thick_Pole_occipital_y5'], 'fr_L_thick_Pole_temporal': df['fr_L_thick_Pole_temporal_y5'], 'fr_L_thick_S_calcarine': df['fr_L_thick_S_calcarine_y5'], 'fr_L_thick_S_central': df['fr_L_thick_S_central_y5'], 'fr_L_thick_S_cingul_Marginalis': df['fr_L_thick_S_cingul_Marginalis_y5'], 'fr_L_thick_S_circular_insula_ant': df['fr_L_thick_S_circular_insula_ant_y5'], 'fr_L_thick_S_circular_insula_inf': df['fr_L_thick_S_circular_insula_inf_y5'], 'fr_L_thick_S_circular_insula_sup': df['fr_L_thick_S_circular_insula_sup_y5'], 'fr_L_thick_S_collat_transv_ant': df['fr_L_thick_S_collat_transv_ant_y5'], 'fr_L_thick_S_collat_transv_post': df['fr_L_thick_S_collat_transv_post_y5'], 'fr_L_thick_S_front_inf': df['fr_L_thick_S_front_inf_y5'], 'fr_L_thick_S_front_middle': df['fr_L_thick_S_front_middle_y5'], 'fr_L_thick_S_front_sup': df['fr_L_thick_S_front_sup_y5'], 'fr_L_thick_S_intraparietandP_trans': df['fr_L_thick_S_intraparietandP_trans_y5'], 'fr_L_thick_S_oc_middleandLunatus': df['fr_L_thick_S_oc_middleandLunatus_y5'], 'fr_L_thick_S_oc_temp_lat': df['fr_L_thick_S_oc_temp_lat_y5'], 'fr_L_thick_S_oc_temp_medandLingual': df['fr_L_thick_S_oc_temp_medandLingual_y5'], 'fr_L_thick_S_orbital_H_Shaped': df['fr_L_thick_S_orbital_H_Shaped_y5'], 'fr_L_thick_S_orbital_lateral': df['fr_L_thick_S_orbital_lateral_y5'], 'fr_L_thick_S_orbital_med_olfact': df['fr_L_thick_S_orbital_med_olfact_y5'], 'fr_L_thick_S_parieto_occipital': df['fr_L_thick_S_parieto_occipital_y5'], 'fr_L_thick_S_postcentral': df['fr_L_thick_S_postcentral_y5'], 'fr_L_thick_S_precentral_inf_part': df['fr_L_thick_S_precentral_inf_part_y5'], 'fr_L_thick_S_precentral_sup_part': df['fr_L_thick_S_precentral_sup_part_y5'], 'fr_L_thick_S_suborbital': df['fr_L_thick_S_suborbital_y5'], 'fr_L_thick_S_subparietal': df['fr_L_thick_S_subparietal_y5'], 'fr_L_thick_S_temporal_inf': df['fr_L_thick_S_temporal_inf_y5'], 'fr_L_thick_S_temporal_sup': df['fr_L_thick_S_temporal_sup_y5']}		
		#frame_free6 = {'handlat': df['lat_manual'],'apoe': df['apoe'], 'y_last':df['ultimavisita'], 'dx_last': df['ultimodx'], 'dx_visita': df['dx_corto_visita6'], 'age_lastdx': df['edad_ultimodx'], 'age': df['edad_visita6'], 'sex': df['sexo'], 'free_R_Thal': df['fr_Right_Thalamus_Proper_y6'], 'free_L_Thal': df['fr_Left_Thalamus_Proper_y6'],'free_R_Puta': df['fr_Right_Putamen_y6'], 'free_L_Puta': df['fr_Left_Putamen_y6'],'free_R_Amyg': df['fr_Right_Amygdala_y6'], 'free_L_Amyg': df['fr_Left_Amygdala_y6'],'free_R_Pall': df['fr_Right_Pallidum_y6'], 'free_L_Pall': df['fr_Left_Pallidum_y6'],'free_R_Caud': df['fr_Right_Caudate_y6'], 'free_L_Caud': df['fr_Left_Caudate_y6'],'free_R_Hipp': df['fr_Right_Hippocampus_y6'], 'free_L_Hipp': df['fr_Left_Hippocampus_y6'],'free_R_Accu': df['fr_Right_Accumbens_area_y6'], 'free_L_Accu': df['fr_Left_Accumbens_area_y6'], 'fr_BrainSegVol': df['fr_BrainSegVol_y6'], 'fr_BrainSegVolNotVent':df['fr_BrainSegVolNotVent_y6'], 'fr_BrainSegVol_to_eTIV': df['fr_BrainSegVol_to_eTIV_y6']}
		frame_free6 = {'education': df['nivel_educativo'],'handlat': df['lat_manual'],'apoe': df['apoe'], 'nvisita':df['nvisita6'] , 'y_last':df['ultimavisita'], 'dx_last': df['ultimodx'], 'dx_visita': df['dx_corto_visita6'], 'age_lastdx': df['edad_ultimodx'], 'age': df['edad_visita6'], 'sex': df['sexo'], 'free_R_Thal': df['fr_Right_Thalamus_Proper_y6'], 'free_L_Thal': df['fr_Left_Thalamus_Proper_y6'],'free_R_Puta': df['fr_Right_Putamen_y6'], 'free_L_Puta': df['fr_Left_Putamen_y6'],'free_R_Amyg': df['fr_Right_Amygdala_y6'], 'free_L_Amyg': df['fr_Left_Amygdala_y6'],'free_R_Pall': df['fr_Right_Pallidum_y6'], 'free_L_Pall': df['fr_Left_Pallidum_y6'],'free_R_Caud': df['fr_Right_Caudate_y6'], 'free_L_Caud': df['fr_Left_Caudate_y6'],'free_R_Hipp': df['fr_Right_Hippocampus_y6'], 'free_L_Hipp': df['fr_Left_Hippocampus_y6'],'free_R_Accu': df['fr_Right_Accumbens_area_y6'], 'free_L_Accu': df['fr_Left_Accumbens_area_y6'], 'fr_BrainSegVol': df['fr_BrainSegVol_y6'], 'fr_BrainSegVolNotVent':df['fr_BrainSegVolNotVent_y6'], 'fr_BrainSegVol_to_eTIV': df['fr_BrainSegVol_to_eTIV_y6'], 'fr_R_thick_G_Ins_lgandS_cent_ins': df['fr_R_thick_G_Ins_lgandS_cent_ins_y6'], 'fr_R_thick_G_cingul_Post_ventral': df['fr_R_thick_G_cingul_Post_ventral_y6'], 'fr_R_thick_G_cuneus': df['fr_R_thick_G_cuneus_y6'], 'fr_R_thick_G_front_inf_Opercular': df['fr_R_thick_G_front_inf_Opercular_y6'], 'fr_R_thick_G_front_inf_Orbital': df['fr_R_thick_G_front_inf_Orbital_y6'], 'fr_R_thick_G_front_inf_Triangul': df['fr_R_thick_G_front_inf_Triangul_y6'], 'fr_R_thick_G_front_middle': df['fr_R_thick_G_front_middle_y6'], 'fr_R_thick_G_front_sup': df['fr_R_thick_G_front_sup_y6'], 'fr_R_thick_G_insular_short': df['fr_R_thick_G_insular_short_y6'], 'fr_R_thick_G_oc_temp_lat_fusifor': df['fr_R_thick_G_oc_temp_lat_fusifor_y6'], 'fr_R_thick_G_oc_temp_med_Lingual': df['fr_R_thick_G_oc_temp_med_Lingual_y6'], 'fr_R_thick_G_oc_temp_med_Parahip': df['fr_R_thick_G_oc_temp_med_Parahip_y6'], 'fr_R_thick_G_occipital_middle': df['fr_R_thick_G_occipital_middle_y6'], 'fr_R_thick_G_occipital_sup': df['fr_R_thick_G_occipital_sup_y6'], 'fr_R_thick_G_orbital': df['fr_R_thick_G_orbital_y6'], 'fr_R_thick_G_pariet_inf_Angular': df['fr_R_thick_G_pariet_inf_Angular_y6'], 'fr_R_thick_G_pariet_inf_Supramar': df['fr_R_thick_G_pariet_inf_Supramar_y6'], 'fr_R_thick_G_parietal_sup': df['fr_R_thick_G_parietal_sup_y6'], 'fr_R_thick_G_postcentral': df['fr_R_thick_G_postcentral_y6'], 'fr_R_thick_G_precentral': df['fr_R_thick_G_precentral_y6'], 'fr_R_thick_G_precuneus': df['fr_R_thick_G_precuneus_y6'], 'fr_R_thick_G_rectus': df['fr_R_thick_G_rectus_y6'], 'fr_R_thick_G_temp_sup_G_T_transv': df['fr_R_thick_G_temp_sup_G_T_transv_y6'], 'fr_R_thick_G_temp_sup_Lateral': df['fr_R_thick_G_temp_sup_Lateral_y6'], 'fr_R_thick_G_temp_sup_Plan_polar': df['fr_R_thick_G_temp_sup_Plan_polar_y6'], 'fr_R_thick_G_temp_sup_Plan_tempo': df['fr_R_thick_G_temp_sup_Plan_tempo_y6'], 'fr_R_thick_G_temporal_inf': df['fr_R_thick_G_temporal_inf_y6'], 'fr_R_thick_G_temporal_middle': df['fr_R_thick_G_temporal_middle_y6'], 'fr_R_thick_GandS_cingul_Ant': df['fr_R_thick_GandS_cingul_Ant_y6'], 'fr_R_thick_GandS_cingul_Mid_Post': df['fr_R_thick_GandS_cingul_Mid_Post_y6'], 'fr_R_thick_GandS_frontomargin': df['fr_R_thick_GandS_frontomargin_y6'], 'fr_R_thick_GandS_occipital_inf': df['fr_R_thick_GandS_occipital_inf_y6'], 'fr_R_thick_GandS_paracentral': df['fr_R_thick_GandS_paracentral_y6'], 'fr_R_thick_GandS_subcentral': df['fr_R_thick_GandS_subcentral_y6'], 'fr_R_thick_GandS_transv_frontopol': df['fr_R_thick_GandS_transv_frontopol_y6'], 'fr_R_thick_Lat_Fis_ant_Horizont': df['fr_R_thick_Lat_Fis_ant_Horizont_y6'], 'fr_R_thick_Lat_Fis_post': df['fr_R_thick_Lat_Fis_post_y6'], 'fr_R_thick_Pole_occipital': df['fr_R_thick_Pole_occipital_y6'], 'fr_R_thick_Pole_temporal': df['fr_R_thick_Pole_temporal_y6'], 'fr_R_thick_S_calcarine': df['fr_R_thick_S_calcarine_y6'], 'fr_R_thick_S_central': df['fr_R_thick_S_central_y6'], 'fr_R_thick_S_cingul_Marginalis': df['fr_R_thick_S_cingul_Marginalis_y6'], 'fr_R_thick_S_circular_insula_ant': df['fr_R_thick_S_circular_insula_ant_y6'], 'fr_R_thick_S_circular_insula_inf': df['fr_R_thick_S_circular_insula_inf_y6'], 'fr_R_thick_S_circular_insula_sup': df['fr_R_thick_S_circular_insula_sup_y6'], 'fr_R_thick_S_collat_transv_ant': df['fr_R_thick_S_collat_transv_ant_y6'], 'fr_R_thick_S_collat_transv_post': df['fr_R_thick_S_collat_transv_post_y6'], 'fr_R_thick_S_front_inf': df['fr_R_thick_S_front_inf_y6'], 'fr_R_thick_S_front_middle': df['fr_R_thick_S_front_middle_y6'], 'fr_R_thick_S_front_sup': df['fr_R_thick_S_front_sup_y6'], 'fr_R_thick_S_interm_prim_Jensen': df['fr_R_thick_S_interm_prim_Jensen_y6'], 'fr_R_thick_S_intraparietandP_trans': df['fr_R_thick_S_intraparietandP_trans_y6'], 'fr_R_thick_S_oc_middleandLunatus': df['fr_R_thick_S_oc_middleandLunatus_y6'], 'fr_R_thick_S_oc_temp_lat': df['fr_R_thick_S_oc_temp_lat_y6'], 'fr_R_thick_S_oc_temp_medandLingual': df['fr_R_thick_S_oc_temp_medandLingual_y6'], 'fr_R_thick_S_orbital_H_Shaped': df['fr_R_thick_S_orbital_H_Shaped_y6'], 'fr_R_thick_S_orbital_lateral': df['fr_R_thick_S_orbital_lateral_y6'], 'fr_R_thick_S_orbital_med_olfact': df['fr_R_thick_S_orbital_med_olfact_y6'], 'fr_R_thick_S_parieto_occipital': df['fr_R_thick_S_parieto_occipital_y6'], 'fr_R_thick_S_pericallosal': df['fr_R_thick_S_pericallosal_y6'], 'fr_R_thick_S_postcentral': df['fr_R_thick_S_postcentral_y6'], 'fr_R_thick_S_precentral_inf_part': df['fr_R_thick_S_precentral_inf_part_y6'], 'fr_R_thick_S_precentral_sup_part': df['fr_R_thick_S_precentral_sup_part_y6'], 'fr_R_thick_S_suborbital': df['fr_R_thick_S_suborbital_y6'], 'fr_R_thick_S_subparietal': df['fr_R_thick_S_subparietal_y6'], 'fr_R_thick_S_temporal_inf': df['fr_R_thick_S_temporal_inf_y6'], 'fr_R_thick_S_temporal_sup': df['fr_R_thick_S_temporal_sup_y6'], 'fr_R_thick_S_temporal_transverse': df['fr_R_thick_S_temporal_transverse_y6'],    'fr_L_thick_G_Ins_lgandS_cent_ins': df['fr_L_thick_G_Ins_lgandS_cent_ins_y6'], 'fr_L_thick_G_cingul_Post_ventral': df['fr_L_thick_G_cingul_Post_ventral_y6'], 'fr_L_thick_G_cuneus': df['fr_L_thick_G_cuneus_y6'], 'fr_L_thick_G_front_inf_Opercular': df['fr_L_thick_G_front_inf_Opercular_y6'], 'fr_L_thick_G_front_inf_Orbital': df['fr_L_thick_G_front_inf_Orbital_y6'], 'fr_L_thick_G_front_inf_Triangul': df['fr_L_thick_G_front_inf_Triangul_y6'], 'fr_L_thick_G_front_middle': df['fr_L_thick_G_front_middle_y6'], 'fr_L_thick_G_front_sup': df['fr_L_thick_G_front_sup_y6'], 'fr_L_thick_G_insular_short': df['fr_L_thick_G_insular_short_y6'], 'fr_L_thick_G_oc_temp_lat_fusifor': df['fr_L_thick_G_oc_temp_lat_fusifor_y6'], 'fr_L_thick_G_oc_temp_med_Lingual': df['fr_L_thick_G_oc_temp_med_Lingual_y6'], 'fr_L_thick_G_oc_temp_med_Parahip': df['fr_L_thick_G_oc_temp_med_Parahip_y6'], 'fr_L_thick_G_occipital_middle': df['fr_L_thick_G_occipital_middle_y6'], 'fr_L_thick_G_occipital_sup': df['fr_L_thick_G_occipital_sup_y6'], 'fr_L_thick_G_orbital': df['fr_L_thick_G_orbital_y6'], 'fr_L_thick_G_pariet_inf_Angular': df['fr_L_thick_G_pariet_inf_Angular_y6'], 'fr_L_thick_G_pariet_inf_Supramar': df['fr_L_thick_G_pariet_inf_Supramar_y6'], 'fr_L_thick_G_parietal_sup': df['fr_L_thick_G_parietal_sup_y6'], 'fr_L_thick_G_postcentral': df['fr_L_thick_G_postcentral_y6'], 'fr_L_thick_G_precentral': df['fr_L_thick_G_precentral_y6'], 'fr_L_thick_G_precuneus': df['fr_L_thick_G_precuneus_y6'], 'fr_L_thick_G_rectus': df['fr_L_thick_G_rectus_y6'], 'fr_L_thick_G_temp_sup_G_T_transv': df['fr_L_thick_G_temp_sup_G_T_transv_y6'], 'fr_L_thick_G_temp_sup_Lateral': df['fr_L_thick_G_temp_sup_Lateral_y6'], 'fr_L_thick_G_temp_sup_Plan_polar': df['fr_L_thick_G_temp_sup_Plan_polar_y6'], 'fr_L_thick_G_temp_sup_Plan_tempo': df['fr_L_thick_G_temp_sup_Plan_tempo_y6'], 'fr_L_thick_G_temporal_inf': df['fr_L_thick_G_temporal_inf_y6'], 'fr_L_thick_G_temporal_middle': df['fr_L_thick_G_temporal_middle_y6'], 'fr_L_thick_GandS_cingul_Ant': df['fr_L_thick_GandS_cingul_Ant_y6'], 'fr_L_thick_GandS_cingul_Mid_Post': df['fr_L_thick_GandS_cingul_Mid_Post_y6'], 'fr_L_thick_GandS_frontomargin': df['fr_L_thick_GandS_frontomargin_y6'], 'fr_L_thick_GandS_occipital_inf': df['fr_L_thick_GandS_occipital_inf_y6'], 'fr_L_thick_GandS_paracentral': df['fr_L_thick_GandS_paracentral_y6'], 'fr_L_thick_GandS_subcentral': df['fr_L_thick_GandS_subcentral_y6'], 'fr_L_thick_GandS_transv_frontopol': df['fr_L_thick_GandS_transv_frontopol_y6'], 'fr_L_thick_Lat_Fis_ant_Horizont': df['fr_L_thick_Lat_Fis_ant_Horizont_y6'], 'fr_L_thick_Lat_Fis_post': df['fr_L_thick_Lat_Fis_post_y6'], 'fr_L_thick_Pole_occipital': df['fr_L_thick_Pole_occipital_y6'], 'fr_L_thick_Pole_temporal': df['fr_L_thick_Pole_temporal_y6'], 'fr_L_thick_S_calcarine': df['fr_L_thick_S_calcarine_y6'], 'fr_L_thick_S_central': df['fr_L_thick_S_central_y6'], 'fr_L_thick_S_cingul_Marginalis': df['fr_L_thick_S_cingul_Marginalis_y6'], 'fr_L_thick_S_circular_insula_ant': df['fr_L_thick_S_circular_insula_ant_y6'], 'fr_L_thick_S_circular_insula_inf': df['fr_L_thick_S_circular_insula_inf_y6'], 'fr_L_thick_S_circular_insula_sup': df['fr_L_thick_S_circular_insula_sup_y6'], 'fr_L_thick_S_collat_transv_ant': df['fr_L_thick_S_collat_transv_ant_y6'], 'fr_L_thick_S_collat_transv_post': df['fr_L_thick_S_collat_transv_post_y6'], 'fr_L_thick_S_front_inf': df['fr_L_thick_S_front_inf_y6'], 'fr_L_thick_S_front_middle': df['fr_L_thick_S_front_middle_y6'], 'fr_L_thick_S_front_sup': df['fr_L_thick_S_front_sup_y6'], 'fr_L_thick_S_intraparietandP_trans': df['fr_L_thick_S_intraparietandP_trans_y6'], 'fr_L_thick_S_oc_middleandLunatus': df['fr_L_thick_S_oc_middleandLunatus_y6'], 'fr_L_thick_S_oc_temp_lat': df['fr_L_thick_S_oc_temp_lat_y6'], 'fr_L_thick_S_oc_temp_medandLingual': df['fr_L_thick_S_oc_temp_medandLingual_y6'], 'fr_L_thick_S_orbital_H_Shaped': df['fr_L_thick_S_orbital_H_Shaped_y6'], 'fr_L_thick_S_orbital_lateral': df['fr_L_thick_S_orbital_lateral_y6'], 'fr_L_thick_S_orbital_med_olfact': df['fr_L_thick_S_orbital_med_olfact_y6'], 'fr_L_thick_S_parieto_occipital': df['fr_L_thick_S_parieto_occipital_y6'], 'fr_L_thick_S_postcentral': df['fr_L_thick_S_postcentral_y6'], 'fr_L_thick_S_precentral_inf_part': df['fr_L_thick_S_precentral_inf_part_y6'], 'fr_L_thick_S_precentral_sup_part': df['fr_L_thick_S_precentral_sup_part_y6'], 'fr_L_thick_S_suborbital': df['fr_L_thick_S_suborbital_y6'], 'fr_L_thick_S_subparietal': df['fr_L_thick_S_subparietal_y6'], 'fr_L_thick_S_temporal_inf': df['fr_L_thick_S_temporal_inf_y6'], 'fr_L_thick_S_temporal_sup': df['fr_L_thick_S_temporal_sup_y6']}
	elif thorsf == 'surf':
		#frame_free1 = {'handlat': df['lat_manual'],'apoe': df['apoe'], 'y_last':df['ultimavisita'], 'dx_last': df['ultimodx'], 'dx_visita': df['dx_corto_visita1'], 'age_lastdx': df['edad_ultimodx'], 'age': df['edad_visita1'], 'sex': df['sexo'], 'free_R_Thal': df['fr_Right_Thalamus_Proper_y1'], 'free_L_Thal': df['fr_Left_Thalamus_Proper_y1'],'free_R_Puta': df['fr_Right_Putamen_y1'], 'free_L_Puta': df['fr_Left_Putamen_y1'],'free_R_Amyg': df['fr_Right_Amygdala_y1'], 'free_L_Amyg': df['fr_Left_Amygdala_y1'],'free_R_Pall': df['fr_Right_Pallidum_y1'], 'free_L_Pall': df['fr_Left_Pallidum_y1'],'free_R_Caud': df['fr_Right_Caudate_y1'], 'free_L_Caud': df['fr_Left_Caudate_y1'],'free_R_Hipp': df['fr_Right_Hippocampus_y1'], 'free_L_Hipp': df['fr_Left_Hippocampus_y1'],'free_R_Accu': df['fr_Right_Accumbens_area_y1'], 'free_L_Accu': df['fr_Left_Accumbens_area_y1'], 'fr_BrainSegVol': df['fr_BrainSegVol_y1'], 'fr_BrainSegVolNotVent':df['fr_BrainSegVolNotVent_y1'], 'fr_BrainSegVol_to_eTIV': df['fr_BrainSegVol_to_eTIV_y1']}
		frame_free1 = {'education': df['nivel_educativo'],'handlat': df['lat_manual'],'apoe': df['apoe'],'nvisita':df['nvisita1'] ,'y_last':df['ultimavisita'], 'dx_last': df['ultimodx'], 'dx_visita': df['dx_corto_visita1'], 'age_lastdx': df['edad_ultimodx'], 'age': df['edad_visita1'], 'sex': df['sexo'], 'free_R_Thal': df['fr_Right_Thalamus_Proper_y1'], 'free_L_Thal': df['fr_Left_Thalamus_Proper_y1'],'free_R_Puta': df['fr_Right_Putamen_y1'], 'free_L_Puta': df['fr_Left_Putamen_y1'],'free_R_Amyg': df['fr_Right_Amygdala_y1'], 'free_L_Amyg': df['fr_Left_Amygdala_y1'],'free_R_Pall': df['fr_Right_Pallidum_y1'], 'free_L_Pall': df['fr_Left_Pallidum_y1'],'free_R_Caud': df['fr_Right_Caudate_y1'], 'free_L_Caud': df['fr_Left_Caudate_y1'],'free_R_Hipp': df['fr_Right_Hippocampus_y1'], 'free_L_Hipp': df['fr_Left_Hippocampus_y1'],'free_R_Accu': df['fr_Right_Accumbens_area_y1'], 'free_L_Accu': df['fr_Left_Accumbens_area_y1'], 'fr_BrainSegVol': df['fr_BrainSegVol_y1'], 'fr_BrainSegVolNotVent':df['fr_BrainSegVolNotVent_y1'], 'fr_BrainSegVol_to_eTIV': df['fr_BrainSegVol_to_eTIV_y1'], 'fr_R_surf_G_Ins_lgandS_cent_ins': df['fr_R_surf_G_Ins_lgandS_cent_ins_y1'], 'fr_R_surf_G_cingul_Post_ventral': df['fr_R_surf_G_cingul_Post_ventral_y1'], 'fr_R_surf_G_cuneus': df['fr_R_surf_G_cuneus_y1'], 'fr_R_surf_G_front_inf_Opercular': df['fr_R_surf_G_front_inf_Opercular_y1'], 'fr_R_surf_G_front_inf_Orbital': df['fr_R_surf_G_front_inf_Orbital_y1'], 'fr_R_surf_G_front_inf_Triangul': df['fr_R_surf_G_front_inf_Triangul_y1'], 'fr_R_surf_G_front_middle': df['fr_R_surf_G_front_middle_y1'], 'fr_R_surf_G_front_sup': df['fr_R_surf_G_front_sup_y1'], 'fr_R_surf_G_insular_short': df['fr_R_surf_G_insular_short_y1'], 'fr_R_surf_G_oc_temp_lat_fusifor': df['fr_R_surf_G_oc_temp_lat_fusifor_y1'], 'fr_R_surf_G_oc_temp_med_Lingual': df['fr_R_surf_G_oc_temp_med_Lingual_y1'], 'fr_R_surf_G_oc_temp_med_Parahip': df['fr_R_surf_G_oc_temp_med_Parahip_y1'], 'fr_R_surf_G_occipital_middle': df['fr_R_surf_G_occipital_middle_y1'], 'fr_R_surf_G_occipital_sup': df['fr_R_surf_G_occipital_sup_y1'], 'fr_R_surf_G_orbital': df['fr_R_surf_G_orbital_y1'], 'fr_R_surf_G_pariet_inf_Angular': df['fr_R_surf_G_pariet_inf_Angular_y1'], 'fr_R_surf_G_pariet_inf_Supramar': df['fr_R_surf_G_pariet_inf_Supramar_y1'], 'fr_R_surf_G_parietal_sup': df['fr_R_surf_G_parietal_sup_y1'], 'fr_R_surf_G_postcentral': df['fr_R_surf_G_postcentral_y1'], 'fr_R_surf_G_precentral': df['fr_R_surf_G_precentral_y1'], 'fr_R_surf_G_precuneus': df['fr_R_surf_G_precuneus_y1'], 'fr_R_surf_G_rectus': df['fr_R_surf_G_rectus_y1'], 'fr_R_surf_G_temp_sup_G_T_transv': df['fr_R_surf_G_temp_sup_G_T_transv_y1'], 'fr_R_surf_G_temp_sup_Lateral': df['fr_R_surf_G_temp_sup_Lateral_y1'], 'fr_R_surf_G_temp_sup_Plan_polar': df['fr_R_surf_G_temp_sup_Plan_polar_y1'], 'fr_R_surf_G_temp_sup_Plan_tempo': df['fr_R_surf_G_temp_sup_Plan_tempo_y1'], 'fr_R_surf_G_temporal_inf': df['fr_R_surf_G_temporal_inf_y1'], 'fr_R_surf_G_temporal_middle': df['fr_R_surf_G_temporal_middle_y1'], 'fr_R_surf_GandS_cingul_Ant': df['fr_R_surf_GandS_cingul_Ant_y1'], 'fr_R_surf_GandS_cingul_Mid_Post': df['fr_R_surf_GandS_cingul_Mid_Post_y1'], 'fr_R_surf_GandS_frontomargin': df['fr_R_surf_GandS_frontomargin_y1'], 'fr_R_surf_GandS_occipital_inf': df['fr_R_surf_GandS_occipital_inf_y1'], 'fr_R_surf_GandS_paracentral': df['fr_R_surf_GandS_paracentral_y1'], 'fr_R_surf_GandS_subcentral': df['fr_R_surf_GandS_subcentral_y1'], 'fr_R_surf_GandS_transv_frontopol': df['fr_R_surf_GandS_transv_frontopol_y1'], 'fr_R_surf_Lat_Fis_ant_Horizont': df['fr_R_surf_Lat_Fis_ant_Horizont_y1'], 'fr_R_surf_Lat_Fis_post': df['fr_R_surf_Lat_Fis_post_y1'], 'fr_R_surf_Pole_occipital': df['fr_R_surf_Pole_occipital_y1'], 'fr_R_surf_Pole_temporal': df['fr_R_surf_Pole_temporal_y1'], 'fr_R_surf_S_calcarine': df['fr_R_surf_S_calcarine_y1'], 'fr_R_surf_S_central': df['fr_R_surf_S_central_y1'], 'fr_R_surf_S_cingul_Marginalis': df['fr_R_surf_S_cingul_Marginalis_y1'], 'fr_R_surf_S_circular_insula_ant': df['fr_R_surf_S_circular_insula_ant_y1'], 'fr_R_surf_S_circular_insula_inf': df['fr_R_surf_S_circular_insula_inf_y1'], 'fr_R_surf_S_circular_insula_sup': df['fr_R_surf_S_circular_insula_sup_y1'], 'fr_R_surf_S_collat_transv_ant': df['fr_R_surf_S_collat_transv_ant_y1'], 'fr_R_surf_S_collat_transv_post': df['fr_R_surf_S_collat_transv_post_y1'], 'fr_R_surf_S_front_inf': df['fr_R_surf_S_front_inf_y1'], 'fr_R_surf_S_front_middle': df['fr_R_surf_S_front_middle_y1'], 'fr_R_surf_S_front_sup': df['fr_R_surf_S_front_sup_y1'], 'fr_R_surf_S_interm_prim_Jensen': df['fr_R_surf_S_interm_prim_Jensen_y1'], 'fr_R_surf_S_intraparietandP_trans': df['fr_R_surf_S_intraparietandP_trans_y1'], 'fr_R_surf_S_oc_middleandLunatus': df['fr_R_surf_S_oc_middleandLunatus_y1'], 'fr_R_surf_S_oc_temp_lat': df['fr_R_surf_S_oc_temp_lat_y1'], 'fr_R_surf_S_oc_temp_medandLingual': df['fr_R_surf_S_oc_temp_medandLingual_y1'], 'fr_R_surf_S_orbital_H_Shaped': df['fr_R_surf_S_orbital_H_Shaped_y1'], 'fr_R_surf_S_orbital_lateral': df['fr_R_surf_S_orbital_lateral_y1'], 'fr_R_surf_S_orbital_med_olfact': df['fr_R_surf_S_orbital_med_olfact_y1'], 'fr_R_surf_S_parieto_occipital': df['fr_R_surf_S_parieto_occipital_y1'], 'fr_R_surf_S_pericallosal': df['fr_R_surf_S_pericallosal_y1'], 'fr_R_surf_S_postcentral': df['fr_R_surf_S_postcentral_y1'], 'fr_R_surf_S_precentral_inf_part': df['fr_R_surf_S_precentral_inf_part_y1'], 'fr_R_surf_S_precentral_sup_part': df['fr_R_surf_S_precentral_sup_part_y1'], 'fr_R_surf_S_suborbital': df['fr_R_surf_S_suborbital_y1'], 'fr_R_surf_S_subparietal': df['fr_R_surf_S_subparietal_y1'], 'fr_R_surf_S_temporal_inf': df['fr_R_surf_S_temporal_inf_y1'], 'fr_R_surf_S_temporal_sup': df['fr_R_surf_S_temporal_sup_y1'], 'fr_R_surf_S_temporal_transverse': df['fr_R_surf_S_temporal_transverse_y1'], 'fr_L_surf_G_Ins_lgandS_cent_ins': df['fr_L_surf_G_Ins_lgandS_cent_ins_y1'], 'fr_L_surf_G_cingul_Post_ventral': df['fr_L_surf_G_cingul_Post_ventral_y1'], 'fr_L_surf_G_cuneus': df['fr_L_surf_G_cuneus_y1'], 'fr_L_surf_G_front_inf_Opercular': df['fr_L_surf_G_front_inf_Opercular_y1'], 'fr_L_surf_G_front_inf_Orbital': df['fr_L_surf_G_front_inf_Orbital_y1'], 'fr_L_surf_G_front_inf_Triangul': df['fr_L_surf_G_front_inf_Triangul_y1'], 'fr_L_surf_G_front_middle': df['fr_L_surf_G_front_middle_y1'], 'fr_L_surf_G_front_sup': df['fr_L_surf_G_front_sup_y1'], 'fr_L_surf_G_insular_short': df['fr_L_surf_G_insular_short_y1'], 'fr_L_surf_G_oc_temp_lat_fusifor': df['fr_L_surf_G_oc_temp_lat_fusifor_y1'], 'fr_L_surf_G_oc_temp_med_Lingual': df['fr_L_surf_G_oc_temp_med_Lingual_y1'], 'fr_L_surf_G_oc_temp_med_Parahip': df['fr_L_surf_G_oc_temp_med_Parahip_y1'], 'fr_L_surf_G_occipital_middle': df['fr_L_surf_G_occipital_middle_y1'], 'fr_L_surf_G_occipital_sup': df['fr_L_surf_G_occipital_sup_y1'], 'fr_L_surf_G_orbital': df['fr_L_surf_G_orbital_y1'], 'fr_L_surf_G_pariet_inf_Angular': df['fr_L_surf_G_pariet_inf_Angular_y1'], 'fr_L_surf_G_pariet_inf_Supramar': df['fr_L_surf_G_pariet_inf_Supramar_y1'], 'fr_L_surf_G_parietal_sup': df['fr_L_surf_G_parietal_sup_y1'], 'fr_L_surf_G_postcentral': df['fr_L_surf_G_postcentral_y1'], 'fr_L_surf_G_precentral': df['fr_L_surf_G_precentral_y1'], 'fr_L_surf_G_precuneus': df['fr_L_surf_G_precuneus_y1'], 'fr_L_surf_G_rectus': df['fr_L_surf_G_rectus_y1'], 'fr_L_surf_G_temp_sup_G_T_transv': df['fr_L_surf_G_temp_sup_G_T_transv_y1'], 'fr_L_surf_G_temp_sup_Lateral': df['fr_L_surf_G_temp_sup_Lateral_y1'], 'fr_L_surf_G_temp_sup_Plan_polar': df['fr_L_surf_G_temp_sup_Plan_polar_y1'], 'fr_L_surf_G_temp_sup_Plan_tempo': df['fr_L_surf_G_temp_sup_Plan_tempo_y1'], 'fr_L_surf_G_temporal_inf': df['fr_L_surf_G_temporal_inf_y1'], 'fr_L_surf_G_temporal_middle': df['fr_L_surf_G_temporal_middle_y1'], 'fr_L_surf_GandS_cingul_Ant': df['fr_L_surf_GandS_cingul_Ant_y1'], 'fr_L_surf_GandS_cingul_Mid_Post': df['fr_L_surf_GandS_cingul_Mid_Post_y1'], 'fr_L_surf_GandS_frontomargin': df['fr_L_surf_GandS_frontomargin_y1'], 'fr_L_surf_GandS_occipital_inf': df['fr_L_surf_GandS_occipital_inf_y1'], 'fr_L_surf_GandS_paracentral': df['fr_L_surf_GandS_paracentral_y1'], 'fr_L_surf_GandS_subcentral': df['fr_L_surf_GandS_subcentral_y1'], 'fr_L_surf_GandS_transv_frontopol': df['fr_L_surf_GandS_transv_frontopol_y1'], 'fr_L_surf_Lat_Fis_ant_Horizont': df['fr_L_surf_Lat_Fis_ant_Horizont_y1'], 'fr_L_surf_Lat_Fis_post': df['fr_L_surf_Lat_Fis_post_y1'], 'fr_L_surf_Pole_occipital': df['fr_L_surf_Pole_occipital_y1'], 'fr_L_surf_Pole_temporal': df['fr_L_surf_Pole_temporal_y1'], 'fr_L_surf_S_calcarine': df['fr_L_surf_S_calcarine_y1'], 'fr_L_surf_S_central': df['fr_L_surf_S_central_y1'], 'fr_L_surf_S_cingul_Marginalis': df['fr_L_surf_S_cingul_Marginalis_y1'], 'fr_L_surf_S_circular_insula_ant': df['fr_L_surf_S_circular_insula_ant_y1'], 'fr_L_surf_S_circular_insula_inf': df['fr_L_surf_S_circular_insula_inf_y1'], 'fr_L_surf_S_circular_insula_sup': df['fr_L_surf_S_circular_insula_sup_y1'], 'fr_L_surf_S_collat_transv_ant': df['fr_L_surf_S_collat_transv_ant_y1'], 'fr_L_surf_S_collat_transv_post': df['fr_L_surf_S_collat_transv_post_y1'], 'fr_L_surf_S_front_inf': df['fr_L_surf_S_front_inf_y1'], 'fr_L_surf_S_front_middle': df['fr_L_surf_S_front_middle_y1'], 'fr_L_surf_S_front_sup': df['fr_L_surf_S_front_sup_y1'],  'fr_L_surf_S_intraparietandP_trans': df['fr_L_surf_S_intraparietandP_trans_y1'], 'fr_L_surf_S_oc_middleandLunatus': df['fr_L_surf_S_oc_middleandLunatus_y1'], 'fr_L_surf_S_oc_temp_lat': df['fr_L_surf_S_oc_temp_lat_y1'], 'fr_L_surf_S_oc_temp_medandLingual': df['fr_L_surf_S_oc_temp_medandLingual_y1'], 'fr_L_surf_S_orbital_H_Shaped': df['fr_L_surf_S_orbital_H_Shaped_y1'], 'fr_L_surf_S_orbital_lateral': df['fr_L_surf_S_orbital_lateral_y1'], 'fr_L_surf_S_orbital_med_olfact': df['fr_L_surf_S_orbital_med_olfact_y1'], 'fr_L_surf_S_parieto_occipital': df['fr_L_surf_S_parieto_occipital_y1'], 'fr_L_surf_S_postcentral': df['fr_L_surf_S_postcentral_y1'], 'fr_L_surf_S_precentral_inf_part': df['fr_L_surf_S_precentral_inf_part_y1'], 'fr_L_surf_S_precentral_sup_part': df['fr_L_surf_S_precentral_sup_part_y1'], 'fr_L_surf_S_suborbital': df['fr_L_surf_S_suborbital_y1'], 'fr_L_surf_S_subparietal': df['fr_L_surf_S_subparietal_y1'], 'fr_L_surf_S_temporal_inf': df['fr_L_surf_S_temporal_inf_y1'], 'fr_L_surf_S_temporal_sup': df['fr_L_surf_S_temporal_sup_y1']}
		#frame_free2 = {'handlat': df['lat_manual'],'apoe': df['apoe'], 'y_last':df['ultimavisita'], 'dx_last': df['ultimodx'], 'dx_visita': df['dx_corto_visita2'], 'age_lastdx': df['edad_ultimodx'], 'age': df['edad_visita2'], 'sex': df['sexo'], 'free_R_Thal': df['fr_Right_Thalamus_Proper_y2'], 'free_L_Thal': df['fr_Left_Thalamus_Proper_y2'],'free_R_Puta': df['fr_Right_Putamen_y2'], 'free_L_Puta': df['fr_Left_Putamen_y2'],'free_R_Amyg': df['fr_Right_Amygdala_y2'], 'free_L_Amyg': df['fr_Left_Amygdala_y2'],'free_R_Pall': df['fr_Right_Pallidum_y2'], 'free_L_Pall': df['fr_Left_Pallidum_y2'],'free_R_Caud': df['fr_Right_Caudate_y2'], 'free_L_Caud': df['fr_Left_Caudate_y2'],'free_R_Hipp': df['fr_Right_Hippocampus_y2'], 'free_L_Hipp': df['fr_Left_Hippocampus_y2'],'free_R_Accu': df['fr_Right_Accumbens_area_y2'], 'free_L_Accu': df['fr_Left_Accumbens_area_y2'], 'fr_BrainSegVol': df['fr_BrainSegVol_y2'], 'fr_BrainSegVolNotVent':df['fr_BrainSegVolNotVent_y2'], 'fr_BrainSegVol_to_eTIV': df['fr_BrainSegVol_to_eTIV_y2']}
		frame_free2 = {'education': df['nivel_educativo'],'handlat': df['lat_manual'],'apoe': df['apoe'], 'nvisita':df['nvisita2'], 'y_last':df['ultimavisita'], 'dx_last': df['ultimodx'], 'dx_visita': df['dx_corto_visita2'], 'age_lastdx': df['edad_ultimodx'], 'age': df['edad_visita2'], 'sex': df['sexo'], 'free_R_Thal': df['fr_Right_Thalamus_Proper_y2'], 'free_L_Thal': df['fr_Left_Thalamus_Proper_y2'],'free_R_Puta': df['fr_Right_Putamen_y2'], 'free_L_Puta': df['fr_Left_Putamen_y2'],'free_R_Amyg': df['fr_Right_Amygdala_y2'], 'free_L_Amyg': df['fr_Left_Amygdala_y2'],'free_R_Pall': df['fr_Right_Pallidum_y2'], 'free_L_Pall': df['fr_Left_Pallidum_y2'],'free_R_Caud': df['fr_Right_Caudate_y2'], 'free_L_Caud': df['fr_Left_Caudate_y2'],'free_R_Hipp': df['fr_Right_Hippocampus_y2'], 'free_L_Hipp': df['fr_Left_Hippocampus_y2'],'free_R_Accu': df['fr_Right_Accumbens_area_y2'], 'free_L_Accu': df['fr_Left_Accumbens_area_y2'], 'fr_BrainSegVol': df['fr_BrainSegVol_y2'], 'fr_BrainSegVolNotVent':df['fr_BrainSegVolNotVent_y2'], 'fr_BrainSegVol_to_eTIV': df['fr_BrainSegVol_to_eTIV_y2'],  'fr_R_surf_G_Ins_lgandS_cent_ins': df['fr_R_surf_G_Ins_lgandS_cent_ins_y2'], 'fr_R_surf_G_cingul_Post_ventral': df['fr_R_surf_G_cingul_Post_ventral_y2'], 'fr_R_surf_G_cuneus': df['fr_R_surf_G_cuneus_y2'], 'fr_R_surf_G_front_inf_Opercular': df['fr_R_surf_G_front_inf_Opercular_y2'], 'fr_R_surf_G_front_inf_Orbital': df['fr_R_surf_G_front_inf_Orbital_y2'], 'fr_R_surf_G_front_inf_Triangul': df['fr_R_surf_G_front_inf_Triangul_y2'], 'fr_R_surf_G_front_middle': df['fr_R_surf_G_front_middle_y2'], 'fr_R_surf_G_front_sup': df['fr_R_surf_G_front_sup_y2'], 'fr_R_surf_G_insular_short': df['fr_R_surf_G_insular_short_y2'], 'fr_R_surf_G_oc_temp_lat_fusifor': df['fr_R_surf_G_oc_temp_lat_fusifor_y2'], 'fr_R_surf_G_oc_temp_med_Lingual': df['fr_R_surf_G_oc_temp_med_Lingual_y2'], 'fr_R_surf_G_oc_temp_med_Parahip': df['fr_R_surf_G_oc_temp_med_Parahip_y2'], 'fr_R_surf_G_occipital_middle': df['fr_R_surf_G_occipital_middle_y2'], 'fr_R_surf_G_occipital_sup': df['fr_R_surf_G_occipital_sup_y2'], 'fr_R_surf_G_orbital': df['fr_R_surf_G_orbital_y2'], 'fr_R_surf_G_pariet_inf_Angular': df['fr_R_surf_G_pariet_inf_Angular_y2'], 'fr_R_surf_G_pariet_inf_Supramar': df['fr_R_surf_G_pariet_inf_Supramar_y2'], 'fr_R_surf_G_parietal_sup': df['fr_R_surf_G_parietal_sup_y2'], 'fr_R_surf_G_postcentral': df['fr_R_surf_G_postcentral_y2'], 'fr_R_surf_G_precentral': df['fr_R_surf_G_precentral_y2'], 'fr_R_surf_G_precuneus': df['fr_R_surf_G_precuneus_y2'], 'fr_R_surf_G_rectus': df['fr_R_surf_G_rectus_y2'], 'fr_R_surf_G_temp_sup_G_T_transv': df['fr_R_surf_G_temp_sup_G_T_transv_y2'], 'fr_R_surf_G_temp_sup_Lateral': df['fr_R_surf_G_temp_sup_Lateral_y2'], 'fr_R_surf_G_temp_sup_Plan_polar': df['fr_R_surf_G_temp_sup_Plan_polar_y2'], 'fr_R_surf_G_temp_sup_Plan_tempo': df['fr_R_surf_G_temp_sup_Plan_tempo_y2'], 'fr_R_surf_G_temporal_inf': df['fr_R_surf_G_temporal_inf_y2'], 'fr_R_surf_G_temporal_middle': df['fr_R_surf_G_temporal_middle_y2'], 'fr_R_surf_GandS_cingul_Ant': df['fr_R_surf_GandS_cingul_Ant_y2'], 'fr_R_surf_GandS_cingul_Mid_Post': df['fr_R_surf_GandS_cingul_Mid_Post_y2'], 'fr_R_surf_GandS_frontomargin': df['fr_R_surf_GandS_frontomargin_y2'], 'fr_R_surf_GandS_occipital_inf': df['fr_R_surf_GandS_occipital_inf_y2'], 'fr_R_surf_GandS_paracentral': df['fr_R_surf_GandS_paracentral_y2'], 'fr_R_surf_GandS_subcentral': df['fr_R_surf_GandS_subcentral_y2'], 'fr_R_surf_GandS_transv_frontopol': df['fr_R_surf_GandS_transv_frontopol_y2'], 'fr_R_surf_Lat_Fis_ant_Horizont': df['fr_R_surf_Lat_Fis_ant_Horizont_y2'], 'fr_R_surf_Lat_Fis_post': df['fr_R_surf_Lat_Fis_post_y2'], 'fr_R_surf_Pole_occipital': df['fr_R_surf_Pole_occipital_y2'], 'fr_R_surf_Pole_temporal': df['fr_R_surf_Pole_temporal_y2'], 'fr_R_surf_S_calcarine': df['fr_R_surf_S_calcarine_y2'], 'fr_R_surf_S_central': df['fr_R_surf_S_central_y2'], 'fr_R_surf_S_cingul_Marginalis': df['fr_R_surf_S_cingul_Marginalis_y2'], 'fr_R_surf_S_circular_insula_ant': df['fr_R_surf_S_circular_insula_ant_y2'], 'fr_R_surf_S_circular_insula_inf': df['fr_R_surf_S_circular_insula_inf_y2'], 'fr_R_surf_S_circular_insula_sup': df['fr_R_surf_S_circular_insula_sup_y2'], 'fr_R_surf_S_collat_transv_ant': df['fr_R_surf_S_collat_transv_ant_y2'], 'fr_R_surf_S_collat_transv_post': df['fr_R_surf_S_collat_transv_post_y2'], 'fr_R_surf_S_front_inf': df['fr_R_surf_S_front_inf_y2'], 'fr_R_surf_S_front_middle': df['fr_R_surf_S_front_middle_y2'], 'fr_R_surf_S_front_sup': df['fr_R_surf_S_front_sup_y2'], 'fr_R_surf_S_interm_prim_Jensen': df['fr_R_surf_S_interm_prim_Jensen_y2'], 'fr_R_surf_S_intraparietandP_trans': df['fr_R_surf_S_intraparietandP_trans_y2'], 'fr_R_surf_S_oc_middleandLunatus': df['fr_R_surf_S_oc_middleandLunatus_y2'], 'fr_R_surf_S_oc_temp_lat': df['fr_R_surf_S_oc_temp_lat_y2'], 'fr_R_surf_S_oc_temp_medandLingual': df['fr_R_surf_S_oc_temp_medandLingual_y2'], 'fr_R_surf_S_orbital_H_Shaped': df['fr_R_surf_S_orbital_H_Shaped_y2'], 'fr_R_surf_S_orbital_lateral': df['fr_R_surf_S_orbital_lateral_y2'], 'fr_R_surf_S_orbital_med_olfact': df['fr_R_surf_S_orbital_med_olfact_y2'], 'fr_R_surf_S_parieto_occipital': df['fr_R_surf_S_parieto_occipital_y2'], 'fr_R_surf_S_pericallosal': df['fr_R_surf_S_pericallosal_y2'], 'fr_R_surf_S_postcentral': df['fr_R_surf_S_postcentral_y2'], 'fr_R_surf_S_precentral_inf_part': df['fr_R_surf_S_precentral_inf_part_y2'], 'fr_R_surf_S_precentral_sup_part': df['fr_R_surf_S_precentral_sup_part_y2'], 'fr_R_surf_S_suborbital': df['fr_R_surf_S_suborbital_y2'], 'fr_R_surf_S_subparietal': df['fr_R_surf_S_subparietal_y2'], 'fr_R_surf_S_temporal_inf': df['fr_R_surf_S_temporal_inf_y2'], 'fr_R_surf_S_temporal_sup': df['fr_R_surf_S_temporal_sup_y2'], 'fr_R_surf_S_temporal_transverse': df['fr_R_surf_S_temporal_transverse_y2'],'fr_L_surf_G_Ins_lgandS_cent_ins': df['fr_L_surf_G_Ins_lgandS_cent_ins_y2'], 'fr_L_surf_G_cingul_Post_ventral': df['fr_L_surf_G_cingul_Post_ventral_y2'], 'fr_L_surf_G_cuneus': df['fr_L_surf_G_cuneus_y2'], 'fr_L_surf_G_front_inf_Opercular': df['fr_L_surf_G_front_inf_Opercular_y2'], 'fr_L_surf_G_front_inf_Orbital': df['fr_L_surf_G_front_inf_Orbital_y2'], 'fr_L_surf_G_front_inf_Triangul': df['fr_L_surf_G_front_inf_Triangul_y2'], 'fr_L_surf_G_front_middle': df['fr_L_surf_G_front_middle_y2'], 'fr_L_surf_G_front_sup': df['fr_L_surf_G_front_sup_y2'], 'fr_L_surf_G_insular_short': df['fr_L_surf_G_insular_short_y2'], 'fr_L_surf_G_oc_temp_lat_fusifor': df['fr_L_surf_G_oc_temp_lat_fusifor_y2'], 'fr_L_surf_G_oc_temp_med_Lingual': df['fr_L_surf_G_oc_temp_med_Lingual_y2'], 'fr_L_surf_G_oc_temp_med_Parahip': df['fr_L_surf_G_oc_temp_med_Parahip_y2'], 'fr_L_surf_G_occipital_middle': df['fr_L_surf_G_occipital_middle_y2'], 'fr_L_surf_G_occipital_sup': df['fr_L_surf_G_occipital_sup_y2'], 'fr_L_surf_G_orbital': df['fr_L_surf_G_orbital_y2'], 'fr_L_surf_G_pariet_inf_Angular': df['fr_L_surf_G_pariet_inf_Angular_y2'], 'fr_L_surf_G_pariet_inf_Supramar': df['fr_L_surf_G_pariet_inf_Supramar_y2'], 'fr_L_surf_G_parietal_sup': df['fr_L_surf_G_parietal_sup_y2'], 'fr_L_surf_G_postcentral': df['fr_L_surf_G_postcentral_y2'], 'fr_L_surf_G_precentral': df['fr_L_surf_G_precentral_y2'], 'fr_L_surf_G_precuneus': df['fr_L_surf_G_precuneus_y2'], 'fr_L_surf_G_rectus': df['fr_L_surf_G_rectus_y2'], 'fr_L_surf_G_temp_sup_G_T_transv': df['fr_L_surf_G_temp_sup_G_T_transv_y2'], 'fr_L_surf_G_temp_sup_Lateral': df['fr_L_surf_G_temp_sup_Lateral_y2'], 'fr_L_surf_G_temp_sup_Plan_polar': df['fr_L_surf_G_temp_sup_Plan_polar_y2'], 'fr_L_surf_G_temp_sup_Plan_tempo': df['fr_L_surf_G_temp_sup_Plan_tempo_y2'], 'fr_L_surf_G_temporal_inf': df['fr_L_surf_G_temporal_inf_y2'], 'fr_L_surf_G_temporal_middle': df['fr_L_surf_G_temporal_middle_y2'], 'fr_L_surf_GandS_cingul_Ant': df['fr_L_surf_GandS_cingul_Ant_y2'], 'fr_L_surf_GandS_cingul_Mid_Post': df['fr_L_surf_GandS_cingul_Mid_Post_y2'], 'fr_L_surf_GandS_frontomargin': df['fr_L_surf_GandS_frontomargin_y2'], 'fr_L_surf_GandS_occipital_inf': df['fr_L_surf_GandS_occipital_inf_y2'], 'fr_L_surf_GandS_paracentral': df['fr_L_surf_GandS_paracentral_y2'], 'fr_L_surf_GandS_subcentral': df['fr_L_surf_GandS_subcentral_y2'], 'fr_L_surf_GandS_transv_frontopol': df['fr_L_surf_GandS_transv_frontopol_y2'], 'fr_L_surf_Lat_Fis_ant_Horizont': df['fr_L_surf_Lat_Fis_ant_Horizont_y2'], 'fr_L_surf_Lat_Fis_post': df['fr_L_surf_Lat_Fis_post_y2'], 'fr_L_surf_Pole_occipital': df['fr_L_surf_Pole_occipital_y2'], 'fr_L_surf_Pole_temporal': df['fr_L_surf_Pole_temporal_y2'], 'fr_L_surf_S_calcarine': df['fr_L_surf_S_calcarine_y2'], 'fr_L_surf_S_central': df['fr_L_surf_S_central_y2'], 'fr_L_surf_S_cingul_Marginalis': df['fr_L_surf_S_cingul_Marginalis_y2'], 'fr_L_surf_S_circular_insula_ant': df['fr_L_surf_S_circular_insula_ant_y2'], 'fr_L_surf_S_circular_insula_inf': df['fr_L_surf_S_circular_insula_inf_y2'], 'fr_L_surf_S_circular_insula_sup': df['fr_L_surf_S_circular_insula_sup_y2'], 'fr_L_surf_S_collat_transv_ant': df['fr_L_surf_S_collat_transv_ant_y2'], 'fr_L_surf_S_collat_transv_post': df['fr_L_surf_S_collat_transv_post_y2'], 'fr_L_surf_S_front_inf': df['fr_L_surf_S_front_inf_y2'], 'fr_L_surf_S_front_middle': df['fr_L_surf_S_front_middle_y2'], 'fr_L_surf_S_front_sup': df['fr_L_surf_S_front_sup_y2'],  'fr_L_surf_S_intraparietandP_trans': df['fr_L_surf_S_intraparietandP_trans_y2'], 'fr_L_surf_S_oc_middleandLunatus': df['fr_L_surf_S_oc_middleandLunatus_y2'], 'fr_L_surf_S_oc_temp_lat': df['fr_L_surf_S_oc_temp_lat_y2'], 'fr_L_surf_S_oc_temp_medandLingual': df['fr_L_surf_S_oc_temp_medandLingual_y2'], 'fr_L_surf_S_orbital_H_Shaped': df['fr_L_surf_S_orbital_H_Shaped_y2'], 'fr_L_surf_S_orbital_lateral': df['fr_L_surf_S_orbital_lateral_y2'], 'fr_L_surf_S_orbital_med_olfact': df['fr_L_surf_S_orbital_med_olfact_y2'], 'fr_L_surf_S_parieto_occipital': df['fr_L_surf_S_parieto_occipital_y2'], 'fr_L_surf_S_postcentral': df['fr_L_surf_S_postcentral_y2'], 'fr_L_surf_S_precentral_inf_part': df['fr_L_surf_S_precentral_inf_part_y2'], 'fr_L_surf_S_precentral_sup_part': df['fr_L_surf_S_precentral_sup_part_y2'], 'fr_L_surf_S_suborbital': df['fr_L_surf_S_suborbital_y2'], 'fr_L_surf_S_subparietal': df['fr_L_surf_S_subparietal_y2'], 'fr_L_surf_S_temporal_inf': df['fr_L_surf_S_temporal_inf_y2'], 'fr_L_surf_S_temporal_sup': df['fr_L_surf_S_temporal_sup_y2']}
		#frame_free3 = {'handlat': df['lat_manual'],'apoe': df['apoe'], 'y_last':df['ultimavisita'], 'dx_last': df['ultimodx'], 'dx_visita': df['dx_corto_visita3'], 'age_lastdx': df['edad_ultimodx'], 'age': df['edad_visita3'], 'sex': df['sexo'], 'free_R_Thal': df['fr_Right_Thalamus_Proper_y3'], 'free_L_Thal': df['fr_Left_Thalamus_Proper_y3'],'free_R_Puta': df['fr_Right_Putamen_y3'], 'free_L_Puta': df['fr_Left_Putamen_y3'],'free_R_Amyg': df['fr_Right_Amygdala_y3'], 'free_L_Amyg': df['fr_Left_Amygdala_y3'],'free_R_Pall': df['fr_Right_Pallidum_y3'], 'free_L_Pall': df['fr_Left_Pallidum_y3'],'free_R_Caud': df['fr_Right_Caudate_y3'], 'free_L_Caud': df['fr_Left_Caudate_y3'],'free_R_Hipp': df['fr_Right_Hippocampus_y3'], 'free_L_Hipp': df['fr_Left_Hippocampus_y3'],'free_R_Accu': df['fr_Right_Accumbens_area_y3'], 'free_L_Accu': df['fr_Left_Accumbens_area_y3'], 'fr_BrainSegVol': df['fr_BrainSegVol_y3'], 'fr_BrainSegVolNotVent':df['fr_BrainSegVolNotVent_y3'], 'fr_BrainSegVol_to_eTIV': df['fr_BrainSegVol_to_eTIV_y3']}
		frame_free3 = {'education': df['nivel_educativo'],'handlat': df['lat_manual'],'apoe': df['apoe'], 'nvisita':df['nvisita3'] ,'y_last':df['ultimavisita'], 'dx_last': df['ultimodx'], 'dx_visita': df['dx_corto_visita3'], 'age_lastdx': df['edad_ultimodx'], 'age': df['edad_visita3'], 'sex': df['sexo'], 'free_R_Thal': df['fr_Right_Thalamus_Proper_y3'], 'free_L_Thal': df['fr_Left_Thalamus_Proper_y3'],'free_R_Puta': df['fr_Right_Putamen_y3'], 'free_L_Puta': df['fr_Left_Putamen_y3'],'free_R_Amyg': df['fr_Right_Amygdala_y3'], 'free_L_Amyg': df['fr_Left_Amygdala_y3'],'free_R_Pall': df['fr_Right_Pallidum_y3'], 'free_L_Pall': df['fr_Left_Pallidum_y3'],'free_R_Caud': df['fr_Right_Caudate_y3'], 'free_L_Caud': df['fr_Left_Caudate_y3'],'free_R_Hipp': df['fr_Right_Hippocampus_y3'], 'free_L_Hipp': df['fr_Left_Hippocampus_y3'],'free_R_Accu': df['fr_Right_Accumbens_area_y3'], 'free_L_Accu': df['fr_Left_Accumbens_area_y3'], 'fr_BrainSegVol': df['fr_BrainSegVol_y3'], 'fr_BrainSegVolNotVent':df['fr_BrainSegVolNotVent_y3'], 'fr_BrainSegVol_to_eTIV': df['fr_BrainSegVol_to_eTIV_y3'],  'fr_R_surf_G_Ins_lgandS_cent_ins': df['fr_R_surf_G_Ins_lgandS_cent_ins_y3'], 'fr_R_surf_G_cingul_Post_ventral': df['fr_R_surf_G_cingul_Post_ventral_y3'], 'fr_R_surf_G_cuneus': df['fr_R_surf_G_cuneus_y3'], 'fr_R_surf_G_front_inf_Opercular': df['fr_R_surf_G_front_inf_Opercular_y3'], 'fr_R_surf_G_front_inf_Orbital': df['fr_R_surf_G_front_inf_Orbital_y3'], 'fr_R_surf_G_front_inf_Triangul': df['fr_R_surf_G_front_inf_Triangul_y3'], 'fr_R_surf_G_front_middle': df['fr_R_surf_G_front_middle_y3'], 'fr_R_surf_G_front_sup': df['fr_R_surf_G_front_sup_y3'], 'fr_R_surf_G_insular_short': df['fr_R_surf_G_insular_short_y3'], 'fr_R_surf_G_oc_temp_lat_fusifor': df['fr_R_surf_G_oc_temp_lat_fusifor_y3'], 'fr_R_surf_G_oc_temp_med_Lingual': df['fr_R_surf_G_oc_temp_med_Lingual_y3'], 'fr_R_surf_G_oc_temp_med_Parahip': df['fr_R_surf_G_oc_temp_med_Parahip_y3'], 'fr_R_surf_G_occipital_middle': df['fr_R_surf_G_occipital_middle_y3'], 'fr_R_surf_G_occipital_sup': df['fr_R_surf_G_occipital_sup_y3'], 'fr_R_surf_G_orbital': df['fr_R_surf_G_orbital_y3'], 'fr_R_surf_G_pariet_inf_Angular': df['fr_R_surf_G_pariet_inf_Angular_y3'], 'fr_R_surf_G_pariet_inf_Supramar': df['fr_R_surf_G_pariet_inf_Supramar_y3'], 'fr_R_surf_G_parietal_sup': df['fr_R_surf_G_parietal_sup_y3'], 'fr_R_surf_G_postcentral': df['fr_R_surf_G_postcentral_y3'], 'fr_R_surf_G_precentral': df['fr_R_surf_G_precentral_y3'], 'fr_R_surf_G_precuneus': df['fr_R_surf_G_precuneus_y3'], 'fr_R_surf_G_rectus': df['fr_R_surf_G_rectus_y3'], 'fr_R_surf_G_temp_sup_G_T_transv': df['fr_R_surf_G_temp_sup_G_T_transv_y3'], 'fr_R_surf_G_temp_sup_Lateral': df['fr_R_surf_G_temp_sup_Lateral_y3'], 'fr_R_surf_G_temp_sup_Plan_polar': df['fr_R_surf_G_temp_sup_Plan_polar_y3'], 'fr_R_surf_G_temp_sup_Plan_tempo': df['fr_R_surf_G_temp_sup_Plan_tempo_y3'], 'fr_R_surf_G_temporal_inf': df['fr_R_surf_G_temporal_inf_y3'], 'fr_R_surf_G_temporal_middle': df['fr_R_surf_G_temporal_middle_y3'], 'fr_R_surf_GandS_cingul_Ant': df['fr_R_surf_GandS_cingul_Ant_y3'], 'fr_R_surf_GandS_cingul_Mid_Post': df['fr_R_surf_GandS_cingul_Mid_Post_y3'], 'fr_R_surf_GandS_frontomargin': df['fr_R_surf_GandS_frontomargin_y3'], 'fr_R_surf_GandS_occipital_inf': df['fr_R_surf_GandS_occipital_inf_y3'], 'fr_R_surf_GandS_paracentral': df['fr_R_surf_GandS_paracentral_y3'], 'fr_R_surf_GandS_subcentral': df['fr_R_surf_GandS_subcentral_y3'], 'fr_R_surf_GandS_transv_frontopol': df['fr_R_surf_GandS_transv_frontopol_y3'], 'fr_R_surf_Lat_Fis_ant_Horizont': df['fr_R_surf_Lat_Fis_ant_Horizont_y3'], 'fr_R_surf_Lat_Fis_post': df['fr_R_surf_Lat_Fis_post_y3'], 'fr_R_surf_Pole_occipital': df['fr_R_surf_Pole_occipital_y3'], 'fr_R_surf_Pole_temporal': df['fr_R_surf_Pole_temporal_y3'], 'fr_R_surf_S_calcarine': df['fr_R_surf_S_calcarine_y3'], 'fr_R_surf_S_central': df['fr_R_surf_S_central_y3'], 'fr_R_surf_S_cingul_Marginalis': df['fr_R_surf_S_cingul_Marginalis_y3'], 'fr_R_surf_S_circular_insula_ant': df['fr_R_surf_S_circular_insula_ant_y3'], 'fr_R_surf_S_circular_insula_inf': df['fr_R_surf_S_circular_insula_inf_y3'], 'fr_R_surf_S_circular_insula_sup': df['fr_R_surf_S_circular_insula_sup_y3'], 'fr_R_surf_S_collat_transv_ant': df['fr_R_surf_S_collat_transv_ant_y3'], 'fr_R_surf_S_collat_transv_post': df['fr_R_surf_S_collat_transv_post_y3'], 'fr_R_surf_S_front_inf': df['fr_R_surf_S_front_inf_y3'], 'fr_R_surf_S_front_middle': df['fr_R_surf_S_front_middle_y3'], 'fr_R_surf_S_front_sup': df['fr_R_surf_S_front_sup_y3'], 'fr_R_surf_S_interm_prim_Jensen': df['fr_R_surf_S_interm_prim_Jensen_y3'], 'fr_R_surf_S_intraparietandP_trans': df['fr_R_surf_S_intraparietandP_trans_y3'], 'fr_R_surf_S_oc_middleandLunatus': df['fr_R_surf_S_oc_middleandLunatus_y3'], 'fr_R_surf_S_oc_temp_lat': df['fr_R_surf_S_oc_temp_lat_y3'], 'fr_R_surf_S_oc_temp_medandLingual': df['fr_R_surf_S_oc_temp_medandLingual_y3'], 'fr_R_surf_S_orbital_H_Shaped': df['fr_R_surf_S_orbital_H_Shaped_y3'], 'fr_R_surf_S_orbital_lateral': df['fr_R_surf_S_orbital_lateral_y3'], 'fr_R_surf_S_orbital_med_olfact': df['fr_R_surf_S_orbital_med_olfact_y3'], 'fr_R_surf_S_parieto_occipital': df['fr_R_surf_S_parieto_occipital_y3'], 'fr_R_surf_S_pericallosal': df['fr_R_surf_S_pericallosal_y3'], 'fr_R_surf_S_postcentral': df['fr_R_surf_S_postcentral_y3'], 'fr_R_surf_S_precentral_inf_part': df['fr_R_surf_S_precentral_inf_part_y3'], 'fr_R_surf_S_precentral_sup_part': df['fr_R_surf_S_precentral_sup_part_y3'], 'fr_R_surf_S_suborbital': df['fr_R_surf_S_suborbital_y3'], 'fr_R_surf_S_subparietal': df['fr_R_surf_S_subparietal_y3'], 'fr_R_surf_S_temporal_inf': df['fr_R_surf_S_temporal_inf_y3'], 'fr_R_surf_S_temporal_sup': df['fr_R_surf_S_temporal_sup_y3'], 'fr_R_surf_S_temporal_transverse': df['fr_R_surf_S_temporal_transverse_y3'], 'fr_L_surf_G_Ins_lgandS_cent_ins': df['fr_L_surf_G_Ins_lgandS_cent_ins_y3'], 'fr_L_surf_G_cingul_Post_ventral': df['fr_L_surf_G_cingul_Post_ventral_y3'], 'fr_L_surf_G_cuneus': df['fr_L_surf_G_cuneus_y3'], 'fr_L_surf_G_front_inf_Opercular': df['fr_L_surf_G_front_inf_Opercular_y3'], 'fr_L_surf_G_front_inf_Orbital': df['fr_L_surf_G_front_inf_Orbital_y3'], 'fr_L_surf_G_front_inf_Triangul': df['fr_L_surf_G_front_inf_Triangul_y3'], 'fr_L_surf_G_front_middle': df['fr_L_surf_G_front_middle_y3'], 'fr_L_surf_G_front_sup': df['fr_L_surf_G_front_sup_y3'], 'fr_L_surf_G_insular_short': df['fr_L_surf_G_insular_short_y3'], 'fr_L_surf_G_oc_temp_lat_fusifor': df['fr_L_surf_G_oc_temp_lat_fusifor_y3'], 'fr_L_surf_G_oc_temp_med_Lingual': df['fr_L_surf_G_oc_temp_med_Lingual_y3'], 'fr_L_surf_G_oc_temp_med_Parahip': df['fr_L_surf_G_oc_temp_med_Parahip_y3'], 'fr_L_surf_G_occipital_middle': df['fr_L_surf_G_occipital_middle_y3'], 'fr_L_surf_G_occipital_sup': df['fr_L_surf_G_occipital_sup_y3'], 'fr_L_surf_G_orbital': df['fr_L_surf_G_orbital_y3'], 'fr_L_surf_G_pariet_inf_Angular': df['fr_L_surf_G_pariet_inf_Angular_y3'], 'fr_L_surf_G_pariet_inf_Supramar': df['fr_L_surf_G_pariet_inf_Supramar_y3'], 'fr_L_surf_G_parietal_sup': df['fr_L_surf_G_parietal_sup_y3'], 'fr_L_surf_G_postcentral': df['fr_L_surf_G_postcentral_y3'], 'fr_L_surf_G_precentral': df['fr_L_surf_G_precentral_y3'], 'fr_L_surf_G_precuneus': df['fr_L_surf_G_precuneus_y3'], 'fr_L_surf_G_rectus': df['fr_L_surf_G_rectus_y3'], 'fr_L_surf_G_temp_sup_G_T_transv': df['fr_L_surf_G_temp_sup_G_T_transv_y3'], 'fr_L_surf_G_temp_sup_Lateral': df['fr_L_surf_G_temp_sup_Lateral_y3'], 'fr_L_surf_G_temp_sup_Plan_polar': df['fr_L_surf_G_temp_sup_Plan_polar_y3'], 'fr_L_surf_G_temp_sup_Plan_tempo': df['fr_L_surf_G_temp_sup_Plan_tempo_y3'], 'fr_L_surf_G_temporal_inf': df['fr_L_surf_G_temporal_inf_y3'], 'fr_L_surf_G_temporal_middle': df['fr_L_surf_G_temporal_middle_y3'], 'fr_L_surf_GandS_cingul_Ant': df['fr_L_surf_GandS_cingul_Ant_y3'], 'fr_L_surf_GandS_cingul_Mid_Post': df['fr_L_surf_GandS_cingul_Mid_Post_y3'], 'fr_L_surf_GandS_frontomargin': df['fr_L_surf_GandS_frontomargin_y3'], 'fr_L_surf_GandS_occipital_inf': df['fr_L_surf_GandS_occipital_inf_y3'], 'fr_L_surf_GandS_paracentral': df['fr_L_surf_GandS_paracentral_y3'], 'fr_L_surf_GandS_subcentral': df['fr_L_surf_GandS_subcentral_y3'], 'fr_L_surf_GandS_transv_frontopol': df['fr_L_surf_GandS_transv_frontopol_y3'], 'fr_L_surf_Lat_Fis_ant_Horizont': df['fr_L_surf_Lat_Fis_ant_Horizont_y3'], 'fr_L_surf_Lat_Fis_post': df['fr_L_surf_Lat_Fis_post_y3'], 'fr_L_surf_Pole_occipital': df['fr_L_surf_Pole_occipital_y3'], 'fr_L_surf_Pole_temporal': df['fr_L_surf_Pole_temporal_y3'], 'fr_L_surf_S_calcarine': df['fr_L_surf_S_calcarine_y3'], 'fr_L_surf_S_central': df['fr_L_surf_S_central_y3'], 'fr_L_surf_S_cingul_Marginalis': df['fr_L_surf_S_cingul_Marginalis_y3'], 'fr_L_surf_S_circular_insula_ant': df['fr_L_surf_S_circular_insula_ant_y3'], 'fr_L_surf_S_circular_insula_inf': df['fr_L_surf_S_circular_insula_inf_y3'], 'fr_L_surf_S_circular_insula_sup': df['fr_L_surf_S_circular_insula_sup_y3'], 'fr_L_surf_S_collat_transv_ant': df['fr_L_surf_S_collat_transv_ant_y3'], 'fr_L_surf_S_collat_transv_post': df['fr_L_surf_S_collat_transv_post_y3'], 'fr_L_surf_S_front_inf': df['fr_L_surf_S_front_inf_y3'], 'fr_L_surf_S_front_middle': df['fr_L_surf_S_front_middle_y3'], 'fr_L_surf_S_front_sup': df['fr_L_surf_S_front_sup_y3'],  'fr_L_surf_S_intraparietandP_trans': df['fr_L_surf_S_intraparietandP_trans_y3'], 'fr_L_surf_S_oc_middleandLunatus': df['fr_L_surf_S_oc_middleandLunatus_y3'], 'fr_L_surf_S_oc_temp_lat': df['fr_L_surf_S_oc_temp_lat_y3'], 'fr_L_surf_S_oc_temp_medandLingual': df['fr_L_surf_S_oc_temp_medandLingual_y3'], 'fr_L_surf_S_orbital_H_Shaped': df['fr_L_surf_S_orbital_H_Shaped_y3'], 'fr_L_surf_S_orbital_lateral': df['fr_L_surf_S_orbital_lateral_y3'], 'fr_L_surf_S_orbital_med_olfact': df['fr_L_surf_S_orbital_med_olfact_y3'], 'fr_L_surf_S_parieto_occipital': df['fr_L_surf_S_parieto_occipital_y3'], 'fr_L_surf_S_postcentral': df['fr_L_surf_S_postcentral_y3'], 'fr_L_surf_S_precentral_inf_part': df['fr_L_surf_S_precentral_inf_part_y3'], 'fr_L_surf_S_precentral_sup_part': df['fr_L_surf_S_precentral_sup_part_y3'], 'fr_L_surf_S_suborbital': df['fr_L_surf_S_suborbital_y3'], 'fr_L_surf_S_subparietal': df['fr_L_surf_S_subparietal_y3'], 'fr_L_surf_S_temporal_inf': df['fr_L_surf_S_temporal_inf_y3'], 'fr_L_surf_S_temporal_sup': df['fr_L_surf_S_temporal_sup_y3']}
		#frame_free4 = {'handlat': df['lat_manual'],'apoe': df['apoe'], 'y_last':df['ultimavisita'], 'dx_last': df['ultimodx'], 'dx_visita': df['dx_corto_visita4'], 'age_lastdx': df['edad_ultimodx'], 'age': df['edad_visita4'], 'sex': df['sexo'], 'free_R_Thal': df['fr_Right_Thalamus_Proper_y4'], 'free_L_Thal': df['fr_Left_Thalamus_Proper_y4'],'free_R_Puta': df['fr_Right_Putamen_y4'], 'free_L_Puta': df['fr_Left_Putamen_y4'],'free_R_Amyg': df['fr_Right_Amygdala_y4'], 'free_L_Amyg': df['fr_Left_Amygdala_y4'],'free_R_Pall': df['fr_Right_Pallidum_y4'], 'free_L_Pall': df['fr_Left_Pallidum_y4'],'free_R_Caud': df['fr_Right_Caudate_y4'], 'free_L_Caud': df['fr_Left_Caudate_y4'],'free_R_Hipp': df['fr_Right_Hippocampus_y4'], 'free_L_Hipp': df['fr_Left_Hippocampus_y4'],'free_R_Accu': df['fr_Right_Accumbens_area_y4'], 'free_L_Accu': df['fr_Left_Accumbens_area_y4'], 'fr_BrainSegVol': df['fr_BrainSegVol_y4'], 'fr_BrainSegVolNotVent':df['fr_BrainSegVolNotVent_y4'], 'fr_BrainSegVol_to_eTIV': df['fr_BrainSegVol_to_eTIV_y4']}
		frame_free4 = {'education': df['nivel_educativo'],'handlat': df['lat_manual'],'apoe': df['apoe'], 'nvisita':df['nvisita4'], 'y_last':df['ultimavisita'], 'dx_last': df['ultimodx'], 'dx_visita': df['dx_corto_visita4'], 'age_lastdx': df['edad_ultimodx'], 'age': df['edad_visita4'], 'sex': df['sexo'], 'free_R_Thal': df['fr_Right_Thalamus_Proper_y4'], 'free_L_Thal': df['fr_Left_Thalamus_Proper_y4'],'free_R_Puta': df['fr_Right_Putamen_y4'], 'free_L_Puta': df['fr_Left_Putamen_y4'],'free_R_Amyg': df['fr_Right_Amygdala_y4'], 'free_L_Amyg': df['fr_Left_Amygdala_y4'],'free_R_Pall': df['fr_Right_Pallidum_y4'], 'free_L_Pall': df['fr_Left_Pallidum_y4'],'free_R_Caud': df['fr_Right_Caudate_y4'], 'free_L_Caud': df['fr_Left_Caudate_y4'],'free_R_Hipp': df['fr_Right_Hippocampus_y4'], 'free_L_Hipp': df['fr_Left_Hippocampus_y4'],'free_R_Accu': df['fr_Right_Accumbens_area_y4'], 'free_L_Accu': df['fr_Left_Accumbens_area_y4'], 'fr_BrainSegVol': df['fr_BrainSegVol_y4'], 'fr_BrainSegVolNotVent':df['fr_BrainSegVolNotVent_y4'], 'fr_BrainSegVol_to_eTIV': df['fr_BrainSegVol_to_eTIV_y4'],   'fr_R_surf_G_Ins_lgandS_cent_ins': df['fr_R_surf_G_Ins_lgandS_cent_ins_y4'], 'fr_R_surf_G_cingul_Post_ventral': df['fr_R_surf_G_cingul_Post_ventral_y4'], 'fr_R_surf_G_cuneus': df['fr_R_surf_G_cuneus_y4'], 'fr_R_surf_G_front_inf_Opercular': df['fr_R_surf_G_front_inf_Opercular_y4'], 'fr_R_surf_G_front_inf_Orbital': df['fr_R_surf_G_front_inf_Orbital_y4'], 'fr_R_surf_G_front_inf_Triangul': df['fr_R_surf_G_front_inf_Triangul_y4'], 'fr_R_surf_G_front_middle': df['fr_R_surf_G_front_middle_y4'], 'fr_R_surf_G_front_sup': df['fr_R_surf_G_front_sup_y4'], 'fr_R_surf_G_insular_short': df['fr_R_surf_G_insular_short_y4'], 'fr_R_surf_G_oc_temp_lat_fusifor': df['fr_R_surf_G_oc_temp_lat_fusifor_y4'], 'fr_R_surf_G_oc_temp_med_Lingual': df['fr_R_surf_G_oc_temp_med_Lingual_y4'], 'fr_R_surf_G_oc_temp_med_Parahip': df['fr_R_surf_G_oc_temp_med_Parahip_y4'], 'fr_R_surf_G_occipital_middle': df['fr_R_surf_G_occipital_middle_y4'], 'fr_R_surf_G_occipital_sup': df['fr_R_surf_G_occipital_sup_y4'], 'fr_R_surf_G_orbital': df['fr_R_surf_G_orbital_y4'], 'fr_R_surf_G_pariet_inf_Angular': df['fr_R_surf_G_pariet_inf_Angular_y4'], 'fr_R_surf_G_pariet_inf_Supramar': df['fr_R_surf_G_pariet_inf_Supramar_y4'], 'fr_R_surf_G_parietal_sup': df['fr_R_surf_G_parietal_sup_y4'], 'fr_R_surf_G_postcentral': df['fr_R_surf_G_postcentral_y4'], 'fr_R_surf_G_precentral': df['fr_R_surf_G_precentral_y4'], 'fr_R_surf_G_precuneus': df['fr_R_surf_G_precuneus_y4'], 'fr_R_surf_G_rectus': df['fr_R_surf_G_rectus_y4'], 'fr_R_surf_G_temp_sup_G_T_transv': df['fr_R_surf_G_temp_sup_G_T_transv_y4'], 'fr_R_surf_G_temp_sup_Lateral': df['fr_R_surf_G_temp_sup_Lateral_y4'], 'fr_R_surf_G_temp_sup_Plan_polar': df['fr_R_surf_G_temp_sup_Plan_polar_y4'], 'fr_R_surf_G_temp_sup_Plan_tempo': df['fr_R_surf_G_temp_sup_Plan_tempo_y4'], 'fr_R_surf_G_temporal_inf': df['fr_R_surf_G_temporal_inf_y4'], 'fr_R_surf_G_temporal_middle': df['fr_R_surf_G_temporal_middle_y4'], 'fr_R_surf_GandS_cingul_Ant': df['fr_R_surf_GandS_cingul_Ant_y4'], 'fr_R_surf_GandS_cingul_Mid_Post': df['fr_R_surf_GandS_cingul_Mid_Post_y4'], 'fr_R_surf_GandS_frontomargin': df['fr_R_surf_GandS_frontomargin_y4'], 'fr_R_surf_GandS_occipital_inf': df['fr_R_surf_GandS_occipital_inf_y4'], 'fr_R_surf_GandS_paracentral': df['fr_R_surf_GandS_paracentral_y4'], 'fr_R_surf_GandS_subcentral': df['fr_R_surf_GandS_subcentral_y4'], 'fr_R_surf_GandS_transv_frontopol': df['fr_R_surf_GandS_transv_frontopol_y4'], 'fr_R_surf_Lat_Fis_ant_Horizont': df['fr_R_surf_Lat_Fis_ant_Horizont_y4'], 'fr_R_surf_Lat_Fis_post': df['fr_R_surf_Lat_Fis_post_y4'], 'fr_R_surf_Pole_occipital': df['fr_R_surf_Pole_occipital_y4'], 'fr_R_surf_Pole_temporal': df['fr_R_surf_Pole_temporal_y4'], 'fr_R_surf_S_calcarine': df['fr_R_surf_S_calcarine_y4'], 'fr_R_surf_S_central': df['fr_R_surf_S_central_y4'], 'fr_R_surf_S_cingul_Marginalis': df['fr_R_surf_S_cingul_Marginalis_y4'], 'fr_R_surf_S_circular_insula_ant': df['fr_R_surf_S_circular_insula_ant_y4'], 'fr_R_surf_S_circular_insula_inf': df['fr_R_surf_S_circular_insula_inf_y4'], 'fr_R_surf_S_circular_insula_sup': df['fr_R_surf_S_circular_insula_sup_y4'], 'fr_R_surf_S_collat_transv_ant': df['fr_R_surf_S_collat_transv_ant_y4'], 'fr_R_surf_S_collat_transv_post': df['fr_R_surf_S_collat_transv_post_y4'], 'fr_R_surf_S_front_inf': df['fr_R_surf_S_front_inf_y4'], 'fr_R_surf_S_front_middle': df['fr_R_surf_S_front_middle_y4'], 'fr_R_surf_S_front_sup': df['fr_R_surf_S_front_sup_y4'], 'fr_R_surf_S_interm_prim_Jensen': df['fr_R_surf_S_interm_prim_Jensen_y4'], 'fr_R_surf_S_intraparietandP_trans': df['fr_R_surf_S_intraparietandP_trans_y4'], 'fr_R_surf_S_oc_middleandLunatus': df['fr_R_surf_S_oc_middleandLunatus_y4'], 'fr_R_surf_S_oc_temp_lat': df['fr_R_surf_S_oc_temp_lat_y4'], 'fr_R_surf_S_oc_temp_medandLingual': df['fr_R_surf_S_oc_temp_medandLingual_y4'], 'fr_R_surf_S_orbital_H_Shaped': df['fr_R_surf_S_orbital_H_Shaped_y4'], 'fr_R_surf_S_orbital_lateral': df['fr_R_surf_S_orbital_lateral_y4'], 'fr_R_surf_S_orbital_med_olfact': df['fr_R_surf_S_orbital_med_olfact_y4'], 'fr_R_surf_S_parieto_occipital': df['fr_R_surf_S_parieto_occipital_y4'], 'fr_R_surf_S_pericallosal': df['fr_R_surf_S_pericallosal_y4'], 'fr_R_surf_S_postcentral': df['fr_R_surf_S_postcentral_y4'], 'fr_R_surf_S_precentral_inf_part': df['fr_R_surf_S_precentral_inf_part_y4'], 'fr_R_surf_S_precentral_sup_part': df['fr_R_surf_S_precentral_sup_part_y4'], 'fr_R_surf_S_suborbital': df['fr_R_surf_S_suborbital_y4'], 'fr_R_surf_S_subparietal': df['fr_R_surf_S_subparietal_y4'], 'fr_R_surf_S_temporal_inf': df['fr_R_surf_S_temporal_inf_y4'], 'fr_R_surf_S_temporal_sup': df['fr_R_surf_S_temporal_sup_y4'], 'fr_R_surf_S_temporal_transverse': df['fr_R_surf_S_temporal_transverse_y4'],'fr_L_surf_G_Ins_lgandS_cent_ins': df['fr_L_surf_G_Ins_lgandS_cent_ins_y4'], 'fr_L_surf_G_cingul_Post_ventral': df['fr_L_surf_G_cingul_Post_ventral_y4'], 'fr_L_surf_G_cuneus': df['fr_L_surf_G_cuneus_y4'], 'fr_L_surf_G_front_inf_Opercular': df['fr_L_surf_G_front_inf_Opercular_y4'], 'fr_L_surf_G_front_inf_Orbital': df['fr_L_surf_G_front_inf_Orbital_y4'], 'fr_L_surf_G_front_inf_Triangul': df['fr_L_surf_G_front_inf_Triangul_y4'], 'fr_L_surf_G_front_middle': df['fr_L_surf_G_front_middle_y4'], 'fr_L_surf_G_front_sup': df['fr_L_surf_G_front_sup_y4'], 'fr_L_surf_G_insular_short': df['fr_L_surf_G_insular_short_y4'], 'fr_L_surf_G_oc_temp_lat_fusifor': df['fr_L_surf_G_oc_temp_lat_fusifor_y4'], 'fr_L_surf_G_oc_temp_med_Lingual': df['fr_L_surf_G_oc_temp_med_Lingual_y4'], 'fr_L_surf_G_oc_temp_med_Parahip': df['fr_L_surf_G_oc_temp_med_Parahip_y4'], 'fr_L_surf_G_occipital_middle': df['fr_L_surf_G_occipital_middle_y4'], 'fr_L_surf_G_occipital_sup': df['fr_L_surf_G_occipital_sup_y4'], 'fr_L_surf_G_orbital': df['fr_L_surf_G_orbital_y4'], 'fr_L_surf_G_pariet_inf_Angular': df['fr_L_surf_G_pariet_inf_Angular_y4'], 'fr_L_surf_G_pariet_inf_Supramar': df['fr_L_surf_G_pariet_inf_Supramar_y4'], 'fr_L_surf_G_parietal_sup': df['fr_L_surf_G_parietal_sup_y4'], 'fr_L_surf_G_postcentral': df['fr_L_surf_G_postcentral_y4'], 'fr_L_surf_G_precentral': df['fr_L_surf_G_precentral_y4'], 'fr_L_surf_G_precuneus': df['fr_L_surf_G_precuneus_y4'], 'fr_L_surf_G_rectus': df['fr_L_surf_G_rectus_y4'], 'fr_L_surf_G_temp_sup_G_T_transv': df['fr_L_surf_G_temp_sup_G_T_transv_y4'], 'fr_L_surf_G_temp_sup_Lateral': df['fr_L_surf_G_temp_sup_Lateral_y4'], 'fr_L_surf_G_temp_sup_Plan_polar': df['fr_L_surf_G_temp_sup_Plan_polar_y4'], 'fr_L_surf_G_temp_sup_Plan_tempo': df['fr_L_surf_G_temp_sup_Plan_tempo_y4'], 'fr_L_surf_G_temporal_inf': df['fr_L_surf_G_temporal_inf_y4'], 'fr_L_surf_G_temporal_middle': df['fr_L_surf_G_temporal_middle_y4'], 'fr_L_surf_GandS_cingul_Ant': df['fr_L_surf_GandS_cingul_Ant_y4'], 'fr_L_surf_GandS_cingul_Mid_Post': df['fr_L_surf_GandS_cingul_Mid_Post_y4'], 'fr_L_surf_GandS_frontomargin': df['fr_L_surf_GandS_frontomargin_y4'], 'fr_L_surf_GandS_occipital_inf': df['fr_L_surf_GandS_occipital_inf_y4'], 'fr_L_surf_GandS_paracentral': df['fr_L_surf_GandS_paracentral_y4'], 'fr_L_surf_GandS_subcentral': df['fr_L_surf_GandS_subcentral_y4'], 'fr_L_surf_GandS_transv_frontopol': df['fr_L_surf_GandS_transv_frontopol_y4'], 'fr_L_surf_Lat_Fis_ant_Horizont': df['fr_L_surf_Lat_Fis_ant_Horizont_y4'], 'fr_L_surf_Lat_Fis_post': df['fr_L_surf_Lat_Fis_post_y4'], 'fr_L_surf_Pole_occipital': df['fr_L_surf_Pole_occipital_y4'], 'fr_L_surf_Pole_temporal': df['fr_L_surf_Pole_temporal_y4'], 'fr_L_surf_S_calcarine': df['fr_L_surf_S_calcarine_y4'], 'fr_L_surf_S_central': df['fr_L_surf_S_central_y4'], 'fr_L_surf_S_cingul_Marginalis': df['fr_L_surf_S_cingul_Marginalis_y4'], 'fr_L_surf_S_circular_insula_ant': df['fr_L_surf_S_circular_insula_ant_y4'], 'fr_L_surf_S_circular_insula_inf': df['fr_L_surf_S_circular_insula_inf_y4'], 'fr_L_surf_S_circular_insula_sup': df['fr_L_surf_S_circular_insula_sup_y4'], 'fr_L_surf_S_collat_transv_ant': df['fr_L_surf_S_collat_transv_ant_y4'], 'fr_L_surf_S_collat_transv_post': df['fr_L_surf_S_collat_transv_post_y4'], 'fr_L_surf_S_front_inf': df['fr_L_surf_S_front_inf_y4'], 'fr_L_surf_S_front_middle': df['fr_L_surf_S_front_middle_y4'], 'fr_L_surf_S_front_sup': df['fr_L_surf_S_front_sup_y4'],  'fr_L_surf_S_intraparietandP_trans': df['fr_L_surf_S_intraparietandP_trans_y4'], 'fr_L_surf_S_oc_middleandLunatus': df['fr_L_surf_S_oc_middleandLunatus_y4'], 'fr_L_surf_S_oc_temp_lat': df['fr_L_surf_S_oc_temp_lat_y4'], 'fr_L_surf_S_oc_temp_medandLingual': df['fr_L_surf_S_oc_temp_medandLingual_y4'], 'fr_L_surf_S_orbital_H_Shaped': df['fr_L_surf_S_orbital_H_Shaped_y4'], 'fr_L_surf_S_orbital_lateral': df['fr_L_surf_S_orbital_lateral_y4'], 'fr_L_surf_S_orbital_med_olfact': df['fr_L_surf_S_orbital_med_olfact_y4'], 'fr_L_surf_S_parieto_occipital': df['fr_L_surf_S_parieto_occipital_y4'], 'fr_L_surf_S_postcentral': df['fr_L_surf_S_postcentral_y4'], 'fr_L_surf_S_precentral_inf_part': df['fr_L_surf_S_precentral_inf_part_y4'], 'fr_L_surf_S_precentral_sup_part': df['fr_L_surf_S_precentral_sup_part_y4'], 'fr_L_surf_S_suborbital': df['fr_L_surf_S_suborbital_y4'], 'fr_L_surf_S_subparietal': df['fr_L_surf_S_subparietal_y4'], 'fr_L_surf_S_temporal_inf': df['fr_L_surf_S_temporal_inf_y4'], 'fr_L_surf_S_temporal_sup': df['fr_L_surf_S_temporal_sup_y4']}
		#frame_free5 = {'handlat': df['lat_manual'],'apoe': df['apoe'], 'y_last':df['ultimavisita'], 'dx_last': df['ultimodx'], 'dx_visita': df['dx_corto_visita5'], 'age_lastdx': df['edad_ultimodx'], 'age': df['edad_visita5'], 'sex': df['sexo'], 'free_R_Thal': df['fr_Right_Thalamus_Proper_y5'], 'free_L_Thal': df['fr_Left_Thalamus_Proper_y5'],'free_R_Puta': df['fr_Right_Putamen_y5'], 'free_L_Puta': df['fr_Left_Putamen_y5'],'free_R_Amyg': df['fr_Right_Amygdala_y5'], 'free_L_Amyg': df['fr_Left_Amygdala_y5'],'free_R_Pall': df['fr_Right_Pallidum_y5'], 'free_L_Pall': df['fr_Left_Pallidum_y5'],'free_R_Caud': df['fr_Right_Caudate_y5'], 'free_L_Caud': df['fr_Left_Caudate_y5'],'free_R_Hipp': df['fr_Right_Hippocampus_y5'], 'free_L_Hipp': df['fr_Left_Hippocampus_y5'],'free_R_Accu': df['fr_Right_Accumbens_area_y5'], 'free_L_Accu': df['fr_Left_Accumbens_area_y5'], 'fr_BrainSegVol': df['fr_BrainSegVol_y5'], 'fr_BrainSegVolNotVent':df['fr_BrainSegVolNotVent_y5'], 'fr_BrainSegVol_to_eTIV': df['fr_BrainSegVol_to_eTIV_y5']}
		frame_free5 = {'education': df['nivel_educativo'],'handlat': df['lat_manual'],'apoe': df['apoe'], 'nvisita':df['nvisita5'] ,'y_last':df['ultimavisita'], 'dx_last': df['ultimodx'], 'dx_visita': df['dx_corto_visita5'], 'age_lastdx': df['edad_ultimodx'], 'age': df['edad_visita5'], 'sex': df['sexo'], 'free_R_Thal': df['fr_Right_Thalamus_Proper_y5'], 'free_L_Thal': df['fr_Left_Thalamus_Proper_y5'],'free_R_Puta': df['fr_Right_Putamen_y5'], 'free_L_Puta': df['fr_Left_Putamen_y5'],'free_R_Amyg': df['fr_Right_Amygdala_y5'], 'free_L_Amyg': df['fr_Left_Amygdala_y5'],'free_R_Pall': df['fr_Right_Pallidum_y5'], 'free_L_Pall': df['fr_Left_Pallidum_y5'],'free_R_Caud': df['fr_Right_Caudate_y5'], 'free_L_Caud': df['fr_Left_Caudate_y5'],'free_R_Hipp': df['fr_Right_Hippocampus_y5'], 'free_L_Hipp': df['fr_Left_Hippocampus_y5'],'free_R_Accu': df['fr_Right_Accumbens_area_y5'], 'free_L_Accu': df['fr_Left_Accumbens_area_y5'], 'fr_BrainSegVol': df['fr_BrainSegVol_y5'], 'fr_BrainSegVolNotVent':df['fr_BrainSegVolNotVent_y5'], 'fr_BrainSegVol_to_eTIV': df['fr_BrainSegVol_to_eTIV_y5'],  'fr_R_surf_G_Ins_lgandS_cent_ins': df['fr_R_surf_G_Ins_lgandS_cent_ins_y5'], 'fr_R_surf_G_cingul_Post_ventral': df['fr_R_surf_G_cingul_Post_ventral_y5'], 'fr_R_surf_G_cuneus': df['fr_R_surf_G_cuneus_y5'], 'fr_R_surf_G_front_inf_Opercular': df['fr_R_surf_G_front_inf_Opercular_y5'], 'fr_R_surf_G_front_inf_Orbital': df['fr_R_surf_G_front_inf_Orbital_y5'], 'fr_R_surf_G_front_inf_Triangul': df['fr_R_surf_G_front_inf_Triangul_y5'], 'fr_R_surf_G_front_middle': df['fr_R_surf_G_front_middle_y5'], 'fr_R_surf_G_front_sup': df['fr_R_surf_G_front_sup_y5'], 'fr_R_surf_G_insular_short': df['fr_R_surf_G_insular_short_y5'], 'fr_R_surf_G_oc_temp_lat_fusifor': df['fr_R_surf_G_oc_temp_lat_fusifor_y5'], 'fr_R_surf_G_oc_temp_med_Lingual': df['fr_R_surf_G_oc_temp_med_Lingual_y5'], 'fr_R_surf_G_oc_temp_med_Parahip': df['fr_R_surf_G_oc_temp_med_Parahip_y5'], 'fr_R_surf_G_occipital_middle': df['fr_R_surf_G_occipital_middle_y5'], 'fr_R_surf_G_occipital_sup': df['fr_R_surf_G_occipital_sup_y5'], 'fr_R_surf_G_orbital': df['fr_R_surf_G_orbital_y5'], 'fr_R_surf_G_pariet_inf_Angular': df['fr_R_surf_G_pariet_inf_Angular_y5'], 'fr_R_surf_G_pariet_inf_Supramar': df['fr_R_surf_G_pariet_inf_Supramar_y5'], 'fr_R_surf_G_parietal_sup': df['fr_R_surf_G_parietal_sup_y5'], 'fr_R_surf_G_postcentral': df['fr_R_surf_G_postcentral_y5'], 'fr_R_surf_G_precentral': df['fr_R_surf_G_precentral_y5'], 'fr_R_surf_G_precuneus': df['fr_R_surf_G_precuneus_y5'], 'fr_R_surf_G_rectus': df['fr_R_surf_G_rectus_y5'], 'fr_R_surf_G_temp_sup_G_T_transv': df['fr_R_surf_G_temp_sup_G_T_transv_y5'], 'fr_R_surf_G_temp_sup_Lateral': df['fr_R_surf_G_temp_sup_Lateral_y5'], 'fr_R_surf_G_temp_sup_Plan_polar': df['fr_R_surf_G_temp_sup_Plan_polar_y5'], 'fr_R_surf_G_temp_sup_Plan_tempo': df['fr_R_surf_G_temp_sup_Plan_tempo_y5'], 'fr_R_surf_G_temporal_inf': df['fr_R_surf_G_temporal_inf_y5'], 'fr_R_surf_G_temporal_middle': df['fr_R_surf_G_temporal_middle_y5'], 'fr_R_surf_GandS_cingul_Ant': df['fr_R_surf_GandS_cingul_Ant_y5'], 'fr_R_surf_GandS_cingul_Mid_Post': df['fr_R_surf_GandS_cingul_Mid_Post_y5'], 'fr_R_surf_GandS_frontomargin': df['fr_R_surf_GandS_frontomargin_y5'], 'fr_R_surf_GandS_occipital_inf': df['fr_R_surf_GandS_occipital_inf_y5'], 'fr_R_surf_GandS_paracentral': df['fr_R_surf_GandS_paracentral_y5'], 'fr_R_surf_GandS_subcentral': df['fr_R_surf_GandS_subcentral_y5'], 'fr_R_surf_GandS_transv_frontopol': df['fr_R_surf_GandS_transv_frontopol_y5'], 'fr_R_surf_Lat_Fis_ant_Horizont': df['fr_R_surf_Lat_Fis_ant_Horizont_y5'], 'fr_R_surf_Lat_Fis_post': df['fr_R_surf_Lat_Fis_post_y5'], 'fr_R_surf_Pole_occipital': df['fr_R_surf_Pole_occipital_y5'], 'fr_R_surf_Pole_temporal': df['fr_R_surf_Pole_temporal_y5'], 'fr_R_surf_S_calcarine': df['fr_R_surf_S_calcarine_y5'], 'fr_R_surf_S_central': df['fr_R_surf_S_central_y5'], 'fr_R_surf_S_cingul_Marginalis': df['fr_R_surf_S_cingul_Marginalis_y5'], 'fr_R_surf_S_circular_insula_ant': df['fr_R_surf_S_circular_insula_ant_y5'], 'fr_R_surf_S_circular_insula_inf': df['fr_R_surf_S_circular_insula_inf_y5'], 'fr_R_surf_S_circular_insula_sup': df['fr_R_surf_S_circular_insula_sup_y5'], 'fr_R_surf_S_collat_transv_ant': df['fr_R_surf_S_collat_transv_ant_y5'], 'fr_R_surf_S_collat_transv_post': df['fr_R_surf_S_collat_transv_post_y5'], 'fr_R_surf_S_front_inf': df['fr_R_surf_S_front_inf_y5'], 'fr_R_surf_S_front_middle': df['fr_R_surf_S_front_middle_y5'], 'fr_R_surf_S_front_sup': df['fr_R_surf_S_front_sup_y5'], 'fr_R_surf_S_interm_prim_Jensen': df['fr_R_surf_S_interm_prim_Jensen_y5'], 'fr_R_surf_S_intraparietandP_trans': df['fr_R_surf_S_intraparietandP_trans_y5'], 'fr_R_surf_S_oc_middleandLunatus': df['fr_R_surf_S_oc_middleandLunatus_y5'], 'fr_R_surf_S_oc_temp_lat': df['fr_R_surf_S_oc_temp_lat_y5'], 'fr_R_surf_S_oc_temp_medandLingual': df['fr_R_surf_S_oc_temp_medandLingual_y5'], 'fr_R_surf_S_orbital_H_Shaped': df['fr_R_surf_S_orbital_H_Shaped_y5'], 'fr_R_surf_S_orbital_lateral': df['fr_R_surf_S_orbital_lateral_y5'], 'fr_R_surf_S_orbital_med_olfact': df['fr_R_surf_S_orbital_med_olfact_y5'], 'fr_R_surf_S_parieto_occipital': df['fr_R_surf_S_parieto_occipital_y5'], 'fr_R_surf_S_pericallosal': df['fr_R_surf_S_pericallosal_y5'], 'fr_R_surf_S_postcentral': df['fr_R_surf_S_postcentral_y5'], 'fr_R_surf_S_precentral_inf_part': df['fr_R_surf_S_precentral_inf_part_y5'], 'fr_R_surf_S_precentral_sup_part': df['fr_R_surf_S_precentral_sup_part_y5'], 'fr_R_surf_S_suborbital': df['fr_R_surf_S_suborbital_y5'], 'fr_R_surf_S_subparietal': df['fr_R_surf_S_subparietal_y5'], 'fr_R_surf_S_temporal_inf': df['fr_R_surf_S_temporal_inf_y5'], 'fr_R_surf_S_temporal_sup': df['fr_R_surf_S_temporal_sup_y5'], 'fr_R_surf_S_temporal_transverse': df['fr_R_surf_S_temporal_transverse_y5'],'fr_L_surf_G_Ins_lgandS_cent_ins': df['fr_L_surf_G_Ins_lgandS_cent_ins_y5'], 'fr_L_surf_G_cingul_Post_ventral': df['fr_L_surf_G_cingul_Post_ventral_y5'], 'fr_L_surf_G_cuneus': df['fr_L_surf_G_cuneus_y5'], 'fr_L_surf_G_front_inf_Opercular': df['fr_L_surf_G_front_inf_Opercular_y5'], 'fr_L_surf_G_front_inf_Orbital': df['fr_L_surf_G_front_inf_Orbital_y5'], 'fr_L_surf_G_front_inf_Triangul': df['fr_L_surf_G_front_inf_Triangul_y5'], 'fr_L_surf_G_front_middle': df['fr_L_surf_G_front_middle_y5'], 'fr_L_surf_G_front_sup': df['fr_L_surf_G_front_sup_y5'], 'fr_L_surf_G_insular_short': df['fr_L_surf_G_insular_short_y5'], 'fr_L_surf_G_oc_temp_lat_fusifor': df['fr_L_surf_G_oc_temp_lat_fusifor_y5'], 'fr_L_surf_G_oc_temp_med_Lingual': df['fr_L_surf_G_oc_temp_med_Lingual_y5'], 'fr_L_surf_G_oc_temp_med_Parahip': df['fr_L_surf_G_oc_temp_med_Parahip_y5'], 'fr_L_surf_G_occipital_middle': df['fr_L_surf_G_occipital_middle_y5'], 'fr_L_surf_G_occipital_sup': df['fr_L_surf_G_occipital_sup_y5'], 'fr_L_surf_G_orbital': df['fr_L_surf_G_orbital_y5'], 'fr_L_surf_G_pariet_inf_Angular': df['fr_L_surf_G_pariet_inf_Angular_y5'], 'fr_L_surf_G_pariet_inf_Supramar': df['fr_L_surf_G_pariet_inf_Supramar_y5'], 'fr_L_surf_G_parietal_sup': df['fr_L_surf_G_parietal_sup_y5'], 'fr_L_surf_G_postcentral': df['fr_L_surf_G_postcentral_y5'], 'fr_L_surf_G_precentral': df['fr_L_surf_G_precentral_y5'], 'fr_L_surf_G_precuneus': df['fr_L_surf_G_precuneus_y5'], 'fr_L_surf_G_rectus': df['fr_L_surf_G_rectus_y5'], 'fr_L_surf_G_temp_sup_G_T_transv': df['fr_L_surf_G_temp_sup_G_T_transv_y5'], 'fr_L_surf_G_temp_sup_Lateral': df['fr_L_surf_G_temp_sup_Lateral_y5'], 'fr_L_surf_G_temp_sup_Plan_polar': df['fr_L_surf_G_temp_sup_Plan_polar_y5'], 'fr_L_surf_G_temp_sup_Plan_tempo': df['fr_L_surf_G_temp_sup_Plan_tempo_y5'], 'fr_L_surf_G_temporal_inf': df['fr_L_surf_G_temporal_inf_y5'], 'fr_L_surf_G_temporal_middle': df['fr_L_surf_G_temporal_middle_y5'], 'fr_L_surf_GandS_cingul_Ant': df['fr_L_surf_GandS_cingul_Ant_y5'], 'fr_L_surf_GandS_cingul_Mid_Post': df['fr_L_surf_GandS_cingul_Mid_Post_y5'], 'fr_L_surf_GandS_frontomargin': df['fr_L_surf_GandS_frontomargin_y5'], 'fr_L_surf_GandS_occipital_inf': df['fr_L_surf_GandS_occipital_inf_y5'], 'fr_L_surf_GandS_paracentral': df['fr_L_surf_GandS_paracentral_y5'], 'fr_L_surf_GandS_subcentral': df['fr_L_surf_GandS_subcentral_y5'], 'fr_L_surf_GandS_transv_frontopol': df['fr_L_surf_GandS_transv_frontopol_y5'], 'fr_L_surf_Lat_Fis_ant_Horizont': df['fr_L_surf_Lat_Fis_ant_Horizont_y5'], 'fr_L_surf_Lat_Fis_post': df['fr_L_surf_Lat_Fis_post_y5'], 'fr_L_surf_Pole_occipital': df['fr_L_surf_Pole_occipital_y5'], 'fr_L_surf_Pole_temporal': df['fr_L_surf_Pole_temporal_y5'], 'fr_L_surf_S_calcarine': df['fr_L_surf_S_calcarine_y5'], 'fr_L_surf_S_central': df['fr_L_surf_S_central_y5'], 'fr_L_surf_S_cingul_Marginalis': df['fr_L_surf_S_cingul_Marginalis_y5'], 'fr_L_surf_S_circular_insula_ant': df['fr_L_surf_S_circular_insula_ant_y5'], 'fr_L_surf_S_circular_insula_inf': df['fr_L_surf_S_circular_insula_inf_y5'], 'fr_L_surf_S_circular_insula_sup': df['fr_L_surf_S_circular_insula_sup_y5'], 'fr_L_surf_S_collat_transv_ant': df['fr_L_surf_S_collat_transv_ant_y5'], 'fr_L_surf_S_collat_transv_post': df['fr_L_surf_S_collat_transv_post_y5'], 'fr_L_surf_S_front_inf': df['fr_L_surf_S_front_inf_y5'], 'fr_L_surf_S_front_middle': df['fr_L_surf_S_front_middle_y5'], 'fr_L_surf_S_front_sup': df['fr_L_surf_S_front_sup_y5'],  'fr_L_surf_S_intraparietandP_trans': df['fr_L_surf_S_intraparietandP_trans_y5'], 'fr_L_surf_S_oc_middleandLunatus': df['fr_L_surf_S_oc_middleandLunatus_y5'], 'fr_L_surf_S_oc_temp_lat': df['fr_L_surf_S_oc_temp_lat_y5'], 'fr_L_surf_S_oc_temp_medandLingual': df['fr_L_surf_S_oc_temp_medandLingual_y5'], 'fr_L_surf_S_orbital_H_Shaped': df['fr_L_surf_S_orbital_H_Shaped_y5'], 'fr_L_surf_S_orbital_lateral': df['fr_L_surf_S_orbital_lateral_y5'], 'fr_L_surf_S_orbital_med_olfact': df['fr_L_surf_S_orbital_med_olfact_y5'], 'fr_L_surf_S_parieto_occipital': df['fr_L_surf_S_parieto_occipital_y5'], 'fr_L_surf_S_postcentral': df['fr_L_surf_S_postcentral_y5'], 'fr_L_surf_S_precentral_inf_part': df['fr_L_surf_S_precentral_inf_part_y5'], 'fr_L_surf_S_precentral_sup_part': df['fr_L_surf_S_precentral_sup_part_y5'], 'fr_L_surf_S_suborbital': df['fr_L_surf_S_suborbital_y5'], 'fr_L_surf_S_subparietal': df['fr_L_surf_S_subparietal_y5'], 'fr_L_surf_S_temporal_inf': df['fr_L_surf_S_temporal_inf_y5'], 'fr_L_surf_S_temporal_sup': df['fr_L_surf_S_temporal_sup_y5']}		
		#frame_free6 = {'handlat': df['lat_manual'],'apoe': df['apoe'], 'y_last':df['ultimavisita'], 'dx_last': df['ultimodx'], 'dx_visita': df['dx_corto_visita6'], 'age_lastdx': df['edad_ultimodx'], 'age': df['edad_visita6'], 'sex': df['sexo'], 'free_R_Thal': df['fr_Right_Thalamus_Proper_y6'], 'free_L_Thal': df['fr_Left_Thalamus_Proper_y6'],'free_R_Puta': df['fr_Right_Putamen_y6'], 'free_L_Puta': df['fr_Left_Putamen_y6'],'free_R_Amyg': df['fr_Right_Amygdala_y6'], 'free_L_Amyg': df['fr_Left_Amygdala_y6'],'free_R_Pall': df['fr_Right_Pallidum_y6'], 'free_L_Pall': df['fr_Left_Pallidum_y6'],'free_R_Caud': df['fr_Right_Caudate_y6'], 'free_L_Caud': df['fr_Left_Caudate_y6'],'free_R_Hipp': df['fr_Right_Hippocampus_y6'], 'free_L_Hipp': df['fr_Left_Hippocampus_y6'],'free_R_Accu': df['fr_Right_Accumbens_area_y6'], 'free_L_Accu': df['fr_Left_Accumbens_area_y6'], 'fr_BrainSegVol': df['fr_BrainSegVol_y6'], 'fr_BrainSegVolNotVent':df['fr_BrainSegVolNotVent_y6'], 'fr_BrainSegVol_to_eTIV': df['fr_BrainSegVol_to_eTIV_y6']}
		frame_free6 = {'education': df['nivel_educativo'],'handlat': df['lat_manual'],'apoe': df['apoe'], 'nvisita':df['nvisita6'] , 'y_last':df['ultimavisita'], 'dx_last': df['ultimodx'], 'dx_visita': df['dx_corto_visita6'], 'age_lastdx': df['edad_ultimodx'], 'age': df['edad_visita6'], 'sex': df['sexo'], 'free_R_Thal': df['fr_Right_Thalamus_Proper_y6'], 'free_L_Thal': df['fr_Left_Thalamus_Proper_y6'],'free_R_Puta': df['fr_Right_Putamen_y6'], 'free_L_Puta': df['fr_Left_Putamen_y6'],'free_R_Amyg': df['fr_Right_Amygdala_y6'], 'free_L_Amyg': df['fr_Left_Amygdala_y6'],'free_R_Pall': df['fr_Right_Pallidum_y6'], 'free_L_Pall': df['fr_Left_Pallidum_y6'],'free_R_Caud': df['fr_Right_Caudate_y6'], 'free_L_Caud': df['fr_Left_Caudate_y6'],'free_R_Hipp': df['fr_Right_Hippocampus_y6'], 'free_L_Hipp': df['fr_Left_Hippocampus_y6'],'free_R_Accu': df['fr_Right_Accumbens_area_y6'], 'free_L_Accu': df['fr_Left_Accumbens_area_y6'], 'fr_BrainSegVol': df['fr_BrainSegVol_y6'], 'fr_BrainSegVolNotVent':df['fr_BrainSegVolNotVent_y6'], 'fr_BrainSegVol_to_eTIV': df['fr_BrainSegVol_to_eTIV_y6'], 'fr_R_surf_G_Ins_lgandS_cent_ins': df['fr_R_surf_G_Ins_lgandS_cent_ins_y6'], 'fr_R_surf_G_cingul_Post_ventral': df['fr_R_surf_G_cingul_Post_ventral_y6'], 'fr_R_surf_G_cuneus': df['fr_R_surf_G_cuneus_y6'], 'fr_R_surf_G_front_inf_Opercular': df['fr_R_surf_G_front_inf_Opercular_y6'], 'fr_R_surf_G_front_inf_Orbital': df['fr_R_surf_G_front_inf_Orbital_y6'], 'fr_R_surf_G_front_inf_Triangul': df['fr_R_surf_G_front_inf_Triangul_y6'], 'fr_R_surf_G_front_middle': df['fr_R_surf_G_front_middle_y6'], 'fr_R_surf_G_front_sup': df['fr_R_surf_G_front_sup_y6'], 'fr_R_surf_G_insular_short': df['fr_R_surf_G_insular_short_y6'], 'fr_R_surf_G_oc_temp_lat_fusifor': df['fr_R_surf_G_oc_temp_lat_fusifor_y6'], 'fr_R_surf_G_oc_temp_med_Lingual': df['fr_R_surf_G_oc_temp_med_Lingual_y6'], 'fr_R_surf_G_oc_temp_med_Parahip': df['fr_R_surf_G_oc_temp_med_Parahip_y6'], 'fr_R_surf_G_occipital_middle': df['fr_R_surf_G_occipital_middle_y6'], 'fr_R_surf_G_occipital_sup': df['fr_R_surf_G_occipital_sup_y6'], 'fr_R_surf_G_orbital': df['fr_R_surf_G_orbital_y6'], 'fr_R_surf_G_pariet_inf_Angular': df['fr_R_surf_G_pariet_inf_Angular_y6'], 'fr_R_surf_G_pariet_inf_Supramar': df['fr_R_surf_G_pariet_inf_Supramar_y6'], 'fr_R_surf_G_parietal_sup': df['fr_R_surf_G_parietal_sup_y6'], 'fr_R_surf_G_postcentral': df['fr_R_surf_G_postcentral_y6'], 'fr_R_surf_G_precentral': df['fr_R_surf_G_precentral_y6'], 'fr_R_surf_G_precuneus': df['fr_R_surf_G_precuneus_y6'], 'fr_R_surf_G_rectus': df['fr_R_surf_G_rectus_y6'], 'fr_R_surf_G_temp_sup_G_T_transv': df['fr_R_surf_G_temp_sup_G_T_transv_y6'], 'fr_R_surf_G_temp_sup_Lateral': df['fr_R_surf_G_temp_sup_Lateral_y6'], 'fr_R_surf_G_temp_sup_Plan_polar': df['fr_R_surf_G_temp_sup_Plan_polar_y6'], 'fr_R_surf_G_temp_sup_Plan_tempo': df['fr_R_surf_G_temp_sup_Plan_tempo_y6'], 'fr_R_surf_G_temporal_inf': df['fr_R_surf_G_temporal_inf_y6'], 'fr_R_surf_G_temporal_middle': df['fr_R_surf_G_temporal_middle_y6'], 'fr_R_surf_GandS_cingul_Ant': df['fr_R_surf_GandS_cingul_Ant_y6'], 'fr_R_surf_GandS_cingul_Mid_Post': df['fr_R_surf_GandS_cingul_Mid_Post_y6'], 'fr_R_surf_GandS_frontomargin': df['fr_R_surf_GandS_frontomargin_y6'], 'fr_R_surf_GandS_occipital_inf': df['fr_R_surf_GandS_occipital_inf_y6'], 'fr_R_surf_GandS_paracentral': df['fr_R_surf_GandS_paracentral_y6'], 'fr_R_surf_GandS_subcentral': df['fr_R_surf_GandS_subcentral_y6'], 'fr_R_surf_GandS_transv_frontopol': df['fr_R_surf_GandS_transv_frontopol_y6'], 'fr_R_surf_Lat_Fis_ant_Horizont': df['fr_R_surf_Lat_Fis_ant_Horizont_y6'], 'fr_R_surf_Lat_Fis_post': df['fr_R_surf_Lat_Fis_post_y6'], 'fr_R_surf_Pole_occipital': df['fr_R_surf_Pole_occipital_y6'], 'fr_R_surf_Pole_temporal': df['fr_R_surf_Pole_temporal_y6'], 'fr_R_surf_S_calcarine': df['fr_R_surf_S_calcarine_y6'], 'fr_R_surf_S_central': df['fr_R_surf_S_central_y6'], 'fr_R_surf_S_cingul_Marginalis': df['fr_R_surf_S_cingul_Marginalis_y6'], 'fr_R_surf_S_circular_insula_ant': df['fr_R_surf_S_circular_insula_ant_y6'], 'fr_R_surf_S_circular_insula_inf': df['fr_R_surf_S_circular_insula_inf_y6'], 'fr_R_surf_S_circular_insula_sup': df['fr_R_surf_S_circular_insula_sup_y6'], 'fr_R_surf_S_collat_transv_ant': df['fr_R_surf_S_collat_transv_ant_y6'], 'fr_R_surf_S_collat_transv_post': df['fr_R_surf_S_collat_transv_post_y6'], 'fr_R_surf_S_front_inf': df['fr_R_surf_S_front_inf_y6'], 'fr_R_surf_S_front_middle': df['fr_R_surf_S_front_middle_y6'], 'fr_R_surf_S_front_sup': df['fr_R_surf_S_front_sup_y6'], 'fr_R_surf_S_interm_prim_Jensen': df['fr_R_surf_S_interm_prim_Jensen_y6'], 'fr_R_surf_S_intraparietandP_trans': df['fr_R_surf_S_intraparietandP_trans_y6'], 'fr_R_surf_S_oc_middleandLunatus': df['fr_R_surf_S_oc_middleandLunatus_y6'], 'fr_R_surf_S_oc_temp_lat': df['fr_R_surf_S_oc_temp_lat_y6'], 'fr_R_surf_S_oc_temp_medandLingual': df['fr_R_surf_S_oc_temp_medandLingual_y6'], 'fr_R_surf_S_orbital_H_Shaped': df['fr_R_surf_S_orbital_H_Shaped_y6'], 'fr_R_surf_S_orbital_lateral': df['fr_R_surf_S_orbital_lateral_y6'], 'fr_R_surf_S_orbital_med_olfact': df['fr_R_surf_S_orbital_med_olfact_y6'], 'fr_R_surf_S_parieto_occipital': df['fr_R_surf_S_parieto_occipital_y6'], 'fr_R_surf_S_pericallosal': df['fr_R_surf_S_pericallosal_y6'], 'fr_R_surf_S_postcentral': df['fr_R_surf_S_postcentral_y6'], 'fr_R_surf_S_precentral_inf_part': df['fr_R_surf_S_precentral_inf_part_y6'], 'fr_R_surf_S_precentral_sup_part': df['fr_R_surf_S_precentral_sup_part_y6'], 'fr_R_surf_S_suborbital': df['fr_R_surf_S_suborbital_y6'], 'fr_R_surf_S_subparietal': df['fr_R_surf_S_subparietal_y6'], 'fr_R_surf_S_temporal_inf': df['fr_R_surf_S_temporal_inf_y6'], 'fr_R_surf_S_temporal_sup': df['fr_R_surf_S_temporal_sup_y6'], 'fr_R_surf_S_temporal_transverse': df['fr_R_surf_S_temporal_transverse_y6'],'fr_L_surf_G_Ins_lgandS_cent_ins': df['fr_L_surf_G_Ins_lgandS_cent_ins_y6'], 'fr_L_surf_G_cingul_Post_ventral': df['fr_L_surf_G_cingul_Post_ventral_y6'], 'fr_L_surf_G_cuneus': df['fr_L_surf_G_cuneus_y6'], 'fr_L_surf_G_front_inf_Opercular': df['fr_L_surf_G_front_inf_Opercular_y6'], 'fr_L_surf_G_front_inf_Orbital': df['fr_L_surf_G_front_inf_Orbital_y6'], 'fr_L_surf_G_front_inf_Triangul': df['fr_L_surf_G_front_inf_Triangul_y6'], 'fr_L_surf_G_front_middle': df['fr_L_surf_G_front_middle_y6'], 'fr_L_surf_G_front_sup': df['fr_L_surf_G_front_sup_y6'], 'fr_L_surf_G_insular_short': df['fr_L_surf_G_insular_short_y6'], 'fr_L_surf_G_oc_temp_lat_fusifor': df['fr_L_surf_G_oc_temp_lat_fusifor_y6'], 'fr_L_surf_G_oc_temp_med_Lingual': df['fr_L_surf_G_oc_temp_med_Lingual_y6'], 'fr_L_surf_G_oc_temp_med_Parahip': df['fr_L_surf_G_oc_temp_med_Parahip_y6'], 'fr_L_surf_G_occipital_middle': df['fr_L_surf_G_occipital_middle_y6'], 'fr_L_surf_G_occipital_sup': df['fr_L_surf_G_occipital_sup_y6'], 'fr_L_surf_G_orbital': df['fr_L_surf_G_orbital_y6'], 'fr_L_surf_G_pariet_inf_Angular': df['fr_L_surf_G_pariet_inf_Angular_y6'], 'fr_L_surf_G_pariet_inf_Supramar': df['fr_L_surf_G_pariet_inf_Supramar_y6'], 'fr_L_surf_G_parietal_sup': df['fr_L_surf_G_parietal_sup_y6'], 'fr_L_surf_G_postcentral': df['fr_L_surf_G_postcentral_y6'], 'fr_L_surf_G_precentral': df['fr_L_surf_G_precentral_y6'], 'fr_L_surf_G_precuneus': df['fr_L_surf_G_precuneus_y6'], 'fr_L_surf_G_rectus': df['fr_L_surf_G_rectus_y6'], 'fr_L_surf_G_temp_sup_G_T_transv': df['fr_L_surf_G_temp_sup_G_T_transv_y6'], 'fr_L_surf_G_temp_sup_Lateral': df['fr_L_surf_G_temp_sup_Lateral_y6'], 'fr_L_surf_G_temp_sup_Plan_polar': df['fr_L_surf_G_temp_sup_Plan_polar_y6'], 'fr_L_surf_G_temp_sup_Plan_tempo': df['fr_L_surf_G_temp_sup_Plan_tempo_y6'], 'fr_L_surf_G_temporal_inf': df['fr_L_surf_G_temporal_inf_y6'], 'fr_L_surf_G_temporal_middle': df['fr_L_surf_G_temporal_middle_y6'], 'fr_L_surf_GandS_cingul_Ant': df['fr_L_surf_GandS_cingul_Ant_y6'], 'fr_L_surf_GandS_cingul_Mid_Post': df['fr_L_surf_GandS_cingul_Mid_Post_y6'], 'fr_L_surf_GandS_frontomargin': df['fr_L_surf_GandS_frontomargin_y6'], 'fr_L_surf_GandS_occipital_inf': df['fr_L_surf_GandS_occipital_inf_y6'], 'fr_L_surf_GandS_paracentral': df['fr_L_surf_GandS_paracentral_y6'], 'fr_L_surf_GandS_subcentral': df['fr_L_surf_GandS_subcentral_y6'], 'fr_L_surf_GandS_transv_frontopol': df['fr_L_surf_GandS_transv_frontopol_y6'], 'fr_L_surf_Lat_Fis_ant_Horizont': df['fr_L_surf_Lat_Fis_ant_Horizont_y6'], 'fr_L_surf_Lat_Fis_post': df['fr_L_surf_Lat_Fis_post_y6'], 'fr_L_surf_Pole_occipital': df['fr_L_surf_Pole_occipital_y6'], 'fr_L_surf_Pole_temporal': df['fr_L_surf_Pole_temporal_y6'], 'fr_L_surf_S_calcarine': df['fr_L_surf_S_calcarine_y6'], 'fr_L_surf_S_central': df['fr_L_surf_S_central_y6'], 'fr_L_surf_S_cingul_Marginalis': df['fr_L_surf_S_cingul_Marginalis_y6'], 'fr_L_surf_S_circular_insula_ant': df['fr_L_surf_S_circular_insula_ant_y6'], 'fr_L_surf_S_circular_insula_inf': df['fr_L_surf_S_circular_insula_inf_y6'], 'fr_L_surf_S_circular_insula_sup': df['fr_L_surf_S_circular_insula_sup_y6'], 'fr_L_surf_S_collat_transv_ant': df['fr_L_surf_S_collat_transv_ant_y6'], 'fr_L_surf_S_collat_transv_post': df['fr_L_surf_S_collat_transv_post_y6'], 'fr_L_surf_S_front_inf': df['fr_L_surf_S_front_inf_y6'], 'fr_L_surf_S_front_middle': df['fr_L_surf_S_front_middle_y6'], 'fr_L_surf_S_front_sup': df['fr_L_surf_S_front_sup_y6'],  'fr_L_surf_S_intraparietandP_trans': df['fr_L_surf_S_intraparietandP_trans_y6'], 'fr_L_surf_S_oc_middleandLunatus': df['fr_L_surf_S_oc_middleandLunatus_y6'], 'fr_L_surf_S_oc_temp_lat': df['fr_L_surf_S_oc_temp_lat_y6'], 'fr_L_surf_S_oc_temp_medandLingual': df['fr_L_surf_S_oc_temp_medandLingual_y6'], 'fr_L_surf_S_orbital_H_Shaped': df['fr_L_surf_S_orbital_H_Shaped_y6'], 'fr_L_surf_S_orbital_lateral': df['fr_L_surf_S_orbital_lateral_y6'], 'fr_L_surf_S_orbital_med_olfact': df['fr_L_surf_S_orbital_med_olfact_y6'], 'fr_L_surf_S_parieto_occipital': df['fr_L_surf_S_parieto_occipital_y6'], 'fr_L_surf_S_postcentral': df['fr_L_surf_S_postcentral_y6'], 'fr_L_surf_S_precentral_inf_part': df['fr_L_surf_S_precentral_inf_part_y6'], 'fr_L_surf_S_precentral_sup_part': df['fr_L_surf_S_precentral_sup_part_y6'], 'fr_L_surf_S_suborbital': df['fr_L_surf_S_suborbital_y6'], 'fr_L_surf_S_subparietal': df['fr_L_surf_S_subparietal_y6'], 'fr_L_surf_S_temporal_inf': df['fr_L_surf_S_temporal_inf_y6'], 'fr_L_surf_S_temporal_sup': df['fr_L_surf_S_temporal_sup_y6']}
	elif thorsf == None:
		#frame_free1 = {'handlat': df['lat_manual'],'apoe': df['apoe'], 'y_last':df['ultimavisita'], 'dx_last': df['ultimodx'], 'dx_visita': df['dx_corto_visita1'], 'age_lastdx': df['edad_ultimodx'], 'age': df['edad_visita1'], 'sex': df['sexo'], 'free_R_Thal': df['fr_Right_Thalamus_Proper_y1'], 'free_L_Thal': df['fr_Left_Thalamus_Proper_y1'],'free_R_Puta': df['fr_Right_Putamen_y1'], 'free_L_Puta': df['fr_Left_Putamen_y1'],'free_R_Amyg': df['fr_Right_Amygdala_y1'], 'free_L_Amyg': df['fr_Left_Amygdala_y1'],'free_R_Pall': df['fr_Right_Pallidum_y1'], 'free_L_Pall': df['fr_Left_Pallidum_y1'],'free_R_Caud': df['fr_Right_Caudate_y1'], 'free_L_Caud': df['fr_Left_Caudate_y1'],'free_R_Hipp': df['fr_Right_Hippocampus_y1'], 'free_L_Hipp': df['fr_Left_Hippocampus_y1'],'free_R_Accu': df['fr_Right_Accumbens_area_y1'], 'free_L_Accu': df['fr_Left_Accumbens_area_y1'], 'fr_BrainSegVol': df['fr_BrainSegVol_y1'], 'fr_BrainSegVolNotVent':df['fr_BrainSegVolNotVent_y1'], 'fr_BrainSegVol_to_eTIV': df['fr_BrainSegVol_to_eTIV_y1']}
		frame_free1 = {'education': df['nivel_educativo'],'handlat': df['lat_manual'],'apoe': df['apoe'],'nvisita':df['nvisita1'] ,'y_last':df['ultimavisita'], 'dx_last': df['ultimodx'], 'dx_visita': df['dx_corto_visita1'], 'age_lastdx': df['edad_ultimodx'], 'age': df['edad_visita1'], 'sex': df['sexo'], 'free_R_Thal': df['fr_Right_Thalamus_Proper_y1'], 'free_L_Thal': df['fr_Left_Thalamus_Proper_y1'],'free_R_Puta': df['fr_Right_Putamen_y1'], 'free_L_Puta': df['fr_Left_Putamen_y1'],'free_R_Amyg': df['fr_Right_Amygdala_y1'], 'free_L_Amyg': df['fr_Left_Amygdala_y1'],'free_R_Pall': df['fr_Right_Pallidum_y1'], 'free_L_Pall': df['fr_Left_Pallidum_y1'],'free_R_Caud': df['fr_Right_Caudate_y1'], 'free_L_Caud': df['fr_Left_Caudate_y1'],'free_R_Hipp': df['fr_Right_Hippocampus_y1'], 'free_L_Hipp': df['fr_Left_Hippocampus_y1'],'free_R_Accu': df['fr_Right_Accumbens_area_y1'], 'free_L_Accu': df['fr_Left_Accumbens_area_y1'], 'fr_BrainSegVol': df['fr_BrainSegVol_y1'], 'fr_BrainSegVolNotVent':df['fr_BrainSegVolNotVent_y1'], 'fr_BrainSegVol_to_eTIV': df['fr_BrainSegVol_to_eTIV_y1'], 'fr_R_surf_G_Ins_lgandS_cent_ins': df['fr_R_surf_G_Ins_lgandS_cent_ins_y1'], 'fr_R_surf_G_cingul_Post_ventral': df['fr_R_surf_G_cingul_Post_ventral_y1'], 'fr_R_surf_G_cuneus': df['fr_R_surf_G_cuneus_y1'], 'fr_R_surf_G_front_inf_Opercular': df['fr_R_surf_G_front_inf_Opercular_y1'], 'fr_R_surf_G_front_inf_Orbital': df['fr_R_surf_G_front_inf_Orbital_y1'], 'fr_R_surf_G_front_inf_Triangul': df['fr_R_surf_G_front_inf_Triangul_y1'], 'fr_R_surf_G_front_middle': df['fr_R_surf_G_front_middle_y1'], 'fr_R_surf_G_front_sup': df['fr_R_surf_G_front_sup_y1'], 'fr_R_surf_G_insular_short': df['fr_R_surf_G_insular_short_y1'], 'fr_R_surf_G_oc_temp_lat_fusifor': df['fr_R_surf_G_oc_temp_lat_fusifor_y1'], 'fr_R_surf_G_oc_temp_med_Lingual': df['fr_R_surf_G_oc_temp_med_Lingual_y1'], 'fr_R_surf_G_oc_temp_med_Parahip': df['fr_R_surf_G_oc_temp_med_Parahip_y1'], 'fr_R_surf_G_occipital_middle': df['fr_R_surf_G_occipital_middle_y1'], 'fr_R_surf_G_occipital_sup': df['fr_R_surf_G_occipital_sup_y1'], 'fr_R_surf_G_orbital': df['fr_R_surf_G_orbital_y1'], 'fr_R_surf_G_pariet_inf_Angular': df['fr_R_surf_G_pariet_inf_Angular_y1'], 'fr_R_surf_G_pariet_inf_Supramar': df['fr_R_surf_G_pariet_inf_Supramar_y1'], 'fr_R_surf_G_parietal_sup': df['fr_R_surf_G_parietal_sup_y1'], 'fr_R_surf_G_postcentral': df['fr_R_surf_G_postcentral_y1'], 'fr_R_surf_G_precentral': df['fr_R_surf_G_precentral_y1'], 'fr_R_surf_G_precuneus': df['fr_R_surf_G_precuneus_y1'], 'fr_R_surf_G_rectus': df['fr_R_surf_G_rectus_y1'], 'fr_R_surf_G_temp_sup_G_T_transv': df['fr_R_surf_G_temp_sup_G_T_transv_y1'], 'fr_R_surf_G_temp_sup_Lateral': df['fr_R_surf_G_temp_sup_Lateral_y1'], 'fr_R_surf_G_temp_sup_Plan_polar': df['fr_R_surf_G_temp_sup_Plan_polar_y1'], 'fr_R_surf_G_temp_sup_Plan_tempo': df['fr_R_surf_G_temp_sup_Plan_tempo_y1'], 'fr_R_surf_G_temporal_inf': df['fr_R_surf_G_temporal_inf_y1'], 'fr_R_surf_G_temporal_middle': df['fr_R_surf_G_temporal_middle_y1'], 'fr_R_surf_GandS_cingul_Ant': df['fr_R_surf_GandS_cingul_Ant_y1'], 'fr_R_surf_GandS_cingul_Mid_Post': df['fr_R_surf_GandS_cingul_Mid_Post_y1'], 'fr_R_surf_GandS_frontomargin': df['fr_R_surf_GandS_frontomargin_y1'], 'fr_R_surf_GandS_occipital_inf': df['fr_R_surf_GandS_occipital_inf_y1'], 'fr_R_surf_GandS_paracentral': df['fr_R_surf_GandS_paracentral_y1'], 'fr_R_surf_GandS_subcentral': df['fr_R_surf_GandS_subcentral_y1'], 'fr_R_surf_GandS_transv_frontopol': df['fr_R_surf_GandS_transv_frontopol_y1'], 'fr_R_surf_Lat_Fis_ant_Horizont': df['fr_R_surf_Lat_Fis_ant_Horizont_y1'], 'fr_R_surf_Lat_Fis_post': df['fr_R_surf_Lat_Fis_post_y1'], 'fr_R_surf_Pole_occipital': df['fr_R_surf_Pole_occipital_y1'], 'fr_R_surf_Pole_temporal': df['fr_R_surf_Pole_temporal_y1'], 'fr_R_surf_S_calcarine': df['fr_R_surf_S_calcarine_y1'], 'fr_R_surf_S_central': df['fr_R_surf_S_central_y1'], 'fr_R_surf_S_cingul_Marginalis': df['fr_R_surf_S_cingul_Marginalis_y1'], 'fr_R_surf_S_circular_insula_ant': df['fr_R_surf_S_circular_insula_ant_y1'], 'fr_R_surf_S_circular_insula_inf': df['fr_R_surf_S_circular_insula_inf_y1'], 'fr_R_surf_S_circular_insula_sup': df['fr_R_surf_S_circular_insula_sup_y1'], 'fr_R_surf_S_collat_transv_ant': df['fr_R_surf_S_collat_transv_ant_y1'], 'fr_R_surf_S_collat_transv_post': df['fr_R_surf_S_collat_transv_post_y1'], 'fr_R_surf_S_front_inf': df['fr_R_surf_S_front_inf_y1'], 'fr_R_surf_S_front_middle': df['fr_R_surf_S_front_middle_y1'], 'fr_R_surf_S_front_sup': df['fr_R_surf_S_front_sup_y1'], 'fr_R_surf_S_interm_prim_Jensen': df['fr_R_surf_S_interm_prim_Jensen_y1'], 'fr_R_surf_S_intraparietandP_trans': df['fr_R_surf_S_intraparietandP_trans_y1'], 'fr_R_surf_S_oc_middleandLunatus': df['fr_R_surf_S_oc_middleandLunatus_y1'], 'fr_R_surf_S_oc_temp_lat': df['fr_R_surf_S_oc_temp_lat_y1'], 'fr_R_surf_S_oc_temp_medandLingual': df['fr_R_surf_S_oc_temp_medandLingual_y1'], 'fr_R_surf_S_orbital_H_Shaped': df['fr_R_surf_S_orbital_H_Shaped_y1'], 'fr_R_surf_S_orbital_lateral': df['fr_R_surf_S_orbital_lateral_y1'], 'fr_R_surf_S_orbital_med_olfact': df['fr_R_surf_S_orbital_med_olfact_y1'], 'fr_R_surf_S_parieto_occipital': df['fr_R_surf_S_parieto_occipital_y1'], 'fr_R_surf_S_pericallosal': df['fr_R_surf_S_pericallosal_y1'], 'fr_R_surf_S_postcentral': df['fr_R_surf_S_postcentral_y1'], 'fr_R_surf_S_precentral_inf_part': df['fr_R_surf_S_precentral_inf_part_y1'], 'fr_R_surf_S_precentral_sup_part': df['fr_R_surf_S_precentral_sup_part_y1'], 'fr_R_surf_S_suborbital': df['fr_R_surf_S_suborbital_y1'], 'fr_R_surf_S_subparietal': df['fr_R_surf_S_subparietal_y1'], 'fr_R_surf_S_temporal_inf': df['fr_R_surf_S_temporal_inf_y1'], 'fr_R_surf_S_temporal_sup': df['fr_R_surf_S_temporal_sup_y1'], 'fr_R_surf_S_temporal_transverse': df['fr_R_surf_S_temporal_transverse_y1'],'fr_R_thick_G_Ins_lgandS_cent_ins': df['fr_R_thick_G_Ins_lgandS_cent_ins_y1'], 'fr_R_thick_G_cingul_Post_ventral': df['fr_R_thick_G_cingul_Post_ventral_y1'], 'fr_R_thick_G_cuneus': df['fr_R_thick_G_cuneus_y1'], 'fr_R_thick_G_front_inf_Opercular': df['fr_R_thick_G_front_inf_Opercular_y1'], 'fr_R_thick_G_front_inf_Orbital': df['fr_R_thick_G_front_inf_Orbital_y1'], 'fr_R_thick_G_front_inf_Triangul': df['fr_R_thick_G_front_inf_Triangul_y1'], 'fr_R_thick_G_front_middle': df['fr_R_thick_G_front_middle_y1'], 'fr_R_thick_G_front_sup': df['fr_R_thick_G_front_sup_y1'], 'fr_R_thick_G_insular_short': df['fr_R_thick_G_insular_short_y1'], 'fr_R_thick_G_oc_temp_lat_fusifor': df['fr_R_thick_G_oc_temp_lat_fusifor_y1'], 'fr_R_thick_G_oc_temp_med_Lingual': df['fr_R_thick_G_oc_temp_med_Lingual_y1'], 'fr_R_thick_G_oc_temp_med_Parahip': df['fr_R_thick_G_oc_temp_med_Parahip_y1'], 'fr_R_thick_G_occipital_middle': df['fr_R_thick_G_occipital_middle_y1'], 'fr_R_thick_G_occipital_sup': df['fr_R_thick_G_occipital_sup_y1'], 'fr_R_thick_G_orbital': df['fr_R_thick_G_orbital_y1'], 'fr_R_thick_G_pariet_inf_Angular': df['fr_R_thick_G_pariet_inf_Angular_y1'], 'fr_R_thick_G_pariet_inf_Supramar': df['fr_R_thick_G_pariet_inf_Supramar_y1'], 'fr_R_thick_G_parietal_sup': df['fr_R_thick_G_parietal_sup_y1'], 'fr_R_thick_G_postcentral': df['fr_R_thick_G_postcentral_y1'], 'fr_R_thick_G_precentral': df['fr_R_thick_G_precentral_y1'], 'fr_R_thick_G_precuneus': df['fr_R_thick_G_precuneus_y1'], 'fr_R_thick_G_rectus': df['fr_R_thick_G_rectus_y1'], 'fr_R_thick_G_temp_sup_G_T_transv': df['fr_R_thick_G_temp_sup_G_T_transv_y1'], 'fr_R_thick_G_temp_sup_Lateral': df['fr_R_thick_G_temp_sup_Lateral_y1'], 'fr_R_thick_G_temp_sup_Plan_polar': df['fr_R_thick_G_temp_sup_Plan_polar_y1'], 'fr_R_thick_G_temp_sup_Plan_tempo': df['fr_R_thick_G_temp_sup_Plan_tempo_y1'], 'fr_R_thick_G_temporal_inf': df['fr_R_thick_G_temporal_inf_y1'], 'fr_R_thick_G_temporal_middle': df['fr_R_thick_G_temporal_middle_y1'], 'fr_R_thick_GandS_cingul_Ant': df['fr_R_thick_GandS_cingul_Ant_y1'], 'fr_R_thick_GandS_cingul_Mid_Post': df['fr_R_thick_GandS_cingul_Mid_Post_y1'], 'fr_R_thick_GandS_frontomargin': df['fr_R_thick_GandS_frontomargin_y1'], 'fr_R_thick_GandS_occipital_inf': df['fr_R_thick_GandS_occipital_inf_y1'], 'fr_R_thick_GandS_paracentral': df['fr_R_thick_GandS_paracentral_y1'], 'fr_R_thick_GandS_subcentral': df['fr_R_thick_GandS_subcentral_y1'], 'fr_R_thick_GandS_transv_frontopol': df['fr_R_thick_GandS_transv_frontopol_y1'], 'fr_R_thick_Lat_Fis_ant_Horizont': df['fr_R_thick_Lat_Fis_ant_Horizont_y1'], 'fr_R_thick_Lat_Fis_post': df['fr_R_thick_Lat_Fis_post_y1'], 'fr_R_thick_Pole_occipital': df['fr_R_thick_Pole_occipital_y1'], 'fr_R_thick_Pole_temporal': df['fr_R_thick_Pole_temporal_y1'], 'fr_R_thick_S_calcarine': df['fr_R_thick_S_calcarine_y1'], 'fr_R_thick_S_central': df['fr_R_thick_S_central_y1'], 'fr_R_thick_S_cingul_Marginalis': df['fr_R_thick_S_cingul_Marginalis_y1'], 'fr_R_thick_S_circular_insula_ant': df['fr_R_thick_S_circular_insula_ant_y1'], 'fr_R_thick_S_circular_insula_inf': df['fr_R_thick_S_circular_insula_inf_y1'], 'fr_R_thick_S_circular_insula_sup': df['fr_R_thick_S_circular_insula_sup_y1'], 'fr_R_thick_S_collat_transv_ant': df['fr_R_thick_S_collat_transv_ant_y1'], 'fr_R_thick_S_collat_transv_post': df['fr_R_thick_S_collat_transv_post_y1'], 'fr_R_thick_S_front_inf': df['fr_R_thick_S_front_inf_y1'], 'fr_R_thick_S_front_middle': df['fr_R_thick_S_front_middle_y1'], 'fr_R_thick_S_front_sup': df['fr_R_thick_S_front_sup_y1'], 'fr_R_thick_S_interm_prim_Jensen': df['fr_R_thick_S_interm_prim_Jensen_y1'], 'fr_R_thick_S_intraparietandP_trans': df['fr_R_thick_S_intraparietandP_trans_y1'], 'fr_R_thick_S_oc_middleandLunatus': df['fr_R_thick_S_oc_middleandLunatus_y1'], 'fr_R_thick_S_oc_temp_lat': df['fr_R_thick_S_oc_temp_lat_y1'], 'fr_R_thick_S_oc_temp_medandLingual': df['fr_R_thick_S_oc_temp_medandLingual_y1'], 'fr_R_thick_S_orbital_H_Shaped': df['fr_R_thick_S_orbital_H_Shaped_y1'], 'fr_R_thick_S_orbital_lateral': df['fr_R_thick_S_orbital_lateral_y1'], 'fr_R_thick_S_orbital_med_olfact': df['fr_R_thick_S_orbital_med_olfact_y1'], 'fr_R_thick_S_parieto_occipital': df['fr_R_thick_S_parieto_occipital_y1'], 'fr_R_thick_S_pericallosal': df['fr_R_thick_S_pericallosal_y1'], 'fr_R_thick_S_postcentral': df['fr_R_thick_S_postcentral_y1'], 'fr_R_thick_S_precentral_inf_part': df['fr_R_thick_S_precentral_inf_part_y1'], 'fr_R_thick_S_precentral_sup_part': df['fr_R_thick_S_precentral_sup_part_y1'], 'fr_R_thick_S_suborbital': df['fr_R_thick_S_suborbital_y1'], 'fr_R_thick_S_subparietal': df['fr_R_thick_S_subparietal_y1'], 'fr_R_thick_S_temporal_inf': df['fr_R_thick_S_temporal_inf_y1'], 'fr_R_thick_S_temporal_sup': df['fr_R_thick_S_temporal_sup_y1'], 'fr_R_thick_S_temporal_transverse': df['fr_R_thick_S_temporal_transverse_y1'], 'fr_L_surf_G_Ins_lgandS_cent_ins': df['fr_L_surf_G_Ins_lgandS_cent_ins_y1'], 'fr_L_surf_G_cingul_Post_ventral': df['fr_L_surf_G_cingul_Post_ventral_y1'], 'fr_L_surf_G_cuneus': df['fr_L_surf_G_cuneus_y1'], 'fr_L_surf_G_front_inf_Opercular': df['fr_L_surf_G_front_inf_Opercular_y1'], 'fr_L_surf_G_front_inf_Orbital': df['fr_L_surf_G_front_inf_Orbital_y1'], 'fr_L_surf_G_front_inf_Triangul': df['fr_L_surf_G_front_inf_Triangul_y1'], 'fr_L_surf_G_front_middle': df['fr_L_surf_G_front_middle_y1'], 'fr_L_surf_G_front_sup': df['fr_L_surf_G_front_sup_y1'], 'fr_L_surf_G_insular_short': df['fr_L_surf_G_insular_short_y1'], 'fr_L_surf_G_oc_temp_lat_fusifor': df['fr_L_surf_G_oc_temp_lat_fusifor_y1'], 'fr_L_surf_G_oc_temp_med_Lingual': df['fr_L_surf_G_oc_temp_med_Lingual_y1'], 'fr_L_surf_G_oc_temp_med_Parahip': df['fr_L_surf_G_oc_temp_med_Parahip_y1'], 'fr_L_surf_G_occipital_middle': df['fr_L_surf_G_occipital_middle_y1'], 'fr_L_surf_G_occipital_sup': df['fr_L_surf_G_occipital_sup_y1'], 'fr_L_surf_G_orbital': df['fr_L_surf_G_orbital_y1'], 'fr_L_surf_G_pariet_inf_Angular': df['fr_L_surf_G_pariet_inf_Angular_y1'], 'fr_L_surf_G_pariet_inf_Supramar': df['fr_L_surf_G_pariet_inf_Supramar_y1'], 'fr_L_surf_G_parietal_sup': df['fr_L_surf_G_parietal_sup_y1'], 'fr_L_surf_G_postcentral': df['fr_L_surf_G_postcentral_y1'], 'fr_L_surf_G_precentral': df['fr_L_surf_G_precentral_y1'], 'fr_L_surf_G_precuneus': df['fr_L_surf_G_precuneus_y1'], 'fr_L_surf_G_rectus': df['fr_L_surf_G_rectus_y1'], 'fr_L_surf_G_temp_sup_G_T_transv': df['fr_L_surf_G_temp_sup_G_T_transv_y1'], 'fr_L_surf_G_temp_sup_Lateral': df['fr_L_surf_G_temp_sup_Lateral_y1'], 'fr_L_surf_G_temp_sup_Plan_polar': df['fr_L_surf_G_temp_sup_Plan_polar_y1'], 'fr_L_surf_G_temp_sup_Plan_tempo': df['fr_L_surf_G_temp_sup_Plan_tempo_y1'], 'fr_L_surf_G_temporal_inf': df['fr_L_surf_G_temporal_inf_y1'], 'fr_L_surf_G_temporal_middle': df['fr_L_surf_G_temporal_middle_y1'], 'fr_L_surf_GandS_cingul_Ant': df['fr_L_surf_GandS_cingul_Ant_y1'], 'fr_L_surf_GandS_cingul_Mid_Post': df['fr_L_surf_GandS_cingul_Mid_Post_y1'], 'fr_L_surf_GandS_frontomargin': df['fr_L_surf_GandS_frontomargin_y1'], 'fr_L_surf_GandS_occipital_inf': df['fr_L_surf_GandS_occipital_inf_y1'], 'fr_L_surf_GandS_paracentral': df['fr_L_surf_GandS_paracentral_y1'], 'fr_L_surf_GandS_subcentral': df['fr_L_surf_GandS_subcentral_y1'], 'fr_L_surf_GandS_transv_frontopol': df['fr_L_surf_GandS_transv_frontopol_y1'], 'fr_L_surf_Lat_Fis_ant_Horizont': df['fr_L_surf_Lat_Fis_ant_Horizont_y1'], 'fr_L_surf_Lat_Fis_post': df['fr_L_surf_Lat_Fis_post_y1'], 'fr_L_surf_Pole_occipital': df['fr_L_surf_Pole_occipital_y1'], 'fr_L_surf_Pole_temporal': df['fr_L_surf_Pole_temporal_y1'], 'fr_L_surf_S_calcarine': df['fr_L_surf_S_calcarine_y1'], 'fr_L_surf_S_central': df['fr_L_surf_S_central_y1'], 'fr_L_surf_S_cingul_Marginalis': df['fr_L_surf_S_cingul_Marginalis_y1'], 'fr_L_surf_S_circular_insula_ant': df['fr_L_surf_S_circular_insula_ant_y1'], 'fr_L_surf_S_circular_insula_inf': df['fr_L_surf_S_circular_insula_inf_y1'], 'fr_L_surf_S_circular_insula_sup': df['fr_L_surf_S_circular_insula_sup_y1'], 'fr_L_surf_S_collat_transv_ant': df['fr_L_surf_S_collat_transv_ant_y1'], 'fr_L_surf_S_collat_transv_post': df['fr_L_surf_S_collat_transv_post_y1'], 'fr_L_surf_S_front_inf': df['fr_L_surf_S_front_inf_y1'], 'fr_L_surf_S_front_middle': df['fr_L_surf_S_front_middle_y1'], 'fr_L_surf_S_front_sup': df['fr_L_surf_S_front_sup_y1'],  'fr_L_surf_S_intraparietandP_trans': df['fr_L_surf_S_intraparietandP_trans_y1'], 'fr_L_surf_S_oc_middleandLunatus': df['fr_L_surf_S_oc_middleandLunatus_y1'], 'fr_L_surf_S_oc_temp_lat': df['fr_L_surf_S_oc_temp_lat_y1'], 'fr_L_surf_S_oc_temp_medandLingual': df['fr_L_surf_S_oc_temp_medandLingual_y1'], 'fr_L_surf_S_orbital_H_Shaped': df['fr_L_surf_S_orbital_H_Shaped_y1'], 'fr_L_surf_S_orbital_lateral': df['fr_L_surf_S_orbital_lateral_y1'], 'fr_L_surf_S_orbital_med_olfact': df['fr_L_surf_S_orbital_med_olfact_y1'], 'fr_L_surf_S_parieto_occipital': df['fr_L_surf_S_parieto_occipital_y1'], 'fr_L_surf_S_postcentral': df['fr_L_surf_S_postcentral_y1'], 'fr_L_surf_S_precentral_inf_part': df['fr_L_surf_S_precentral_inf_part_y1'], 'fr_L_surf_S_precentral_sup_part': df['fr_L_surf_S_precentral_sup_part_y1'], 'fr_L_surf_S_suborbital': df['fr_L_surf_S_suborbital_y1'], 'fr_L_surf_S_subparietal': df['fr_L_surf_S_subparietal_y1'], 'fr_L_surf_S_temporal_inf': df['fr_L_surf_S_temporal_inf_y1'], 'fr_L_surf_S_temporal_sup': df['fr_L_surf_S_temporal_sup_y1']}
		#frame_free2 = {'handlat': df['lat_manual'],'apoe': df['apoe'], 'y_last':df['ultimavisita'], 'dx_last': df['ultimodx'], 'dx_visita': df['dx_corto_visita2'], 'age_lastdx': df['edad_ultimodx'], 'age': df['edad_visita2'], 'sex': df['sexo'], 'free_R_Thal': df['fr_Right_Thalamus_Proper_y2'], 'free_L_Thal': df['fr_Left_Thalamus_Proper_y2'],'free_R_Puta': df['fr_Right_Putamen_y2'], 'free_L_Puta': df['fr_Left_Putamen_y2'],'free_R_Amyg': df['fr_Right_Amygdala_y2'], 'free_L_Amyg': df['fr_Left_Amygdala_y2'],'free_R_Pall': df['fr_Right_Pallidum_y2'], 'free_L_Pall': df['fr_Left_Pallidum_y2'],'free_R_Caud': df['fr_Right_Caudate_y2'], 'free_L_Caud': df['fr_Left_Caudate_y2'],'free_R_Hipp': df['fr_Right_Hippocampus_y2'], 'free_L_Hipp': df['fr_Left_Hippocampus_y2'],'free_R_Accu': df['fr_Right_Accumbens_area_y2'], 'free_L_Accu': df['fr_Left_Accumbens_area_y2'], 'fr_BrainSegVol': df['fr_BrainSegVol_y2'], 'fr_BrainSegVolNotVent':df['fr_BrainSegVolNotVent_y2'], 'fr_BrainSegVol_to_eTIV': df['fr_BrainSegVol_to_eTIV_y2']}
		frame_free2 = {'education': df['nivel_educativo'],'handlat': df['lat_manual'],'apoe': df['apoe'], 'nvisita':df['nvisita2'], 'y_last':df['ultimavisita'], 'dx_last': df['ultimodx'], 'dx_visita': df['dx_corto_visita2'], 'age_lastdx': df['edad_ultimodx'], 'age': df['edad_visita2'], 'sex': df['sexo'], 'free_R_Thal': df['fr_Right_Thalamus_Proper_y2'], 'free_L_Thal': df['fr_Left_Thalamus_Proper_y2'],'free_R_Puta': df['fr_Right_Putamen_y2'], 'free_L_Puta': df['fr_Left_Putamen_y2'],'free_R_Amyg': df['fr_Right_Amygdala_y2'], 'free_L_Amyg': df['fr_Left_Amygdala_y2'],'free_R_Pall': df['fr_Right_Pallidum_y2'], 'free_L_Pall': df['fr_Left_Pallidum_y2'],'free_R_Caud': df['fr_Right_Caudate_y2'], 'free_L_Caud': df['fr_Left_Caudate_y2'],'free_R_Hipp': df['fr_Right_Hippocampus_y2'], 'free_L_Hipp': df['fr_Left_Hippocampus_y2'],'free_R_Accu': df['fr_Right_Accumbens_area_y2'], 'free_L_Accu': df['fr_Left_Accumbens_area_y2'], 'fr_BrainSegVol': df['fr_BrainSegVol_y2'], 'fr_BrainSegVolNotVent':df['fr_BrainSegVolNotVent_y2'], 'fr_BrainSegVol_to_eTIV': df['fr_BrainSegVol_to_eTIV_y2'],  'fr_R_surf_G_Ins_lgandS_cent_ins': df['fr_R_surf_G_Ins_lgandS_cent_ins_y2'], 'fr_R_surf_G_cingul_Post_ventral': df['fr_R_surf_G_cingul_Post_ventral_y2'], 'fr_R_surf_G_cuneus': df['fr_R_surf_G_cuneus_y2'], 'fr_R_surf_G_front_inf_Opercular': df['fr_R_surf_G_front_inf_Opercular_y2'], 'fr_R_surf_G_front_inf_Orbital': df['fr_R_surf_G_front_inf_Orbital_y2'], 'fr_R_surf_G_front_inf_Triangul': df['fr_R_surf_G_front_inf_Triangul_y2'], 'fr_R_surf_G_front_middle': df['fr_R_surf_G_front_middle_y2'], 'fr_R_surf_G_front_sup': df['fr_R_surf_G_front_sup_y2'], 'fr_R_surf_G_insular_short': df['fr_R_surf_G_insular_short_y2'], 'fr_R_surf_G_oc_temp_lat_fusifor': df['fr_R_surf_G_oc_temp_lat_fusifor_y2'], 'fr_R_surf_G_oc_temp_med_Lingual': df['fr_R_surf_G_oc_temp_med_Lingual_y2'], 'fr_R_surf_G_oc_temp_med_Parahip': df['fr_R_surf_G_oc_temp_med_Parahip_y2'], 'fr_R_surf_G_occipital_middle': df['fr_R_surf_G_occipital_middle_y2'], 'fr_R_surf_G_occipital_sup': df['fr_R_surf_G_occipital_sup_y2'], 'fr_R_surf_G_orbital': df['fr_R_surf_G_orbital_y2'], 'fr_R_surf_G_pariet_inf_Angular': df['fr_R_surf_G_pariet_inf_Angular_y2'], 'fr_R_surf_G_pariet_inf_Supramar': df['fr_R_surf_G_pariet_inf_Supramar_y2'], 'fr_R_surf_G_parietal_sup': df['fr_R_surf_G_parietal_sup_y2'], 'fr_R_surf_G_postcentral': df['fr_R_surf_G_postcentral_y2'], 'fr_R_surf_G_precentral': df['fr_R_surf_G_precentral_y2'], 'fr_R_surf_G_precuneus': df['fr_R_surf_G_precuneus_y2'], 'fr_R_surf_G_rectus': df['fr_R_surf_G_rectus_y2'], 'fr_R_surf_G_temp_sup_G_T_transv': df['fr_R_surf_G_temp_sup_G_T_transv_y2'], 'fr_R_surf_G_temp_sup_Lateral': df['fr_R_surf_G_temp_sup_Lateral_y2'], 'fr_R_surf_G_temp_sup_Plan_polar': df['fr_R_surf_G_temp_sup_Plan_polar_y2'], 'fr_R_surf_G_temp_sup_Plan_tempo': df['fr_R_surf_G_temp_sup_Plan_tempo_y2'], 'fr_R_surf_G_temporal_inf': df['fr_R_surf_G_temporal_inf_y2'], 'fr_R_surf_G_temporal_middle': df['fr_R_surf_G_temporal_middle_y2'], 'fr_R_surf_GandS_cingul_Ant': df['fr_R_surf_GandS_cingul_Ant_y2'], 'fr_R_surf_GandS_cingul_Mid_Post': df['fr_R_surf_GandS_cingul_Mid_Post_y2'], 'fr_R_surf_GandS_frontomargin': df['fr_R_surf_GandS_frontomargin_y2'], 'fr_R_surf_GandS_occipital_inf': df['fr_R_surf_GandS_occipital_inf_y2'], 'fr_R_surf_GandS_paracentral': df['fr_R_surf_GandS_paracentral_y2'], 'fr_R_surf_GandS_subcentral': df['fr_R_surf_GandS_subcentral_y2'], 'fr_R_surf_GandS_transv_frontopol': df['fr_R_surf_GandS_transv_frontopol_y2'], 'fr_R_surf_Lat_Fis_ant_Horizont': df['fr_R_surf_Lat_Fis_ant_Horizont_y2'], 'fr_R_surf_Lat_Fis_post': df['fr_R_surf_Lat_Fis_post_y2'], 'fr_R_surf_Pole_occipital': df['fr_R_surf_Pole_occipital_y2'], 'fr_R_surf_Pole_temporal': df['fr_R_surf_Pole_temporal_y2'], 'fr_R_surf_S_calcarine': df['fr_R_surf_S_calcarine_y2'], 'fr_R_surf_S_central': df['fr_R_surf_S_central_y2'], 'fr_R_surf_S_cingul_Marginalis': df['fr_R_surf_S_cingul_Marginalis_y2'], 'fr_R_surf_S_circular_insula_ant': df['fr_R_surf_S_circular_insula_ant_y2'], 'fr_R_surf_S_circular_insula_inf': df['fr_R_surf_S_circular_insula_inf_y2'], 'fr_R_surf_S_circular_insula_sup': df['fr_R_surf_S_circular_insula_sup_y2'], 'fr_R_surf_S_collat_transv_ant': df['fr_R_surf_S_collat_transv_ant_y2'], 'fr_R_surf_S_collat_transv_post': df['fr_R_surf_S_collat_transv_post_y2'], 'fr_R_surf_S_front_inf': df['fr_R_surf_S_front_inf_y2'], 'fr_R_surf_S_front_middle': df['fr_R_surf_S_front_middle_y2'], 'fr_R_surf_S_front_sup': df['fr_R_surf_S_front_sup_y2'], 'fr_R_surf_S_interm_prim_Jensen': df['fr_R_surf_S_interm_prim_Jensen_y2'], 'fr_R_surf_S_intraparietandP_trans': df['fr_R_surf_S_intraparietandP_trans_y2'], 'fr_R_surf_S_oc_middleandLunatus': df['fr_R_surf_S_oc_middleandLunatus_y2'], 'fr_R_surf_S_oc_temp_lat': df['fr_R_surf_S_oc_temp_lat_y2'], 'fr_R_surf_S_oc_temp_medandLingual': df['fr_R_surf_S_oc_temp_medandLingual_y2'], 'fr_R_surf_S_orbital_H_Shaped': df['fr_R_surf_S_orbital_H_Shaped_y2'], 'fr_R_surf_S_orbital_lateral': df['fr_R_surf_S_orbital_lateral_y2'], 'fr_R_surf_S_orbital_med_olfact': df['fr_R_surf_S_orbital_med_olfact_y2'], 'fr_R_surf_S_parieto_occipital': df['fr_R_surf_S_parieto_occipital_y2'], 'fr_R_surf_S_pericallosal': df['fr_R_surf_S_pericallosal_y2'], 'fr_R_surf_S_postcentral': df['fr_R_surf_S_postcentral_y2'], 'fr_R_surf_S_precentral_inf_part': df['fr_R_surf_S_precentral_inf_part_y2'], 'fr_R_surf_S_precentral_sup_part': df['fr_R_surf_S_precentral_sup_part_y2'], 'fr_R_surf_S_suborbital': df['fr_R_surf_S_suborbital_y2'], 'fr_R_surf_S_subparietal': df['fr_R_surf_S_subparietal_y2'], 'fr_R_surf_S_temporal_inf': df['fr_R_surf_S_temporal_inf_y2'], 'fr_R_surf_S_temporal_sup': df['fr_R_surf_S_temporal_sup_y2'], 'fr_R_surf_S_temporal_transverse': df['fr_R_surf_S_temporal_transverse_y2'],'fr_R_thick_G_Ins_lgandS_cent_ins': df['fr_R_thick_G_Ins_lgandS_cent_ins_y2'], 'fr_R_thick_G_cingul_Post_ventral': df['fr_R_thick_G_cingul_Post_ventral_y2'], 'fr_R_thick_G_cuneus': df['fr_R_thick_G_cuneus_y2'], 'fr_R_thick_G_front_inf_Opercular': df['fr_R_thick_G_front_inf_Opercular_y2'], 'fr_R_thick_G_front_inf_Orbital': df['fr_R_thick_G_front_inf_Orbital_y2'], 'fr_R_thick_G_front_inf_Triangul': df['fr_R_thick_G_front_inf_Triangul_y2'], 'fr_R_thick_G_front_middle': df['fr_R_thick_G_front_middle_y2'], 'fr_R_thick_G_front_sup': df['fr_R_thick_G_front_sup_y2'], 'fr_R_thick_G_insular_short': df['fr_R_thick_G_insular_short_y2'], 'fr_R_thick_G_oc_temp_lat_fusifor': df['fr_R_thick_G_oc_temp_lat_fusifor_y2'], 'fr_R_thick_G_oc_temp_med_Lingual': df['fr_R_thick_G_oc_temp_med_Lingual_y2'], 'fr_R_thick_G_oc_temp_med_Parahip': df['fr_R_thick_G_oc_temp_med_Parahip_y2'], 'fr_R_thick_G_occipital_middle': df['fr_R_thick_G_occipital_middle_y2'], 'fr_R_thick_G_occipital_sup': df['fr_R_thick_G_occipital_sup_y2'], 'fr_R_thick_G_orbital': df['fr_R_thick_G_orbital_y2'], 'fr_R_thick_G_pariet_inf_Angular': df['fr_R_thick_G_pariet_inf_Angular_y2'], 'fr_R_thick_G_pariet_inf_Supramar': df['fr_R_thick_G_pariet_inf_Supramar_y2'], 'fr_R_thick_G_parietal_sup': df['fr_R_thick_G_parietal_sup_y2'], 'fr_R_thick_G_postcentral': df['fr_R_thick_G_postcentral_y2'], 'fr_R_thick_G_precentral': df['fr_R_thick_G_precentral_y2'], 'fr_R_thick_G_precuneus': df['fr_R_thick_G_precuneus_y2'], 'fr_R_thick_G_rectus': df['fr_R_thick_G_rectus_y2'], 'fr_R_thick_G_temp_sup_G_T_transv': df['fr_R_thick_G_temp_sup_G_T_transv_y2'], 'fr_R_thick_G_temp_sup_Lateral': df['fr_R_thick_G_temp_sup_Lateral_y2'], 'fr_R_thick_G_temp_sup_Plan_polar': df['fr_R_thick_G_temp_sup_Plan_polar_y2'], 'fr_R_thick_G_temp_sup_Plan_tempo': df['fr_R_thick_G_temp_sup_Plan_tempo_y2'], 'fr_R_thick_G_temporal_inf': df['fr_R_thick_G_temporal_inf_y2'], 'fr_R_thick_G_temporal_middle': df['fr_R_thick_G_temporal_middle_y2'], 'fr_R_thick_GandS_cingul_Ant': df['fr_R_thick_GandS_cingul_Ant_y2'], 'fr_R_thick_GandS_cingul_Mid_Post': df['fr_R_thick_GandS_cingul_Mid_Post_y2'], 'fr_R_thick_GandS_frontomargin': df['fr_R_thick_GandS_frontomargin_y2'], 'fr_R_thick_GandS_occipital_inf': df['fr_R_thick_GandS_occipital_inf_y2'], 'fr_R_thick_GandS_paracentral': df['fr_R_thick_GandS_paracentral_y2'], 'fr_R_thick_GandS_subcentral': df['fr_R_thick_GandS_subcentral_y2'], 'fr_R_thick_GandS_transv_frontopol': df['fr_R_thick_GandS_transv_frontopol_y2'], 'fr_R_thick_Lat_Fis_ant_Horizont': df['fr_R_thick_Lat_Fis_ant_Horizont_y2'], 'fr_R_thick_Lat_Fis_post': df['fr_R_thick_Lat_Fis_post_y2'], 'fr_R_thick_Pole_occipital': df['fr_R_thick_Pole_occipital_y2'], 'fr_R_thick_Pole_temporal': df['fr_R_thick_Pole_temporal_y2'], 'fr_R_thick_S_calcarine': df['fr_R_thick_S_calcarine_y2'], 'fr_R_thick_S_central': df['fr_R_thick_S_central_y2'], 'fr_R_thick_S_cingul_Marginalis': df['fr_R_thick_S_cingul_Marginalis_y2'], 'fr_R_thick_S_circular_insula_ant': df['fr_R_thick_S_circular_insula_ant_y2'], 'fr_R_thick_S_circular_insula_inf': df['fr_R_thick_S_circular_insula_inf_y2'], 'fr_R_thick_S_circular_insula_sup': df['fr_R_thick_S_circular_insula_sup_y2'], 'fr_R_thick_S_collat_transv_ant': df['fr_R_thick_S_collat_transv_ant_y2'], 'fr_R_thick_S_collat_transv_post': df['fr_R_thick_S_collat_transv_post_y2'], 'fr_R_thick_S_front_inf': df['fr_R_thick_S_front_inf_y2'], 'fr_R_thick_S_front_middle': df['fr_R_thick_S_front_middle_y2'], 'fr_R_thick_S_front_sup': df['fr_R_thick_S_front_sup_y2'], 'fr_R_thick_S_interm_prim_Jensen': df['fr_R_thick_S_interm_prim_Jensen_y2'], 'fr_R_thick_S_intraparietandP_trans': df['fr_R_thick_S_intraparietandP_trans_y2'], 'fr_R_thick_S_oc_middleandLunatus': df['fr_R_thick_S_oc_middleandLunatus_y2'], 'fr_R_thick_S_oc_temp_lat': df['fr_R_thick_S_oc_temp_lat_y2'], 'fr_R_thick_S_oc_temp_medandLingual': df['fr_R_thick_S_oc_temp_medandLingual_y2'], 'fr_R_thick_S_orbital_H_Shaped': df['fr_R_thick_S_orbital_H_Shaped_y2'], 'fr_R_thick_S_orbital_lateral': df['fr_R_thick_S_orbital_lateral_y2'], 'fr_R_thick_S_orbital_med_olfact': df['fr_R_thick_S_orbital_med_olfact_y2'], 'fr_R_thick_S_parieto_occipital': df['fr_R_thick_S_parieto_occipital_y2'], 'fr_R_thick_S_pericallosal': df['fr_R_thick_S_pericallosal_y2'], 'fr_R_thick_S_postcentral': df['fr_R_thick_S_postcentral_y2'], 'fr_R_thick_S_precentral_inf_part': df['fr_R_thick_S_precentral_inf_part_y2'], 'fr_R_thick_S_precentral_sup_part': df['fr_R_thick_S_precentral_sup_part_y2'], 'fr_R_thick_S_suborbital': df['fr_R_thick_S_suborbital_y2'], 'fr_R_thick_S_subparietal': df['fr_R_thick_S_subparietal_y2'], 'fr_R_thick_S_temporal_inf': df['fr_R_thick_S_temporal_inf_y2'], 'fr_R_thick_S_temporal_sup': df['fr_R_thick_S_temporal_sup_y2'], 'fr_R_thick_S_temporal_transverse': df['fr_R_thick_S_temporal_transverse_y2'], 'fr_L_surf_G_Ins_lgandS_cent_ins': df['fr_L_surf_G_Ins_lgandS_cent_ins_y2'], 'fr_L_surf_G_cingul_Post_ventral': df['fr_L_surf_G_cingul_Post_ventral_y2'], 'fr_L_surf_G_cuneus': df['fr_L_surf_G_cuneus_y2'], 'fr_L_surf_G_front_inf_Opercular': df['fr_L_surf_G_front_inf_Opercular_y2'], 'fr_L_surf_G_front_inf_Orbital': df['fr_L_surf_G_front_inf_Orbital_y2'], 'fr_L_surf_G_front_inf_Triangul': df['fr_L_surf_G_front_inf_Triangul_y2'], 'fr_L_surf_G_front_middle': df['fr_L_surf_G_front_middle_y2'], 'fr_L_surf_G_front_sup': df['fr_L_surf_G_front_sup_y2'], 'fr_L_surf_G_insular_short': df['fr_L_surf_G_insular_short_y2'], 'fr_L_surf_G_oc_temp_lat_fusifor': df['fr_L_surf_G_oc_temp_lat_fusifor_y2'], 'fr_L_surf_G_oc_temp_med_Lingual': df['fr_L_surf_G_oc_temp_med_Lingual_y2'], 'fr_L_surf_G_oc_temp_med_Parahip': df['fr_L_surf_G_oc_temp_med_Parahip_y2'], 'fr_L_surf_G_occipital_middle': df['fr_L_surf_G_occipital_middle_y2'], 'fr_L_surf_G_occipital_sup': df['fr_L_surf_G_occipital_sup_y2'], 'fr_L_surf_G_orbital': df['fr_L_surf_G_orbital_y2'], 'fr_L_surf_G_pariet_inf_Angular': df['fr_L_surf_G_pariet_inf_Angular_y2'], 'fr_L_surf_G_pariet_inf_Supramar': df['fr_L_surf_G_pariet_inf_Supramar_y2'], 'fr_L_surf_G_parietal_sup': df['fr_L_surf_G_parietal_sup_y2'], 'fr_L_surf_G_postcentral': df['fr_L_surf_G_postcentral_y2'], 'fr_L_surf_G_precentral': df['fr_L_surf_G_precentral_y2'], 'fr_L_surf_G_precuneus': df['fr_L_surf_G_precuneus_y2'], 'fr_L_surf_G_rectus': df['fr_L_surf_G_rectus_y2'], 'fr_L_surf_G_temp_sup_G_T_transv': df['fr_L_surf_G_temp_sup_G_T_transv_y2'], 'fr_L_surf_G_temp_sup_Lateral': df['fr_L_surf_G_temp_sup_Lateral_y2'], 'fr_L_surf_G_temp_sup_Plan_polar': df['fr_L_surf_G_temp_sup_Plan_polar_y2'], 'fr_L_surf_G_temp_sup_Plan_tempo': df['fr_L_surf_G_temp_sup_Plan_tempo_y2'], 'fr_L_surf_G_temporal_inf': df['fr_L_surf_G_temporal_inf_y2'], 'fr_L_surf_G_temporal_middle': df['fr_L_surf_G_temporal_middle_y2'], 'fr_L_surf_GandS_cingul_Ant': df['fr_L_surf_GandS_cingul_Ant_y2'], 'fr_L_surf_GandS_cingul_Mid_Post': df['fr_L_surf_GandS_cingul_Mid_Post_y2'], 'fr_L_surf_GandS_frontomargin': df['fr_L_surf_GandS_frontomargin_y2'], 'fr_L_surf_GandS_occipital_inf': df['fr_L_surf_GandS_occipital_inf_y2'], 'fr_L_surf_GandS_paracentral': df['fr_L_surf_GandS_paracentral_y2'], 'fr_L_surf_GandS_subcentral': df['fr_L_surf_GandS_subcentral_y2'], 'fr_L_surf_GandS_transv_frontopol': df['fr_L_surf_GandS_transv_frontopol_y2'], 'fr_L_surf_Lat_Fis_ant_Horizont': df['fr_L_surf_Lat_Fis_ant_Horizont_y2'], 'fr_L_surf_Lat_Fis_post': df['fr_L_surf_Lat_Fis_post_y2'], 'fr_L_surf_Pole_occipital': df['fr_L_surf_Pole_occipital_y2'], 'fr_L_surf_Pole_temporal': df['fr_L_surf_Pole_temporal_y2'], 'fr_L_surf_S_calcarine': df['fr_L_surf_S_calcarine_y2'], 'fr_L_surf_S_central': df['fr_L_surf_S_central_y2'], 'fr_L_surf_S_cingul_Marginalis': df['fr_L_surf_S_cingul_Marginalis_y2'], 'fr_L_surf_S_circular_insula_ant': df['fr_L_surf_S_circular_insula_ant_y2'], 'fr_L_surf_S_circular_insula_inf': df['fr_L_surf_S_circular_insula_inf_y2'], 'fr_L_surf_S_circular_insula_sup': df['fr_L_surf_S_circular_insula_sup_y2'], 'fr_L_surf_S_collat_transv_ant': df['fr_L_surf_S_collat_transv_ant_y2'], 'fr_L_surf_S_collat_transv_post': df['fr_L_surf_S_collat_transv_post_y2'], 'fr_L_surf_S_front_inf': df['fr_L_surf_S_front_inf_y2'], 'fr_L_surf_S_front_middle': df['fr_L_surf_S_front_middle_y2'], 'fr_L_surf_S_front_sup': df['fr_L_surf_S_front_sup_y2'],  'fr_L_surf_S_intraparietandP_trans': df['fr_L_surf_S_intraparietandP_trans_y2'], 'fr_L_surf_S_oc_middleandLunatus': df['fr_L_surf_S_oc_middleandLunatus_y2'], 'fr_L_surf_S_oc_temp_lat': df['fr_L_surf_S_oc_temp_lat_y2'], 'fr_L_surf_S_oc_temp_medandLingual': df['fr_L_surf_S_oc_temp_medandLingual_y2'], 'fr_L_surf_S_orbital_H_Shaped': df['fr_L_surf_S_orbital_H_Shaped_y2'], 'fr_L_surf_S_orbital_lateral': df['fr_L_surf_S_orbital_lateral_y2'], 'fr_L_surf_S_orbital_med_olfact': df['fr_L_surf_S_orbital_med_olfact_y2'], 'fr_L_surf_S_parieto_occipital': df['fr_L_surf_S_parieto_occipital_y2'], 'fr_L_surf_S_postcentral': df['fr_L_surf_S_postcentral_y2'], 'fr_L_surf_S_precentral_inf_part': df['fr_L_surf_S_precentral_inf_part_y2'], 'fr_L_surf_S_precentral_sup_part': df['fr_L_surf_S_precentral_sup_part_y2'], 'fr_L_surf_S_suborbital': df['fr_L_surf_S_suborbital_y2'], 'fr_L_surf_S_subparietal': df['fr_L_surf_S_subparietal_y2'], 'fr_L_surf_S_temporal_inf': df['fr_L_surf_S_temporal_inf_y2'], 'fr_L_surf_S_temporal_sup': df['fr_L_surf_S_temporal_sup_y2']}
		#frame_free3 = {'handlat': df['lat_manual'],'apoe': df['apoe'], 'y_last':df['ultimavisita'], 'dx_last': df['ultimodx'], 'dx_visita': df['dx_corto_visita3'], 'age_lastdx': df['edad_ultimodx'], 'age': df['edad_visita3'], 'sex': df['sexo'], 'free_R_Thal': df['fr_Right_Thalamus_Proper_y3'], 'free_L_Thal': df['fr_Left_Thalamus_Proper_y3'],'free_R_Puta': df['fr_Right_Putamen_y3'], 'free_L_Puta': df['fr_Left_Putamen_y3'],'free_R_Amyg': df['fr_Right_Amygdala_y3'], 'free_L_Amyg': df['fr_Left_Amygdala_y3'],'free_R_Pall': df['fr_Right_Pallidum_y3'], 'free_L_Pall': df['fr_Left_Pallidum_y3'],'free_R_Caud': df['fr_Right_Caudate_y3'], 'free_L_Caud': df['fr_Left_Caudate_y3'],'free_R_Hipp': df['fr_Right_Hippocampus_y3'], 'free_L_Hipp': df['fr_Left_Hippocampus_y3'],'free_R_Accu': df['fr_Right_Accumbens_area_y3'], 'free_L_Accu': df['fr_Left_Accumbens_area_y3'], 'fr_BrainSegVol': df['fr_BrainSegVol_y3'], 'fr_BrainSegVolNotVent':df['fr_BrainSegVolNotVent_y3'], 'fr_BrainSegVol_to_eTIV': df['fr_BrainSegVol_to_eTIV_y3']}
		frame_free3 = {'education': df['nivel_educativo'],'handlat': df['lat_manual'],'apoe': df['apoe'], 'nvisita':df['nvisita3'] ,'y_last':df['ultimavisita'], 'dx_last': df['ultimodx'], 'dx_visita': df['dx_corto_visita3'], 'age_lastdx': df['edad_ultimodx'], 'age': df['edad_visita3'], 'sex': df['sexo'], 'free_R_Thal': df['fr_Right_Thalamus_Proper_y3'], 'free_L_Thal': df['fr_Left_Thalamus_Proper_y3'],'free_R_Puta': df['fr_Right_Putamen_y3'], 'free_L_Puta': df['fr_Left_Putamen_y3'],'free_R_Amyg': df['fr_Right_Amygdala_y3'], 'free_L_Amyg': df['fr_Left_Amygdala_y3'],'free_R_Pall': df['fr_Right_Pallidum_y3'], 'free_L_Pall': df['fr_Left_Pallidum_y3'],'free_R_Caud': df['fr_Right_Caudate_y3'], 'free_L_Caud': df['fr_Left_Caudate_y3'],'free_R_Hipp': df['fr_Right_Hippocampus_y3'], 'free_L_Hipp': df['fr_Left_Hippocampus_y3'],'free_R_Accu': df['fr_Right_Accumbens_area_y3'], 'free_L_Accu': df['fr_Left_Accumbens_area_y3'], 'fr_BrainSegVol': df['fr_BrainSegVol_y3'], 'fr_BrainSegVolNotVent':df['fr_BrainSegVolNotVent_y3'], 'fr_BrainSegVol_to_eTIV': df['fr_BrainSegVol_to_eTIV_y3'],  'fr_R_surf_G_Ins_lgandS_cent_ins': df['fr_R_surf_G_Ins_lgandS_cent_ins_y3'], 'fr_R_surf_G_cingul_Post_ventral': df['fr_R_surf_G_cingul_Post_ventral_y3'], 'fr_R_surf_G_cuneus': df['fr_R_surf_G_cuneus_y3'], 'fr_R_surf_G_front_inf_Opercular': df['fr_R_surf_G_front_inf_Opercular_y3'], 'fr_R_surf_G_front_inf_Orbital': df['fr_R_surf_G_front_inf_Orbital_y3'], 'fr_R_surf_G_front_inf_Triangul': df['fr_R_surf_G_front_inf_Triangul_y3'], 'fr_R_surf_G_front_middle': df['fr_R_surf_G_front_middle_y3'], 'fr_R_surf_G_front_sup': df['fr_R_surf_G_front_sup_y3'], 'fr_R_surf_G_insular_short': df['fr_R_surf_G_insular_short_y3'], 'fr_R_surf_G_oc_temp_lat_fusifor': df['fr_R_surf_G_oc_temp_lat_fusifor_y3'], 'fr_R_surf_G_oc_temp_med_Lingual': df['fr_R_surf_G_oc_temp_med_Lingual_y3'], 'fr_R_surf_G_oc_temp_med_Parahip': df['fr_R_surf_G_oc_temp_med_Parahip_y3'], 'fr_R_surf_G_occipital_middle': df['fr_R_surf_G_occipital_middle_y3'], 'fr_R_surf_G_occipital_sup': df['fr_R_surf_G_occipital_sup_y3'], 'fr_R_surf_G_orbital': df['fr_R_surf_G_orbital_y3'], 'fr_R_surf_G_pariet_inf_Angular': df['fr_R_surf_G_pariet_inf_Angular_y3'], 'fr_R_surf_G_pariet_inf_Supramar': df['fr_R_surf_G_pariet_inf_Supramar_y3'], 'fr_R_surf_G_parietal_sup': df['fr_R_surf_G_parietal_sup_y3'], 'fr_R_surf_G_postcentral': df['fr_R_surf_G_postcentral_y3'], 'fr_R_surf_G_precentral': df['fr_R_surf_G_precentral_y3'], 'fr_R_surf_G_precuneus': df['fr_R_surf_G_precuneus_y3'], 'fr_R_surf_G_rectus': df['fr_R_surf_G_rectus_y3'], 'fr_R_surf_G_temp_sup_G_T_transv': df['fr_R_surf_G_temp_sup_G_T_transv_y3'], 'fr_R_surf_G_temp_sup_Lateral': df['fr_R_surf_G_temp_sup_Lateral_y3'], 'fr_R_surf_G_temp_sup_Plan_polar': df['fr_R_surf_G_temp_sup_Plan_polar_y3'], 'fr_R_surf_G_temp_sup_Plan_tempo': df['fr_R_surf_G_temp_sup_Plan_tempo_y3'], 'fr_R_surf_G_temporal_inf': df['fr_R_surf_G_temporal_inf_y3'], 'fr_R_surf_G_temporal_middle': df['fr_R_surf_G_temporal_middle_y3'], 'fr_R_surf_GandS_cingul_Ant': df['fr_R_surf_GandS_cingul_Ant_y3'], 'fr_R_surf_GandS_cingul_Mid_Post': df['fr_R_surf_GandS_cingul_Mid_Post_y3'], 'fr_R_surf_GandS_frontomargin': df['fr_R_surf_GandS_frontomargin_y3'], 'fr_R_surf_GandS_occipital_inf': df['fr_R_surf_GandS_occipital_inf_y3'], 'fr_R_surf_GandS_paracentral': df['fr_R_surf_GandS_paracentral_y3'], 'fr_R_surf_GandS_subcentral': df['fr_R_surf_GandS_subcentral_y3'], 'fr_R_surf_GandS_transv_frontopol': df['fr_R_surf_GandS_transv_frontopol_y3'], 'fr_R_surf_Lat_Fis_ant_Horizont': df['fr_R_surf_Lat_Fis_ant_Horizont_y3'], 'fr_R_surf_Lat_Fis_post': df['fr_R_surf_Lat_Fis_post_y3'], 'fr_R_surf_Pole_occipital': df['fr_R_surf_Pole_occipital_y3'], 'fr_R_surf_Pole_temporal': df['fr_R_surf_Pole_temporal_y3'], 'fr_R_surf_S_calcarine': df['fr_R_surf_S_calcarine_y3'], 'fr_R_surf_S_central': df['fr_R_surf_S_central_y3'], 'fr_R_surf_S_cingul_Marginalis': df['fr_R_surf_S_cingul_Marginalis_y3'], 'fr_R_surf_S_circular_insula_ant': df['fr_R_surf_S_circular_insula_ant_y3'], 'fr_R_surf_S_circular_insula_inf': df['fr_R_surf_S_circular_insula_inf_y3'], 'fr_R_surf_S_circular_insula_sup': df['fr_R_surf_S_circular_insula_sup_y3'], 'fr_R_surf_S_collat_transv_ant': df['fr_R_surf_S_collat_transv_ant_y3'], 'fr_R_surf_S_collat_transv_post': df['fr_R_surf_S_collat_transv_post_y3'], 'fr_R_surf_S_front_inf': df['fr_R_surf_S_front_inf_y3'], 'fr_R_surf_S_front_middle': df['fr_R_surf_S_front_middle_y3'], 'fr_R_surf_S_front_sup': df['fr_R_surf_S_front_sup_y3'], 'fr_R_surf_S_interm_prim_Jensen': df['fr_R_surf_S_interm_prim_Jensen_y3'], 'fr_R_surf_S_intraparietandP_trans': df['fr_R_surf_S_intraparietandP_trans_y3'], 'fr_R_surf_S_oc_middleandLunatus': df['fr_R_surf_S_oc_middleandLunatus_y3'], 'fr_R_surf_S_oc_temp_lat': df['fr_R_surf_S_oc_temp_lat_y3'], 'fr_R_surf_S_oc_temp_medandLingual': df['fr_R_surf_S_oc_temp_medandLingual_y3'], 'fr_R_surf_S_orbital_H_Shaped': df['fr_R_surf_S_orbital_H_Shaped_y3'], 'fr_R_surf_S_orbital_lateral': df['fr_R_surf_S_orbital_lateral_y3'], 'fr_R_surf_S_orbital_med_olfact': df['fr_R_surf_S_orbital_med_olfact_y3'], 'fr_R_surf_S_parieto_occipital': df['fr_R_surf_S_parieto_occipital_y3'], 'fr_R_surf_S_pericallosal': df['fr_R_surf_S_pericallosal_y3'], 'fr_R_surf_S_postcentral': df['fr_R_surf_S_postcentral_y3'], 'fr_R_surf_S_precentral_inf_part': df['fr_R_surf_S_precentral_inf_part_y3'], 'fr_R_surf_S_precentral_sup_part': df['fr_R_surf_S_precentral_sup_part_y3'], 'fr_R_surf_S_suborbital': df['fr_R_surf_S_suborbital_y3'], 'fr_R_surf_S_subparietal': df['fr_R_surf_S_subparietal_y3'], 'fr_R_surf_S_temporal_inf': df['fr_R_surf_S_temporal_inf_y3'], 'fr_R_surf_S_temporal_sup': df['fr_R_surf_S_temporal_sup_y3'], 'fr_R_surf_S_temporal_transverse': df['fr_R_surf_S_temporal_transverse_y3'], 'fr_R_thick_G_Ins_lgandS_cent_ins': df['fr_R_thick_G_Ins_lgandS_cent_ins_y3'], 'fr_R_thick_G_cingul_Post_ventral': df['fr_R_thick_G_cingul_Post_ventral_y3'], 'fr_R_thick_G_cuneus': df['fr_R_thick_G_cuneus_y3'], 'fr_R_thick_G_front_inf_Opercular': df['fr_R_thick_G_front_inf_Opercular_y3'], 'fr_R_thick_G_front_inf_Orbital': df['fr_R_thick_G_front_inf_Orbital_y3'], 'fr_R_thick_G_front_inf_Triangul': df['fr_R_thick_G_front_inf_Triangul_y3'], 'fr_R_thick_G_front_middle': df['fr_R_thick_G_front_middle_y3'], 'fr_R_thick_G_front_sup': df['fr_R_thick_G_front_sup_y3'], 'fr_R_thick_G_insular_short': df['fr_R_thick_G_insular_short_y3'], 'fr_R_thick_G_oc_temp_lat_fusifor': df['fr_R_thick_G_oc_temp_lat_fusifor_y3'], 'fr_R_thick_G_oc_temp_med_Lingual': df['fr_R_thick_G_oc_temp_med_Lingual_y3'], 'fr_R_thick_G_oc_temp_med_Parahip': df['fr_R_thick_G_oc_temp_med_Parahip_y3'], 'fr_R_thick_G_occipital_middle': df['fr_R_thick_G_occipital_middle_y3'], 'fr_R_thick_G_occipital_sup': df['fr_R_thick_G_occipital_sup_y3'], 'fr_R_thick_G_orbital': df['fr_R_thick_G_orbital_y3'], 'fr_R_thick_G_pariet_inf_Angular': df['fr_R_thick_G_pariet_inf_Angular_y3'], 'fr_R_thick_G_pariet_inf_Supramar': df['fr_R_thick_G_pariet_inf_Supramar_y3'], 'fr_R_thick_G_parietal_sup': df['fr_R_thick_G_parietal_sup_y3'], 'fr_R_thick_G_postcentral': df['fr_R_thick_G_postcentral_y3'], 'fr_R_thick_G_precentral': df['fr_R_thick_G_precentral_y3'], 'fr_R_thick_G_precuneus': df['fr_R_thick_G_precuneus_y3'], 'fr_R_thick_G_rectus': df['fr_R_thick_G_rectus_y3'], 'fr_R_thick_G_temp_sup_G_T_transv': df['fr_R_thick_G_temp_sup_G_T_transv_y3'], 'fr_R_thick_G_temp_sup_Lateral': df['fr_R_thick_G_temp_sup_Lateral_y3'], 'fr_R_thick_G_temp_sup_Plan_polar': df['fr_R_thick_G_temp_sup_Plan_polar_y3'], 'fr_R_thick_G_temp_sup_Plan_tempo': df['fr_R_thick_G_temp_sup_Plan_tempo_y3'], 'fr_R_thick_G_temporal_inf': df['fr_R_thick_G_temporal_inf_y3'], 'fr_R_thick_G_temporal_middle': df['fr_R_thick_G_temporal_middle_y3'], 'fr_R_thick_GandS_cingul_Ant': df['fr_R_thick_GandS_cingul_Ant_y3'], 'fr_R_thick_GandS_cingul_Mid_Post': df['fr_R_thick_GandS_cingul_Mid_Post_y3'], 'fr_R_thick_GandS_frontomargin': df['fr_R_thick_GandS_frontomargin_y3'], 'fr_R_thick_GandS_occipital_inf': df['fr_R_thick_GandS_occipital_inf_y3'], 'fr_R_thick_GandS_paracentral': df['fr_R_thick_GandS_paracentral_y3'], 'fr_R_thick_GandS_subcentral': df['fr_R_thick_GandS_subcentral_y3'], 'fr_R_thick_GandS_transv_frontopol': df['fr_R_thick_GandS_transv_frontopol_y3'], 'fr_R_thick_Lat_Fis_ant_Horizont': df['fr_R_thick_Lat_Fis_ant_Horizont_y3'], 'fr_R_thick_Lat_Fis_post': df['fr_R_thick_Lat_Fis_post_y3'], 'fr_R_thick_Pole_occipital': df['fr_R_thick_Pole_occipital_y3'], 'fr_R_thick_Pole_temporal': df['fr_R_thick_Pole_temporal_y3'], 'fr_R_thick_S_calcarine': df['fr_R_thick_S_calcarine_y3'], 'fr_R_thick_S_central': df['fr_R_thick_S_central_y3'], 'fr_R_thick_S_cingul_Marginalis': df['fr_R_thick_S_cingul_Marginalis_y3'], 'fr_R_thick_S_circular_insula_ant': df['fr_R_thick_S_circular_insula_ant_y3'], 'fr_R_thick_S_circular_insula_inf': df['fr_R_thick_S_circular_insula_inf_y3'], 'fr_R_thick_S_circular_insula_sup': df['fr_R_thick_S_circular_insula_sup_y3'], 'fr_R_thick_S_collat_transv_ant': df['fr_R_thick_S_collat_transv_ant_y3'], 'fr_R_thick_S_collat_transv_post': df['fr_R_thick_S_collat_transv_post_y3'], 'fr_R_thick_S_front_inf': df['fr_R_thick_S_front_inf_y3'], 'fr_R_thick_S_front_middle': df['fr_R_thick_S_front_middle_y3'], 'fr_R_thick_S_front_sup': df['fr_R_thick_S_front_sup_y3'], 'fr_R_thick_S_interm_prim_Jensen': df['fr_R_thick_S_interm_prim_Jensen_y3'], 'fr_R_thick_S_intraparietandP_trans': df['fr_R_thick_S_intraparietandP_trans_y3'], 'fr_R_thick_S_oc_middleandLunatus': df['fr_R_thick_S_oc_middleandLunatus_y3'], 'fr_R_thick_S_oc_temp_lat': df['fr_R_thick_S_oc_temp_lat_y3'], 'fr_R_thick_S_oc_temp_medandLingual': df['fr_R_thick_S_oc_temp_medandLingual_y3'], 'fr_R_thick_S_orbital_H_Shaped': df['fr_R_thick_S_orbital_H_Shaped_y3'], 'fr_R_thick_S_orbital_lateral': df['fr_R_thick_S_orbital_lateral_y3'], 'fr_R_thick_S_orbital_med_olfact': df['fr_R_thick_S_orbital_med_olfact_y3'], 'fr_R_thick_S_parieto_occipital': df['fr_R_thick_S_parieto_occipital_y3'], 'fr_R_thick_S_pericallosal': df['fr_R_thick_S_pericallosal_y3'], 'fr_R_thick_S_postcentral': df['fr_R_thick_S_postcentral_y3'], 'fr_R_thick_S_precentral_inf_part': df['fr_R_thick_S_precentral_inf_part_y3'], 'fr_R_thick_S_precentral_sup_part': df['fr_R_thick_S_precentral_sup_part_y3'], 'fr_R_thick_S_suborbital': df['fr_R_thick_S_suborbital_y3'], 'fr_R_thick_S_subparietal': df['fr_R_thick_S_subparietal_y3'], 'fr_R_thick_S_temporal_inf': df['fr_R_thick_S_temporal_inf_y3'], 'fr_R_thick_S_temporal_sup': df['fr_R_thick_S_temporal_sup_y3'], 'fr_R_thick_S_temporal_transverse': df['fr_R_thick_S_temporal_transverse_y3'], 'fr_L_surf_G_Ins_lgandS_cent_ins': df['fr_L_surf_G_Ins_lgandS_cent_ins_y3'], 'fr_L_surf_G_cingul_Post_ventral': df['fr_L_surf_G_cingul_Post_ventral_y3'], 'fr_L_surf_G_cuneus': df['fr_L_surf_G_cuneus_y3'], 'fr_L_surf_G_front_inf_Opercular': df['fr_L_surf_G_front_inf_Opercular_y3'], 'fr_L_surf_G_front_inf_Orbital': df['fr_L_surf_G_front_inf_Orbital_y3'], 'fr_L_surf_G_front_inf_Triangul': df['fr_L_surf_G_front_inf_Triangul_y3'], 'fr_L_surf_G_front_middle': df['fr_L_surf_G_front_middle_y3'], 'fr_L_surf_G_front_sup': df['fr_L_surf_G_front_sup_y3'], 'fr_L_surf_G_insular_short': df['fr_L_surf_G_insular_short_y3'], 'fr_L_surf_G_oc_temp_lat_fusifor': df['fr_L_surf_G_oc_temp_lat_fusifor_y3'], 'fr_L_surf_G_oc_temp_med_Lingual': df['fr_L_surf_G_oc_temp_med_Lingual_y3'], 'fr_L_surf_G_oc_temp_med_Parahip': df['fr_L_surf_G_oc_temp_med_Parahip_y3'], 'fr_L_surf_G_occipital_middle': df['fr_L_surf_G_occipital_middle_y3'], 'fr_L_surf_G_occipital_sup': df['fr_L_surf_G_occipital_sup_y3'], 'fr_L_surf_G_orbital': df['fr_L_surf_G_orbital_y3'], 'fr_L_surf_G_pariet_inf_Angular': df['fr_L_surf_G_pariet_inf_Angular_y3'], 'fr_L_surf_G_pariet_inf_Supramar': df['fr_L_surf_G_pariet_inf_Supramar_y3'], 'fr_L_surf_G_parietal_sup': df['fr_L_surf_G_parietal_sup_y3'], 'fr_L_surf_G_postcentral': df['fr_L_surf_G_postcentral_y3'], 'fr_L_surf_G_precentral': df['fr_L_surf_G_precentral_y3'], 'fr_L_surf_G_precuneus': df['fr_L_surf_G_precuneus_y3'], 'fr_L_surf_G_rectus': df['fr_L_surf_G_rectus_y3'], 'fr_L_surf_G_temp_sup_G_T_transv': df['fr_L_surf_G_temp_sup_G_T_transv_y3'], 'fr_L_surf_G_temp_sup_Lateral': df['fr_L_surf_G_temp_sup_Lateral_y3'], 'fr_L_surf_G_temp_sup_Plan_polar': df['fr_L_surf_G_temp_sup_Plan_polar_y3'], 'fr_L_surf_G_temp_sup_Plan_tempo': df['fr_L_surf_G_temp_sup_Plan_tempo_y3'], 'fr_L_surf_G_temporal_inf': df['fr_L_surf_G_temporal_inf_y3'], 'fr_L_surf_G_temporal_middle': df['fr_L_surf_G_temporal_middle_y3'], 'fr_L_surf_GandS_cingul_Ant': df['fr_L_surf_GandS_cingul_Ant_y3'], 'fr_L_surf_GandS_cingul_Mid_Post': df['fr_L_surf_GandS_cingul_Mid_Post_y3'], 'fr_L_surf_GandS_frontomargin': df['fr_L_surf_GandS_frontomargin_y3'], 'fr_L_surf_GandS_occipital_inf': df['fr_L_surf_GandS_occipital_inf_y3'], 'fr_L_surf_GandS_paracentral': df['fr_L_surf_GandS_paracentral_y3'], 'fr_L_surf_GandS_subcentral': df['fr_L_surf_GandS_subcentral_y3'], 'fr_L_surf_GandS_transv_frontopol': df['fr_L_surf_GandS_transv_frontopol_y3'], 'fr_L_surf_Lat_Fis_ant_Horizont': df['fr_L_surf_Lat_Fis_ant_Horizont_y3'], 'fr_L_surf_Lat_Fis_post': df['fr_L_surf_Lat_Fis_post_y3'], 'fr_L_surf_Pole_occipital': df['fr_L_surf_Pole_occipital_y3'], 'fr_L_surf_Pole_temporal': df['fr_L_surf_Pole_temporal_y3'], 'fr_L_surf_S_calcarine': df['fr_L_surf_S_calcarine_y3'], 'fr_L_surf_S_central': df['fr_L_surf_S_central_y3'], 'fr_L_surf_S_cingul_Marginalis': df['fr_L_surf_S_cingul_Marginalis_y3'], 'fr_L_surf_S_circular_insula_ant': df['fr_L_surf_S_circular_insula_ant_y3'], 'fr_L_surf_S_circular_insula_inf': df['fr_L_surf_S_circular_insula_inf_y3'], 'fr_L_surf_S_circular_insula_sup': df['fr_L_surf_S_circular_insula_sup_y3'], 'fr_L_surf_S_collat_transv_ant': df['fr_L_surf_S_collat_transv_ant_y3'], 'fr_L_surf_S_collat_transv_post': df['fr_L_surf_S_collat_transv_post_y3'], 'fr_L_surf_S_front_inf': df['fr_L_surf_S_front_inf_y3'], 'fr_L_surf_S_front_middle': df['fr_L_surf_S_front_middle_y3'], 'fr_L_surf_S_front_sup': df['fr_L_surf_S_front_sup_y3'],  'fr_L_surf_S_intraparietandP_trans': df['fr_L_surf_S_intraparietandP_trans_y3'], 'fr_L_surf_S_oc_middleandLunatus': df['fr_L_surf_S_oc_middleandLunatus_y3'], 'fr_L_surf_S_oc_temp_lat': df['fr_L_surf_S_oc_temp_lat_y3'], 'fr_L_surf_S_oc_temp_medandLingual': df['fr_L_surf_S_oc_temp_medandLingual_y3'], 'fr_L_surf_S_orbital_H_Shaped': df['fr_L_surf_S_orbital_H_Shaped_y3'], 'fr_L_surf_S_orbital_lateral': df['fr_L_surf_S_orbital_lateral_y3'], 'fr_L_surf_S_orbital_med_olfact': df['fr_L_surf_S_orbital_med_olfact_y3'], 'fr_L_surf_S_parieto_occipital': df['fr_L_surf_S_parieto_occipital_y3'], 'fr_L_surf_S_postcentral': df['fr_L_surf_S_postcentral_y3'], 'fr_L_surf_S_precentral_inf_part': df['fr_L_surf_S_precentral_inf_part_y3'], 'fr_L_surf_S_precentral_sup_part': df['fr_L_surf_S_precentral_sup_part_y3'], 'fr_L_surf_S_suborbital': df['fr_L_surf_S_suborbital_y3'], 'fr_L_surf_S_subparietal': df['fr_L_surf_S_subparietal_y3'], 'fr_L_surf_S_temporal_inf': df['fr_L_surf_S_temporal_inf_y3'], 'fr_L_surf_S_temporal_sup': df['fr_L_surf_S_temporal_sup_y3']}
		#frame_free4 = {'handlat': df['lat_manual'],'apoe': df['apoe'], 'y_last':df['ultimavisita'], 'dx_last': df['ultimodx'], 'dx_visita': df['dx_corto_visita4'], 'age_lastdx': df['edad_ultimodx'], 'age': df['edad_visita4'], 'sex': df['sexo'], 'free_R_Thal': df['fr_Right_Thalamus_Proper_y4'], 'free_L_Thal': df['fr_Left_Thalamus_Proper_y4'],'free_R_Puta': df['fr_Right_Putamen_y4'], 'free_L_Puta': df['fr_Left_Putamen_y4'],'free_R_Amyg': df['fr_Right_Amygdala_y4'], 'free_L_Amyg': df['fr_Left_Amygdala_y4'],'free_R_Pall': df['fr_Right_Pallidum_y4'], 'free_L_Pall': df['fr_Left_Pallidum_y4'],'free_R_Caud': df['fr_Right_Caudate_y4'], 'free_L_Caud': df['fr_Left_Caudate_y4'],'free_R_Hipp': df['fr_Right_Hippocampus_y4'], 'free_L_Hipp': df['fr_Left_Hippocampus_y4'],'free_R_Accu': df['fr_Right_Accumbens_area_y4'], 'free_L_Accu': df['fr_Left_Accumbens_area_y4'], 'fr_BrainSegVol': df['fr_BrainSegVol_y4'], 'fr_BrainSegVolNotVent':df['fr_BrainSegVolNotVent_y4'], 'fr_BrainSegVol_to_eTIV': df['fr_BrainSegVol_to_eTIV_y4']}
		frame_free4 = {'education': df['nivel_educativo'],'handlat': df['lat_manual'],'apoe': df['apoe'], 'nvisita':df['nvisita4'], 'y_last':df['ultimavisita'], 'dx_last': df['ultimodx'], 'dx_visita': df['dx_corto_visita4'], 'age_lastdx': df['edad_ultimodx'], 'age': df['edad_visita4'], 'sex': df['sexo'], 'free_R_Thal': df['fr_Right_Thalamus_Proper_y4'], 'free_L_Thal': df['fr_Left_Thalamus_Proper_y4'],'free_R_Puta': df['fr_Right_Putamen_y4'], 'free_L_Puta': df['fr_Left_Putamen_y4'],'free_R_Amyg': df['fr_Right_Amygdala_y4'], 'free_L_Amyg': df['fr_Left_Amygdala_y4'],'free_R_Pall': df['fr_Right_Pallidum_y4'], 'free_L_Pall': df['fr_Left_Pallidum_y4'],'free_R_Caud': df['fr_Right_Caudate_y4'], 'free_L_Caud': df['fr_Left_Caudate_y4'],'free_R_Hipp': df['fr_Right_Hippocampus_y4'], 'free_L_Hipp': df['fr_Left_Hippocampus_y4'],'free_R_Accu': df['fr_Right_Accumbens_area_y4'], 'free_L_Accu': df['fr_Left_Accumbens_area_y4'], 'fr_BrainSegVol': df['fr_BrainSegVol_y4'], 'fr_BrainSegVolNotVent':df['fr_BrainSegVolNotVent_y4'], 'fr_BrainSegVol_to_eTIV': df['fr_BrainSegVol_to_eTIV_y4'],   'fr_R_surf_G_Ins_lgandS_cent_ins': df['fr_R_surf_G_Ins_lgandS_cent_ins_y4'], 'fr_R_surf_G_cingul_Post_ventral': df['fr_R_surf_G_cingul_Post_ventral_y4'], 'fr_R_surf_G_cuneus': df['fr_R_surf_G_cuneus_y4'], 'fr_R_surf_G_front_inf_Opercular': df['fr_R_surf_G_front_inf_Opercular_y4'], 'fr_R_surf_G_front_inf_Orbital': df['fr_R_surf_G_front_inf_Orbital_y4'], 'fr_R_surf_G_front_inf_Triangul': df['fr_R_surf_G_front_inf_Triangul_y4'], 'fr_R_surf_G_front_middle': df['fr_R_surf_G_front_middle_y4'], 'fr_R_surf_G_front_sup': df['fr_R_surf_G_front_sup_y4'], 'fr_R_surf_G_insular_short': df['fr_R_surf_G_insular_short_y4'], 'fr_R_surf_G_oc_temp_lat_fusifor': df['fr_R_surf_G_oc_temp_lat_fusifor_y4'], 'fr_R_surf_G_oc_temp_med_Lingual': df['fr_R_surf_G_oc_temp_med_Lingual_y4'], 'fr_R_surf_G_oc_temp_med_Parahip': df['fr_R_surf_G_oc_temp_med_Parahip_y4'], 'fr_R_surf_G_occipital_middle': df['fr_R_surf_G_occipital_middle_y4'], 'fr_R_surf_G_occipital_sup': df['fr_R_surf_G_occipital_sup_y4'], 'fr_R_surf_G_orbital': df['fr_R_surf_G_orbital_y4'], 'fr_R_surf_G_pariet_inf_Angular': df['fr_R_surf_G_pariet_inf_Angular_y4'], 'fr_R_surf_G_pariet_inf_Supramar': df['fr_R_surf_G_pariet_inf_Supramar_y4'], 'fr_R_surf_G_parietal_sup': df['fr_R_surf_G_parietal_sup_y4'], 'fr_R_surf_G_postcentral': df['fr_R_surf_G_postcentral_y4'], 'fr_R_surf_G_precentral': df['fr_R_surf_G_precentral_y4'], 'fr_R_surf_G_precuneus': df['fr_R_surf_G_precuneus_y4'], 'fr_R_surf_G_rectus': df['fr_R_surf_G_rectus_y4'], 'fr_R_surf_G_temp_sup_G_T_transv': df['fr_R_surf_G_temp_sup_G_T_transv_y4'], 'fr_R_surf_G_temp_sup_Lateral': df['fr_R_surf_G_temp_sup_Lateral_y4'], 'fr_R_surf_G_temp_sup_Plan_polar': df['fr_R_surf_G_temp_sup_Plan_polar_y4'], 'fr_R_surf_G_temp_sup_Plan_tempo': df['fr_R_surf_G_temp_sup_Plan_tempo_y4'], 'fr_R_surf_G_temporal_inf': df['fr_R_surf_G_temporal_inf_y4'], 'fr_R_surf_G_temporal_middle': df['fr_R_surf_G_temporal_middle_y4'], 'fr_R_surf_GandS_cingul_Ant': df['fr_R_surf_GandS_cingul_Ant_y4'], 'fr_R_surf_GandS_cingul_Mid_Post': df['fr_R_surf_GandS_cingul_Mid_Post_y4'], 'fr_R_surf_GandS_frontomargin': df['fr_R_surf_GandS_frontomargin_y4'], 'fr_R_surf_GandS_occipital_inf': df['fr_R_surf_GandS_occipital_inf_y4'], 'fr_R_surf_GandS_paracentral': df['fr_R_surf_GandS_paracentral_y4'], 'fr_R_surf_GandS_subcentral': df['fr_R_surf_GandS_subcentral_y4'], 'fr_R_surf_GandS_transv_frontopol': df['fr_R_surf_GandS_transv_frontopol_y4'], 'fr_R_surf_Lat_Fis_ant_Horizont': df['fr_R_surf_Lat_Fis_ant_Horizont_y4'], 'fr_R_surf_Lat_Fis_post': df['fr_R_surf_Lat_Fis_post_y4'], 'fr_R_surf_Pole_occipital': df['fr_R_surf_Pole_occipital_y4'], 'fr_R_surf_Pole_temporal': df['fr_R_surf_Pole_temporal_y4'], 'fr_R_surf_S_calcarine': df['fr_R_surf_S_calcarine_y4'], 'fr_R_surf_S_central': df['fr_R_surf_S_central_y4'], 'fr_R_surf_S_cingul_Marginalis': df['fr_R_surf_S_cingul_Marginalis_y4'], 'fr_R_surf_S_circular_insula_ant': df['fr_R_surf_S_circular_insula_ant_y4'], 'fr_R_surf_S_circular_insula_inf': df['fr_R_surf_S_circular_insula_inf_y4'], 'fr_R_surf_S_circular_insula_sup': df['fr_R_surf_S_circular_insula_sup_y4'], 'fr_R_surf_S_collat_transv_ant': df['fr_R_surf_S_collat_transv_ant_y4'], 'fr_R_surf_S_collat_transv_post': df['fr_R_surf_S_collat_transv_post_y4'], 'fr_R_surf_S_front_inf': df['fr_R_surf_S_front_inf_y4'], 'fr_R_surf_S_front_middle': df['fr_R_surf_S_front_middle_y4'], 'fr_R_surf_S_front_sup': df['fr_R_surf_S_front_sup_y4'], 'fr_R_surf_S_interm_prim_Jensen': df['fr_R_surf_S_interm_prim_Jensen_y4'], 'fr_R_surf_S_intraparietandP_trans': df['fr_R_surf_S_intraparietandP_trans_y4'], 'fr_R_surf_S_oc_middleandLunatus': df['fr_R_surf_S_oc_middleandLunatus_y4'], 'fr_R_surf_S_oc_temp_lat': df['fr_R_surf_S_oc_temp_lat_y4'], 'fr_R_surf_S_oc_temp_medandLingual': df['fr_R_surf_S_oc_temp_medandLingual_y4'], 'fr_R_surf_S_orbital_H_Shaped': df['fr_R_surf_S_orbital_H_Shaped_y4'], 'fr_R_surf_S_orbital_lateral': df['fr_R_surf_S_orbital_lateral_y4'], 'fr_R_surf_S_orbital_med_olfact': df['fr_R_surf_S_orbital_med_olfact_y4'], 'fr_R_surf_S_parieto_occipital': df['fr_R_surf_S_parieto_occipital_y4'], 'fr_R_surf_S_pericallosal': df['fr_R_surf_S_pericallosal_y4'], 'fr_R_surf_S_postcentral': df['fr_R_surf_S_postcentral_y4'], 'fr_R_surf_S_precentral_inf_part': df['fr_R_surf_S_precentral_inf_part_y4'], 'fr_R_surf_S_precentral_sup_part': df['fr_R_surf_S_precentral_sup_part_y4'], 'fr_R_surf_S_suborbital': df['fr_R_surf_S_suborbital_y4'], 'fr_R_surf_S_subparietal': df['fr_R_surf_S_subparietal_y4'], 'fr_R_surf_S_temporal_inf': df['fr_R_surf_S_temporal_inf_y4'], 'fr_R_surf_S_temporal_sup': df['fr_R_surf_S_temporal_sup_y4'], 'fr_R_surf_S_temporal_transverse': df['fr_R_surf_S_temporal_transverse_y4'], 'fr_R_thick_G_Ins_lgandS_cent_ins': df['fr_R_thick_G_Ins_lgandS_cent_ins_y4'], 'fr_R_thick_G_cingul_Post_ventral': df['fr_R_thick_G_cingul_Post_ventral_y4'], 'fr_R_thick_G_cuneus': df['fr_R_thick_G_cuneus_y4'], 'fr_R_thick_G_front_inf_Opercular': df['fr_R_thick_G_front_inf_Opercular_y4'], 'fr_R_thick_G_front_inf_Orbital': df['fr_R_thick_G_front_inf_Orbital_y4'], 'fr_R_thick_G_front_inf_Triangul': df['fr_R_thick_G_front_inf_Triangul_y4'], 'fr_R_thick_G_front_middle': df['fr_R_thick_G_front_middle_y4'], 'fr_R_thick_G_front_sup': df['fr_R_thick_G_front_sup_y4'], 'fr_R_thick_G_insular_short': df['fr_R_thick_G_insular_short_y4'], 'fr_R_thick_G_oc_temp_lat_fusifor': df['fr_R_thick_G_oc_temp_lat_fusifor_y4'], 'fr_R_thick_G_oc_temp_med_Lingual': df['fr_R_thick_G_oc_temp_med_Lingual_y4'], 'fr_R_thick_G_oc_temp_med_Parahip': df['fr_R_thick_G_oc_temp_med_Parahip_y4'], 'fr_R_thick_G_occipital_middle': df['fr_R_thick_G_occipital_middle_y4'], 'fr_R_thick_G_occipital_sup': df['fr_R_thick_G_occipital_sup_y4'], 'fr_R_thick_G_orbital': df['fr_R_thick_G_orbital_y4'], 'fr_R_thick_G_pariet_inf_Angular': df['fr_R_thick_G_pariet_inf_Angular_y4'], 'fr_R_thick_G_pariet_inf_Supramar': df['fr_R_thick_G_pariet_inf_Supramar_y4'], 'fr_R_thick_G_parietal_sup': df['fr_R_thick_G_parietal_sup_y4'], 'fr_R_thick_G_postcentral': df['fr_R_thick_G_postcentral_y4'], 'fr_R_thick_G_precentral': df['fr_R_thick_G_precentral_y4'], 'fr_R_thick_G_precuneus': df['fr_R_thick_G_precuneus_y4'], 'fr_R_thick_G_rectus': df['fr_R_thick_G_rectus_y4'], 'fr_R_thick_G_temp_sup_G_T_transv': df['fr_R_thick_G_temp_sup_G_T_transv_y4'], 'fr_R_thick_G_temp_sup_Lateral': df['fr_R_thick_G_temp_sup_Lateral_y4'], 'fr_R_thick_G_temp_sup_Plan_polar': df['fr_R_thick_G_temp_sup_Plan_polar_y4'], 'fr_R_thick_G_temp_sup_Plan_tempo': df['fr_R_thick_G_temp_sup_Plan_tempo_y4'], 'fr_R_thick_G_temporal_inf': df['fr_R_thick_G_temporal_inf_y4'], 'fr_R_thick_G_temporal_middle': df['fr_R_thick_G_temporal_middle_y4'], 'fr_R_thick_GandS_cingul_Ant': df['fr_R_thick_GandS_cingul_Ant_y4'], 'fr_R_thick_GandS_cingul_Mid_Post': df['fr_R_thick_GandS_cingul_Mid_Post_y4'], 'fr_R_thick_GandS_frontomargin': df['fr_R_thick_GandS_frontomargin_y4'], 'fr_R_thick_GandS_occipital_inf': df['fr_R_thick_GandS_occipital_inf_y4'], 'fr_R_thick_GandS_paracentral': df['fr_R_thick_GandS_paracentral_y4'], 'fr_R_thick_GandS_subcentral': df['fr_R_thick_GandS_subcentral_y4'], 'fr_R_thick_GandS_transv_frontopol': df['fr_R_thick_GandS_transv_frontopol_y4'], 'fr_R_thick_Lat_Fis_ant_Horizont': df['fr_R_thick_Lat_Fis_ant_Horizont_y4'], 'fr_R_thick_Lat_Fis_post': df['fr_R_thick_Lat_Fis_post_y4'], 'fr_R_thick_Pole_occipital': df['fr_R_thick_Pole_occipital_y4'], 'fr_R_thick_Pole_temporal': df['fr_R_thick_Pole_temporal_y4'], 'fr_R_thick_S_calcarine': df['fr_R_thick_S_calcarine_y4'], 'fr_R_thick_S_central': df['fr_R_thick_S_central_y4'], 'fr_R_thick_S_cingul_Marginalis': df['fr_R_thick_S_cingul_Marginalis_y4'], 'fr_R_thick_S_circular_insula_ant': df['fr_R_thick_S_circular_insula_ant_y4'], 'fr_R_thick_S_circular_insula_inf': df['fr_R_thick_S_circular_insula_inf_y4'], 'fr_R_thick_S_circular_insula_sup': df['fr_R_thick_S_circular_insula_sup_y4'], 'fr_R_thick_S_collat_transv_ant': df['fr_R_thick_S_collat_transv_ant_y4'], 'fr_R_thick_S_collat_transv_post': df['fr_R_thick_S_collat_transv_post_y4'], 'fr_R_thick_S_front_inf': df['fr_R_thick_S_front_inf_y4'], 'fr_R_thick_S_front_middle': df['fr_R_thick_S_front_middle_y4'], 'fr_R_thick_S_front_sup': df['fr_R_thick_S_front_sup_y4'], 'fr_R_thick_S_interm_prim_Jensen': df['fr_R_thick_S_interm_prim_Jensen_y4'], 'fr_R_thick_S_intraparietandP_trans': df['fr_R_thick_S_intraparietandP_trans_y4'], 'fr_R_thick_S_oc_middleandLunatus': df['fr_R_thick_S_oc_middleandLunatus_y4'], 'fr_R_thick_S_oc_temp_lat': df['fr_R_thick_S_oc_temp_lat_y4'], 'fr_R_thick_S_oc_temp_medandLingual': df['fr_R_thick_S_oc_temp_medandLingual_y4'], 'fr_R_thick_S_orbital_H_Shaped': df['fr_R_thick_S_orbital_H_Shaped_y4'], 'fr_R_thick_S_orbital_lateral': df['fr_R_thick_S_orbital_lateral_y4'], 'fr_R_thick_S_orbital_med_olfact': df['fr_R_thick_S_orbital_med_olfact_y4'], 'fr_R_thick_S_parieto_occipital': df['fr_R_thick_S_parieto_occipital_y4'], 'fr_R_thick_S_pericallosal': df['fr_R_thick_S_pericallosal_y4'], 'fr_R_thick_S_postcentral': df['fr_R_thick_S_postcentral_y4'], 'fr_R_thick_S_precentral_inf_part': df['fr_R_thick_S_precentral_inf_part_y4'], 'fr_R_thick_S_precentral_sup_part': df['fr_R_thick_S_precentral_sup_part_y4'], 'fr_R_thick_S_suborbital': df['fr_R_thick_S_suborbital_y4'], 'fr_R_thick_S_subparietal': df['fr_R_thick_S_subparietal_y4'], 'fr_R_thick_S_temporal_inf': df['fr_R_thick_S_temporal_inf_y4'], 'fr_R_thick_S_temporal_sup': df['fr_R_thick_S_temporal_sup_y4'], 'fr_R_thick_S_temporal_transverse': df['fr_R_thick_S_temporal_transverse_y4'], 'fr_L_surf_G_Ins_lgandS_cent_ins': df['fr_L_surf_G_Ins_lgandS_cent_ins_y4'], 'fr_L_surf_G_cingul_Post_ventral': df['fr_L_surf_G_cingul_Post_ventral_y4'], 'fr_L_surf_G_cuneus': df['fr_L_surf_G_cuneus_y4'], 'fr_L_surf_G_front_inf_Opercular': df['fr_L_surf_G_front_inf_Opercular_y4'], 'fr_L_surf_G_front_inf_Orbital': df['fr_L_surf_G_front_inf_Orbital_y4'], 'fr_L_surf_G_front_inf_Triangul': df['fr_L_surf_G_front_inf_Triangul_y4'], 'fr_L_surf_G_front_middle': df['fr_L_surf_G_front_middle_y4'], 'fr_L_surf_G_front_sup': df['fr_L_surf_G_front_sup_y4'], 'fr_L_surf_G_insular_short': df['fr_L_surf_G_insular_short_y4'], 'fr_L_surf_G_oc_temp_lat_fusifor': df['fr_L_surf_G_oc_temp_lat_fusifor_y4'], 'fr_L_surf_G_oc_temp_med_Lingual': df['fr_L_surf_G_oc_temp_med_Lingual_y4'], 'fr_L_surf_G_oc_temp_med_Parahip': df['fr_L_surf_G_oc_temp_med_Parahip_y4'], 'fr_L_surf_G_occipital_middle': df['fr_L_surf_G_occipital_middle_y4'], 'fr_L_surf_G_occipital_sup': df['fr_L_surf_G_occipital_sup_y4'], 'fr_L_surf_G_orbital': df['fr_L_surf_G_orbital_y4'], 'fr_L_surf_G_pariet_inf_Angular': df['fr_L_surf_G_pariet_inf_Angular_y4'], 'fr_L_surf_G_pariet_inf_Supramar': df['fr_L_surf_G_pariet_inf_Supramar_y4'], 'fr_L_surf_G_parietal_sup': df['fr_L_surf_G_parietal_sup_y4'], 'fr_L_surf_G_postcentral': df['fr_L_surf_G_postcentral_y4'], 'fr_L_surf_G_precentral': df['fr_L_surf_G_precentral_y4'], 'fr_L_surf_G_precuneus': df['fr_L_surf_G_precuneus_y4'], 'fr_L_surf_G_rectus': df['fr_L_surf_G_rectus_y4'], 'fr_L_surf_G_temp_sup_G_T_transv': df['fr_L_surf_G_temp_sup_G_T_transv_y4'], 'fr_L_surf_G_temp_sup_Lateral': df['fr_L_surf_G_temp_sup_Lateral_y4'], 'fr_L_surf_G_temp_sup_Plan_polar': df['fr_L_surf_G_temp_sup_Plan_polar_y4'], 'fr_L_surf_G_temp_sup_Plan_tempo': df['fr_L_surf_G_temp_sup_Plan_tempo_y4'], 'fr_L_surf_G_temporal_inf': df['fr_L_surf_G_temporal_inf_y4'], 'fr_L_surf_G_temporal_middle': df['fr_L_surf_G_temporal_middle_y4'], 'fr_L_surf_GandS_cingul_Ant': df['fr_L_surf_GandS_cingul_Ant_y4'], 'fr_L_surf_GandS_cingul_Mid_Post': df['fr_L_surf_GandS_cingul_Mid_Post_y4'], 'fr_L_surf_GandS_frontomargin': df['fr_L_surf_GandS_frontomargin_y4'], 'fr_L_surf_GandS_occipital_inf': df['fr_L_surf_GandS_occipital_inf_y4'], 'fr_L_surf_GandS_paracentral': df['fr_L_surf_GandS_paracentral_y4'], 'fr_L_surf_GandS_subcentral': df['fr_L_surf_GandS_subcentral_y4'], 'fr_L_surf_GandS_transv_frontopol': df['fr_L_surf_GandS_transv_frontopol_y4'], 'fr_L_surf_Lat_Fis_ant_Horizont': df['fr_L_surf_Lat_Fis_ant_Horizont_y4'], 'fr_L_surf_Lat_Fis_post': df['fr_L_surf_Lat_Fis_post_y4'], 'fr_L_surf_Pole_occipital': df['fr_L_surf_Pole_occipital_y4'], 'fr_L_surf_Pole_temporal': df['fr_L_surf_Pole_temporal_y4'], 'fr_L_surf_S_calcarine': df['fr_L_surf_S_calcarine_y4'], 'fr_L_surf_S_central': df['fr_L_surf_S_central_y4'], 'fr_L_surf_S_cingul_Marginalis': df['fr_L_surf_S_cingul_Marginalis_y4'], 'fr_L_surf_S_circular_insula_ant': df['fr_L_surf_S_circular_insula_ant_y4'], 'fr_L_surf_S_circular_insula_inf': df['fr_L_surf_S_circular_insula_inf_y4'], 'fr_L_surf_S_circular_insula_sup': df['fr_L_surf_S_circular_insula_sup_y4'], 'fr_L_surf_S_collat_transv_ant': df['fr_L_surf_S_collat_transv_ant_y4'], 'fr_L_surf_S_collat_transv_post': df['fr_L_surf_S_collat_transv_post_y4'], 'fr_L_surf_S_front_inf': df['fr_L_surf_S_front_inf_y4'], 'fr_L_surf_S_front_middle': df['fr_L_surf_S_front_middle_y4'], 'fr_L_surf_S_front_sup': df['fr_L_surf_S_front_sup_y4'],  'fr_L_surf_S_intraparietandP_trans': df['fr_L_surf_S_intraparietandP_trans_y4'], 'fr_L_surf_S_oc_middleandLunatus': df['fr_L_surf_S_oc_middleandLunatus_y4'], 'fr_L_surf_S_oc_temp_lat': df['fr_L_surf_S_oc_temp_lat_y4'], 'fr_L_surf_S_oc_temp_medandLingual': df['fr_L_surf_S_oc_temp_medandLingual_y4'], 'fr_L_surf_S_orbital_H_Shaped': df['fr_L_surf_S_orbital_H_Shaped_y4'], 'fr_L_surf_S_orbital_lateral': df['fr_L_surf_S_orbital_lateral_y4'], 'fr_L_surf_S_orbital_med_olfact': df['fr_L_surf_S_orbital_med_olfact_y4'], 'fr_L_surf_S_parieto_occipital': df['fr_L_surf_S_parieto_occipital_y4'], 'fr_L_surf_S_postcentral': df['fr_L_surf_S_postcentral_y4'], 'fr_L_surf_S_precentral_inf_part': df['fr_L_surf_S_precentral_inf_part_y4'], 'fr_L_surf_S_precentral_sup_part': df['fr_L_surf_S_precentral_sup_part_y4'], 'fr_L_surf_S_suborbital': df['fr_L_surf_S_suborbital_y4'], 'fr_L_surf_S_subparietal': df['fr_L_surf_S_subparietal_y4'], 'fr_L_surf_S_temporal_inf': df['fr_L_surf_S_temporal_inf_y4'], 'fr_L_surf_S_temporal_sup': df['fr_L_surf_S_temporal_sup_y4']}
		#frame_free5 = {'handlat': df['lat_manual'],'apoe': df['apoe'], 'y_last':df['ultimavisita'], 'dx_last': df['ultimodx'], 'dx_visita': df['dx_corto_visita5'], 'age_lastdx': df['edad_ultimodx'], 'age': df['edad_visita5'], 'sex': df['sexo'], 'free_R_Thal': df['fr_Right_Thalamus_Proper_y5'], 'free_L_Thal': df['fr_Left_Thalamus_Proper_y5'],'free_R_Puta': df['fr_Right_Putamen_y5'], 'free_L_Puta': df['fr_Left_Putamen_y5'],'free_R_Amyg': df['fr_Right_Amygdala_y5'], 'free_L_Amyg': df['fr_Left_Amygdala_y5'],'free_R_Pall': df['fr_Right_Pallidum_y5'], 'free_L_Pall': df['fr_Left_Pallidum_y5'],'free_R_Caud': df['fr_Right_Caudate_y5'], 'free_L_Caud': df['fr_Left_Caudate_y5'],'free_R_Hipp': df['fr_Right_Hippocampus_y5'], 'free_L_Hipp': df['fr_Left_Hippocampus_y5'],'free_R_Accu': df['fr_Right_Accumbens_area_y5'], 'free_L_Accu': df['fr_Left_Accumbens_area_y5'], 'fr_BrainSegVol': df['fr_BrainSegVol_y5'], 'fr_BrainSegVolNotVent':df['fr_BrainSegVolNotVent_y5'], 'fr_BrainSegVol_to_eTIV': df['fr_BrainSegVol_to_eTIV_y5']}
		frame_free5 = {'education': df['nivel_educativo'],'handlat': df['lat_manual'],'apoe': df['apoe'], 'nvisita':df['nvisita5'] ,'y_last':df['ultimavisita'], 'dx_last': df['ultimodx'], 'dx_visita': df['dx_corto_visita5'], 'age_lastdx': df['edad_ultimodx'], 'age': df['edad_visita5'], 'sex': df['sexo'], 'free_R_Thal': df['fr_Right_Thalamus_Proper_y5'], 'free_L_Thal': df['fr_Left_Thalamus_Proper_y5'],'free_R_Puta': df['fr_Right_Putamen_y5'], 'free_L_Puta': df['fr_Left_Putamen_y5'],'free_R_Amyg': df['fr_Right_Amygdala_y5'], 'free_L_Amyg': df['fr_Left_Amygdala_y5'],'free_R_Pall': df['fr_Right_Pallidum_y5'], 'free_L_Pall': df['fr_Left_Pallidum_y5'],'free_R_Caud': df['fr_Right_Caudate_y5'], 'free_L_Caud': df['fr_Left_Caudate_y5'],'free_R_Hipp': df['fr_Right_Hippocampus_y5'], 'free_L_Hipp': df['fr_Left_Hippocampus_y5'],'free_R_Accu': df['fr_Right_Accumbens_area_y5'], 'free_L_Accu': df['fr_Left_Accumbens_area_y5'], 'fr_BrainSegVol': df['fr_BrainSegVol_y5'], 'fr_BrainSegVolNotVent':df['fr_BrainSegVolNotVent_y5'], 'fr_BrainSegVol_to_eTIV': df['fr_BrainSegVol_to_eTIV_y5'],  'fr_R_surf_G_Ins_lgandS_cent_ins': df['fr_R_surf_G_Ins_lgandS_cent_ins_y5'], 'fr_R_surf_G_cingul_Post_ventral': df['fr_R_surf_G_cingul_Post_ventral_y5'], 'fr_R_surf_G_cuneus': df['fr_R_surf_G_cuneus_y5'], 'fr_R_surf_G_front_inf_Opercular': df['fr_R_surf_G_front_inf_Opercular_y5'], 'fr_R_surf_G_front_inf_Orbital': df['fr_R_surf_G_front_inf_Orbital_y5'], 'fr_R_surf_G_front_inf_Triangul': df['fr_R_surf_G_front_inf_Triangul_y5'], 'fr_R_surf_G_front_middle': df['fr_R_surf_G_front_middle_y5'], 'fr_R_surf_G_front_sup': df['fr_R_surf_G_front_sup_y5'], 'fr_R_surf_G_insular_short': df['fr_R_surf_G_insular_short_y5'], 'fr_R_surf_G_oc_temp_lat_fusifor': df['fr_R_surf_G_oc_temp_lat_fusifor_y5'], 'fr_R_surf_G_oc_temp_med_Lingual': df['fr_R_surf_G_oc_temp_med_Lingual_y5'], 'fr_R_surf_G_oc_temp_med_Parahip': df['fr_R_surf_G_oc_temp_med_Parahip_y5'], 'fr_R_surf_G_occipital_middle': df['fr_R_surf_G_occipital_middle_y5'], 'fr_R_surf_G_occipital_sup': df['fr_R_surf_G_occipital_sup_y5'], 'fr_R_surf_G_orbital': df['fr_R_surf_G_orbital_y5'], 'fr_R_surf_G_pariet_inf_Angular': df['fr_R_surf_G_pariet_inf_Angular_y5'], 'fr_R_surf_G_pariet_inf_Supramar': df['fr_R_surf_G_pariet_inf_Supramar_y5'], 'fr_R_surf_G_parietal_sup': df['fr_R_surf_G_parietal_sup_y5'], 'fr_R_surf_G_postcentral': df['fr_R_surf_G_postcentral_y5'], 'fr_R_surf_G_precentral': df['fr_R_surf_G_precentral_y5'], 'fr_R_surf_G_precuneus': df['fr_R_surf_G_precuneus_y5'], 'fr_R_surf_G_rectus': df['fr_R_surf_G_rectus_y5'], 'fr_R_surf_G_temp_sup_G_T_transv': df['fr_R_surf_G_temp_sup_G_T_transv_y5'], 'fr_R_surf_G_temp_sup_Lateral': df['fr_R_surf_G_temp_sup_Lateral_y5'], 'fr_R_surf_G_temp_sup_Plan_polar': df['fr_R_surf_G_temp_sup_Plan_polar_y5'], 'fr_R_surf_G_temp_sup_Plan_tempo': df['fr_R_surf_G_temp_sup_Plan_tempo_y5'], 'fr_R_surf_G_temporal_inf': df['fr_R_surf_G_temporal_inf_y5'], 'fr_R_surf_G_temporal_middle': df['fr_R_surf_G_temporal_middle_y5'], 'fr_R_surf_GandS_cingul_Ant': df['fr_R_surf_GandS_cingul_Ant_y5'], 'fr_R_surf_GandS_cingul_Mid_Post': df['fr_R_surf_GandS_cingul_Mid_Post_y5'], 'fr_R_surf_GandS_frontomargin': df['fr_R_surf_GandS_frontomargin_y5'], 'fr_R_surf_GandS_occipital_inf': df['fr_R_surf_GandS_occipital_inf_y5'], 'fr_R_surf_GandS_paracentral': df['fr_R_surf_GandS_paracentral_y5'], 'fr_R_surf_GandS_subcentral': df['fr_R_surf_GandS_subcentral_y5'], 'fr_R_surf_GandS_transv_frontopol': df['fr_R_surf_GandS_transv_frontopol_y5'], 'fr_R_surf_Lat_Fis_ant_Horizont': df['fr_R_surf_Lat_Fis_ant_Horizont_y5'], 'fr_R_surf_Lat_Fis_post': df['fr_R_surf_Lat_Fis_post_y5'], 'fr_R_surf_Pole_occipital': df['fr_R_surf_Pole_occipital_y5'], 'fr_R_surf_Pole_temporal': df['fr_R_surf_Pole_temporal_y5'], 'fr_R_surf_S_calcarine': df['fr_R_surf_S_calcarine_y5'], 'fr_R_surf_S_central': df['fr_R_surf_S_central_y5'], 'fr_R_surf_S_cingul_Marginalis': df['fr_R_surf_S_cingul_Marginalis_y5'], 'fr_R_surf_S_circular_insula_ant': df['fr_R_surf_S_circular_insula_ant_y5'], 'fr_R_surf_S_circular_insula_inf': df['fr_R_surf_S_circular_insula_inf_y5'], 'fr_R_surf_S_circular_insula_sup': df['fr_R_surf_S_circular_insula_sup_y5'], 'fr_R_surf_S_collat_transv_ant': df['fr_R_surf_S_collat_transv_ant_y5'], 'fr_R_surf_S_collat_transv_post': df['fr_R_surf_S_collat_transv_post_y5'], 'fr_R_surf_S_front_inf': df['fr_R_surf_S_front_inf_y5'], 'fr_R_surf_S_front_middle': df['fr_R_surf_S_front_middle_y5'], 'fr_R_surf_S_front_sup': df['fr_R_surf_S_front_sup_y5'], 'fr_R_surf_S_interm_prim_Jensen': df['fr_R_surf_S_interm_prim_Jensen_y5'], 'fr_R_surf_S_intraparietandP_trans': df['fr_R_surf_S_intraparietandP_trans_y5'], 'fr_R_surf_S_oc_middleandLunatus': df['fr_R_surf_S_oc_middleandLunatus_y5'], 'fr_R_surf_S_oc_temp_lat': df['fr_R_surf_S_oc_temp_lat_y5'], 'fr_R_surf_S_oc_temp_medandLingual': df['fr_R_surf_S_oc_temp_medandLingual_y5'], 'fr_R_surf_S_orbital_H_Shaped': df['fr_R_surf_S_orbital_H_Shaped_y5'], 'fr_R_surf_S_orbital_lateral': df['fr_R_surf_S_orbital_lateral_y5'], 'fr_R_surf_S_orbital_med_olfact': df['fr_R_surf_S_orbital_med_olfact_y5'], 'fr_R_surf_S_parieto_occipital': df['fr_R_surf_S_parieto_occipital_y5'], 'fr_R_surf_S_pericallosal': df['fr_R_surf_S_pericallosal_y5'], 'fr_R_surf_S_postcentral': df['fr_R_surf_S_postcentral_y5'], 'fr_R_surf_S_precentral_inf_part': df['fr_R_surf_S_precentral_inf_part_y5'], 'fr_R_surf_S_precentral_sup_part': df['fr_R_surf_S_precentral_sup_part_y5'], 'fr_R_surf_S_suborbital': df['fr_R_surf_S_suborbital_y5'], 'fr_R_surf_S_subparietal': df['fr_R_surf_S_subparietal_y5'], 'fr_R_surf_S_temporal_inf': df['fr_R_surf_S_temporal_inf_y5'], 'fr_R_surf_S_temporal_sup': df['fr_R_surf_S_temporal_sup_y5'], 'fr_R_surf_S_temporal_transverse': df['fr_R_surf_S_temporal_transverse_y5'], 'fr_R_thick_G_Ins_lgandS_cent_ins': df['fr_R_thick_G_Ins_lgandS_cent_ins_y5'], 'fr_R_thick_G_cingul_Post_ventral': df['fr_R_thick_G_cingul_Post_ventral_y5'], 'fr_R_thick_G_cuneus': df['fr_R_thick_G_cuneus_y5'], 'fr_R_thick_G_front_inf_Opercular': df['fr_R_thick_G_front_inf_Opercular_y5'], 'fr_R_thick_G_front_inf_Orbital': df['fr_R_thick_G_front_inf_Orbital_y5'], 'fr_R_thick_G_front_inf_Triangul': df['fr_R_thick_G_front_inf_Triangul_y5'], 'fr_R_thick_G_front_middle': df['fr_R_thick_G_front_middle_y5'], 'fr_R_thick_G_front_sup': df['fr_R_thick_G_front_sup_y5'], 'fr_R_thick_G_insular_short': df['fr_R_thick_G_insular_short_y5'], 'fr_R_thick_G_oc_temp_lat_fusifor': df['fr_R_thick_G_oc_temp_lat_fusifor_y5'], 'fr_R_thick_G_oc_temp_med_Lingual': df['fr_R_thick_G_oc_temp_med_Lingual_y5'], 'fr_R_thick_G_oc_temp_med_Parahip': df['fr_R_thick_G_oc_temp_med_Parahip_y5'], 'fr_R_thick_G_occipital_middle': df['fr_R_thick_G_occipital_middle_y5'], 'fr_R_thick_G_occipital_sup': df['fr_R_thick_G_occipital_sup_y5'], 'fr_R_thick_G_orbital': df['fr_R_thick_G_orbital_y5'], 'fr_R_thick_G_pariet_inf_Angular': df['fr_R_thick_G_pariet_inf_Angular_y5'], 'fr_R_thick_G_pariet_inf_Supramar': df['fr_R_thick_G_pariet_inf_Supramar_y5'], 'fr_R_thick_G_parietal_sup': df['fr_R_thick_G_parietal_sup_y5'], 'fr_R_thick_G_postcentral': df['fr_R_thick_G_postcentral_y5'], 'fr_R_thick_G_precentral': df['fr_R_thick_G_precentral_y5'], 'fr_R_thick_G_precuneus': df['fr_R_thick_G_precuneus_y5'], 'fr_R_thick_G_rectus': df['fr_R_thick_G_rectus_y5'], 'fr_R_thick_G_temp_sup_G_T_transv': df['fr_R_thick_G_temp_sup_G_T_transv_y5'], 'fr_R_thick_G_temp_sup_Lateral': df['fr_R_thick_G_temp_sup_Lateral_y5'], 'fr_R_thick_G_temp_sup_Plan_polar': df['fr_R_thick_G_temp_sup_Plan_polar_y5'], 'fr_R_thick_G_temp_sup_Plan_tempo': df['fr_R_thick_G_temp_sup_Plan_tempo_y5'], 'fr_R_thick_G_temporal_inf': df['fr_R_thick_G_temporal_inf_y5'], 'fr_R_thick_G_temporal_middle': df['fr_R_thick_G_temporal_middle_y5'], 'fr_R_thick_GandS_cingul_Ant': df['fr_R_thick_GandS_cingul_Ant_y5'], 'fr_R_thick_GandS_cingul_Mid_Post': df['fr_R_thick_GandS_cingul_Mid_Post_y5'], 'fr_R_thick_GandS_frontomargin': df['fr_R_thick_GandS_frontomargin_y5'], 'fr_R_thick_GandS_occipital_inf': df['fr_R_thick_GandS_occipital_inf_y5'], 'fr_R_thick_GandS_paracentral': df['fr_R_thick_GandS_paracentral_y5'], 'fr_R_thick_GandS_subcentral': df['fr_R_thick_GandS_subcentral_y5'], 'fr_R_thick_GandS_transv_frontopol': df['fr_R_thick_GandS_transv_frontopol_y5'], 'fr_R_thick_Lat_Fis_ant_Horizont': df['fr_R_thick_Lat_Fis_ant_Horizont_y5'], 'fr_R_thick_Lat_Fis_post': df['fr_R_thick_Lat_Fis_post_y5'], 'fr_R_thick_Pole_occipital': df['fr_R_thick_Pole_occipital_y5'], 'fr_R_thick_Pole_temporal': df['fr_R_thick_Pole_temporal_y5'], 'fr_R_thick_S_calcarine': df['fr_R_thick_S_calcarine_y5'], 'fr_R_thick_S_central': df['fr_R_thick_S_central_y5'], 'fr_R_thick_S_cingul_Marginalis': df['fr_R_thick_S_cingul_Marginalis_y5'], 'fr_R_thick_S_circular_insula_ant': df['fr_R_thick_S_circular_insula_ant_y5'], 'fr_R_thick_S_circular_insula_inf': df['fr_R_thick_S_circular_insula_inf_y5'], 'fr_R_thick_S_circular_insula_sup': df['fr_R_thick_S_circular_insula_sup_y5'], 'fr_R_thick_S_collat_transv_ant': df['fr_R_thick_S_collat_transv_ant_y5'], 'fr_R_thick_S_collat_transv_post': df['fr_R_thick_S_collat_transv_post_y5'], 'fr_R_thick_S_front_inf': df['fr_R_thick_S_front_inf_y5'], 'fr_R_thick_S_front_middle': df['fr_R_thick_S_front_middle_y5'], 'fr_R_thick_S_front_sup': df['fr_R_thick_S_front_sup_y5'], 'fr_R_thick_S_interm_prim_Jensen': df['fr_R_thick_S_interm_prim_Jensen_y5'], 'fr_R_thick_S_intraparietandP_trans': df['fr_R_thick_S_intraparietandP_trans_y5'], 'fr_R_thick_S_oc_middleandLunatus': df['fr_R_thick_S_oc_middleandLunatus_y5'], 'fr_R_thick_S_oc_temp_lat': df['fr_R_thick_S_oc_temp_lat_y5'], 'fr_R_thick_S_oc_temp_medandLingual': df['fr_R_thick_S_oc_temp_medandLingual_y5'], 'fr_R_thick_S_orbital_H_Shaped': df['fr_R_thick_S_orbital_H_Shaped_y5'], 'fr_R_thick_S_orbital_lateral': df['fr_R_thick_S_orbital_lateral_y5'], 'fr_R_thick_S_orbital_med_olfact': df['fr_R_thick_S_orbital_med_olfact_y5'], 'fr_R_thick_S_parieto_occipital': df['fr_R_thick_S_parieto_occipital_y5'], 'fr_R_thick_S_pericallosal': df['fr_R_thick_S_pericallosal_y5'], 'fr_R_thick_S_postcentral': df['fr_R_thick_S_postcentral_y5'], 'fr_R_thick_S_precentral_inf_part': df['fr_R_thick_S_precentral_inf_part_y5'], 'fr_R_thick_S_precentral_sup_part': df['fr_R_thick_S_precentral_sup_part_y5'], 'fr_R_thick_S_suborbital': df['fr_R_thick_S_suborbital_y5'], 'fr_R_thick_S_subparietal': df['fr_R_thick_S_subparietal_y5'], 'fr_R_thick_S_temporal_inf': df['fr_R_thick_S_temporal_inf_y5'], 'fr_R_thick_S_temporal_sup': df['fr_R_thick_S_temporal_sup_y5'], 'fr_R_thick_S_temporal_transverse': df['fr_R_thick_S_temporal_transverse_y5'], 'fr_L_surf_G_Ins_lgandS_cent_ins': df['fr_L_surf_G_Ins_lgandS_cent_ins_y5'], 'fr_L_surf_G_cingul_Post_ventral': df['fr_L_surf_G_cingul_Post_ventral_y5'], 'fr_L_surf_G_cuneus': df['fr_L_surf_G_cuneus_y5'], 'fr_L_surf_G_front_inf_Opercular': df['fr_L_surf_G_front_inf_Opercular_y5'], 'fr_L_surf_G_front_inf_Orbital': df['fr_L_surf_G_front_inf_Orbital_y5'], 'fr_L_surf_G_front_inf_Triangul': df['fr_L_surf_G_front_inf_Triangul_y5'], 'fr_L_surf_G_front_middle': df['fr_L_surf_G_front_middle_y5'], 'fr_L_surf_G_front_sup': df['fr_L_surf_G_front_sup_y5'], 'fr_L_surf_G_insular_short': df['fr_L_surf_G_insular_short_y5'], 'fr_L_surf_G_oc_temp_lat_fusifor': df['fr_L_surf_G_oc_temp_lat_fusifor_y5'], 'fr_L_surf_G_oc_temp_med_Lingual': df['fr_L_surf_G_oc_temp_med_Lingual_y5'], 'fr_L_surf_G_oc_temp_med_Parahip': df['fr_L_surf_G_oc_temp_med_Parahip_y5'], 'fr_L_surf_G_occipital_middle': df['fr_L_surf_G_occipital_middle_y5'], 'fr_L_surf_G_occipital_sup': df['fr_L_surf_G_occipital_sup_y5'], 'fr_L_surf_G_orbital': df['fr_L_surf_G_orbital_y5'], 'fr_L_surf_G_pariet_inf_Angular': df['fr_L_surf_G_pariet_inf_Angular_y5'], 'fr_L_surf_G_pariet_inf_Supramar': df['fr_L_surf_G_pariet_inf_Supramar_y5'], 'fr_L_surf_G_parietal_sup': df['fr_L_surf_G_parietal_sup_y5'], 'fr_L_surf_G_postcentral': df['fr_L_surf_G_postcentral_y5'], 'fr_L_surf_G_precentral': df['fr_L_surf_G_precentral_y5'], 'fr_L_surf_G_precuneus': df['fr_L_surf_G_precuneus_y5'], 'fr_L_surf_G_rectus': df['fr_L_surf_G_rectus_y5'], 'fr_L_surf_G_temp_sup_G_T_transv': df['fr_L_surf_G_temp_sup_G_T_transv_y5'], 'fr_L_surf_G_temp_sup_Lateral': df['fr_L_surf_G_temp_sup_Lateral_y5'], 'fr_L_surf_G_temp_sup_Plan_polar': df['fr_L_surf_G_temp_sup_Plan_polar_y5'], 'fr_L_surf_G_temp_sup_Plan_tempo': df['fr_L_surf_G_temp_sup_Plan_tempo_y5'], 'fr_L_surf_G_temporal_inf': df['fr_L_surf_G_temporal_inf_y5'], 'fr_L_surf_G_temporal_middle': df['fr_L_surf_G_temporal_middle_y5'], 'fr_L_surf_GandS_cingul_Ant': df['fr_L_surf_GandS_cingul_Ant_y5'], 'fr_L_surf_GandS_cingul_Mid_Post': df['fr_L_surf_GandS_cingul_Mid_Post_y5'], 'fr_L_surf_GandS_frontomargin': df['fr_L_surf_GandS_frontomargin_y5'], 'fr_L_surf_GandS_occipital_inf': df['fr_L_surf_GandS_occipital_inf_y5'], 'fr_L_surf_GandS_paracentral': df['fr_L_surf_GandS_paracentral_y5'], 'fr_L_surf_GandS_subcentral': df['fr_L_surf_GandS_subcentral_y5'], 'fr_L_surf_GandS_transv_frontopol': df['fr_L_surf_GandS_transv_frontopol_y5'], 'fr_L_surf_Lat_Fis_ant_Horizont': df['fr_L_surf_Lat_Fis_ant_Horizont_y5'], 'fr_L_surf_Lat_Fis_post': df['fr_L_surf_Lat_Fis_post_y5'], 'fr_L_surf_Pole_occipital': df['fr_L_surf_Pole_occipital_y5'], 'fr_L_surf_Pole_temporal': df['fr_L_surf_Pole_temporal_y5'], 'fr_L_surf_S_calcarine': df['fr_L_surf_S_calcarine_y5'], 'fr_L_surf_S_central': df['fr_L_surf_S_central_y5'], 'fr_L_surf_S_cingul_Marginalis': df['fr_L_surf_S_cingul_Marginalis_y5'], 'fr_L_surf_S_circular_insula_ant': df['fr_L_surf_S_circular_insula_ant_y5'], 'fr_L_surf_S_circular_insula_inf': df['fr_L_surf_S_circular_insula_inf_y5'], 'fr_L_surf_S_circular_insula_sup': df['fr_L_surf_S_circular_insula_sup_y5'], 'fr_L_surf_S_collat_transv_ant': df['fr_L_surf_S_collat_transv_ant_y5'], 'fr_L_surf_S_collat_transv_post': df['fr_L_surf_S_collat_transv_post_y5'], 'fr_L_surf_S_front_inf': df['fr_L_surf_S_front_inf_y5'], 'fr_L_surf_S_front_middle': df['fr_L_surf_S_front_middle_y5'], 'fr_L_surf_S_front_sup': df['fr_L_surf_S_front_sup_y5'],  'fr_L_surf_S_intraparietandP_trans': df['fr_L_surf_S_intraparietandP_trans_y5'], 'fr_L_surf_S_oc_middleandLunatus': df['fr_L_surf_S_oc_middleandLunatus_y5'], 'fr_L_surf_S_oc_temp_lat': df['fr_L_surf_S_oc_temp_lat_y5'], 'fr_L_surf_S_oc_temp_medandLingual': df['fr_L_surf_S_oc_temp_medandLingual_y5'], 'fr_L_surf_S_orbital_H_Shaped': df['fr_L_surf_S_orbital_H_Shaped_y5'], 'fr_L_surf_S_orbital_lateral': df['fr_L_surf_S_orbital_lateral_y5'], 'fr_L_surf_S_orbital_med_olfact': df['fr_L_surf_S_orbital_med_olfact_y5'], 'fr_L_surf_S_parieto_occipital': df['fr_L_surf_S_parieto_occipital_y5'], 'fr_L_surf_S_postcentral': df['fr_L_surf_S_postcentral_y5'], 'fr_L_surf_S_precentral_inf_part': df['fr_L_surf_S_precentral_inf_part_y5'], 'fr_L_surf_S_precentral_sup_part': df['fr_L_surf_S_precentral_sup_part_y5'], 'fr_L_surf_S_suborbital': df['fr_L_surf_S_suborbital_y5'], 'fr_L_surf_S_subparietal': df['fr_L_surf_S_subparietal_y5'], 'fr_L_surf_S_temporal_inf': df['fr_L_surf_S_temporal_inf_y5'], 'fr_L_surf_S_temporal_sup': df['fr_L_surf_S_temporal_sup_y5']}		
		#frame_free6 = {'handlat': df['lat_manual'],'apoe': df['apoe'], 'y_last':df['ultimavisita'], 'dx_last': df['ultimodx'], 'dx_visita': df['dx_corto_visita6'], 'age_lastdx': df['edad_ultimodx'], 'age': df['edad_visita6'], 'sex': df['sexo'], 'free_R_Thal': df['fr_Right_Thalamus_Proper_y6'], 'free_L_Thal': df['fr_Left_Thalamus_Proper_y6'],'free_R_Puta': df['fr_Right_Putamen_y6'], 'free_L_Puta': df['fr_Left_Putamen_y6'],'free_R_Amyg': df['fr_Right_Amygdala_y6'], 'free_L_Amyg': df['fr_Left_Amygdala_y6'],'free_R_Pall': df['fr_Right_Pallidum_y6'], 'free_L_Pall': df['fr_Left_Pallidum_y6'],'free_R_Caud': df['fr_Right_Caudate_y6'], 'free_L_Caud': df['fr_Left_Caudate_y6'],'free_R_Hipp': df['fr_Right_Hippocampus_y6'], 'free_L_Hipp': df['fr_Left_Hippocampus_y6'],'free_R_Accu': df['fr_Right_Accumbens_area_y6'], 'free_L_Accu': df['fr_Left_Accumbens_area_y6'], 'fr_BrainSegVol': df['fr_BrainSegVol_y6'], 'fr_BrainSegVolNotVent':df['fr_BrainSegVolNotVent_y6'], 'fr_BrainSegVol_to_eTIV': df['fr_BrainSegVol_to_eTIV_y6']}
		frame_free6 = {'education': df['nivel_educativo'],'handlat': df['lat_manual'],'apoe': df['apoe'], 'nvisita':df['nvisita6'] , 'y_last':df['ultimavisita'], 'dx_last': df['ultimodx'], 'dx_visita': df['dx_corto_visita6'], 'age_lastdx': df['edad_ultimodx'], 'age': df['edad_visita6'], 'sex': df['sexo'], 'free_R_Thal': df['fr_Right_Thalamus_Proper_y6'], 'free_L_Thal': df['fr_Left_Thalamus_Proper_y6'],'free_R_Puta': df['fr_Right_Putamen_y6'], 'free_L_Puta': df['fr_Left_Putamen_y6'],'free_R_Amyg': df['fr_Right_Amygdala_y6'], 'free_L_Amyg': df['fr_Left_Amygdala_y6'],'free_R_Pall': df['fr_Right_Pallidum_y6'], 'free_L_Pall': df['fr_Left_Pallidum_y6'],'free_R_Caud': df['fr_Right_Caudate_y6'], 'free_L_Caud': df['fr_Left_Caudate_y6'],'free_R_Hipp': df['fr_Right_Hippocampus_y6'], 'free_L_Hipp': df['fr_Left_Hippocampus_y6'],'free_R_Accu': df['fr_Right_Accumbens_area_y6'], 'free_L_Accu': df['fr_Left_Accumbens_area_y6'], 'fr_BrainSegVol': df['fr_BrainSegVol_y6'], 'fr_BrainSegVolNotVent':df['fr_BrainSegVolNotVent_y6'], 'fr_BrainSegVol_to_eTIV': df['fr_BrainSegVol_to_eTIV_y6'], 'fr_R_surf_G_Ins_lgandS_cent_ins': df['fr_R_surf_G_Ins_lgandS_cent_ins_y6'], 'fr_R_surf_G_cingul_Post_ventral': df['fr_R_surf_G_cingul_Post_ventral_y6'], 'fr_R_surf_G_cuneus': df['fr_R_surf_G_cuneus_y6'], 'fr_R_surf_G_front_inf_Opercular': df['fr_R_surf_G_front_inf_Opercular_y6'], 'fr_R_surf_G_front_inf_Orbital': df['fr_R_surf_G_front_inf_Orbital_y6'], 'fr_R_surf_G_front_inf_Triangul': df['fr_R_surf_G_front_inf_Triangul_y6'], 'fr_R_surf_G_front_middle': df['fr_R_surf_G_front_middle_y6'], 'fr_R_surf_G_front_sup': df['fr_R_surf_G_front_sup_y6'], 'fr_R_surf_G_insular_short': df['fr_R_surf_G_insular_short_y6'], 'fr_R_surf_G_oc_temp_lat_fusifor': df['fr_R_surf_G_oc_temp_lat_fusifor_y6'], 'fr_R_surf_G_oc_temp_med_Lingual': df['fr_R_surf_G_oc_temp_med_Lingual_y6'], 'fr_R_surf_G_oc_temp_med_Parahip': df['fr_R_surf_G_oc_temp_med_Parahip_y6'], 'fr_R_surf_G_occipital_middle': df['fr_R_surf_G_occipital_middle_y6'], 'fr_R_surf_G_occipital_sup': df['fr_R_surf_G_occipital_sup_y6'], 'fr_R_surf_G_orbital': df['fr_R_surf_G_orbital_y6'], 'fr_R_surf_G_pariet_inf_Angular': df['fr_R_surf_G_pariet_inf_Angular_y6'], 'fr_R_surf_G_pariet_inf_Supramar': df['fr_R_surf_G_pariet_inf_Supramar_y6'], 'fr_R_surf_G_parietal_sup': df['fr_R_surf_G_parietal_sup_y6'], 'fr_R_surf_G_postcentral': df['fr_R_surf_G_postcentral_y6'], 'fr_R_surf_G_precentral': df['fr_R_surf_G_precentral_y6'], 'fr_R_surf_G_precuneus': df['fr_R_surf_G_precuneus_y6'], 'fr_R_surf_G_rectus': df['fr_R_surf_G_rectus_y6'], 'fr_R_surf_G_temp_sup_G_T_transv': df['fr_R_surf_G_temp_sup_G_T_transv_y6'], 'fr_R_surf_G_temp_sup_Lateral': df['fr_R_surf_G_temp_sup_Lateral_y6'], 'fr_R_surf_G_temp_sup_Plan_polar': df['fr_R_surf_G_temp_sup_Plan_polar_y6'], 'fr_R_surf_G_temp_sup_Plan_tempo': df['fr_R_surf_G_temp_sup_Plan_tempo_y6'], 'fr_R_surf_G_temporal_inf': df['fr_R_surf_G_temporal_inf_y6'], 'fr_R_surf_G_temporal_middle': df['fr_R_surf_G_temporal_middle_y6'], 'fr_R_surf_GandS_cingul_Ant': df['fr_R_surf_GandS_cingul_Ant_y6'], 'fr_R_surf_GandS_cingul_Mid_Post': df['fr_R_surf_GandS_cingul_Mid_Post_y6'], 'fr_R_surf_GandS_frontomargin': df['fr_R_surf_GandS_frontomargin_y6'], 'fr_R_surf_GandS_occipital_inf': df['fr_R_surf_GandS_occipital_inf_y6'], 'fr_R_surf_GandS_paracentral': df['fr_R_surf_GandS_paracentral_y6'], 'fr_R_surf_GandS_subcentral': df['fr_R_surf_GandS_subcentral_y6'], 'fr_R_surf_GandS_transv_frontopol': df['fr_R_surf_GandS_transv_frontopol_y6'], 'fr_R_surf_Lat_Fis_ant_Horizont': df['fr_R_surf_Lat_Fis_ant_Horizont_y6'], 'fr_R_surf_Lat_Fis_post': df['fr_R_surf_Lat_Fis_post_y6'], 'fr_R_surf_Pole_occipital': df['fr_R_surf_Pole_occipital_y6'], 'fr_R_surf_Pole_temporal': df['fr_R_surf_Pole_temporal_y6'], 'fr_R_surf_S_calcarine': df['fr_R_surf_S_calcarine_y6'], 'fr_R_surf_S_central': df['fr_R_surf_S_central_y6'], 'fr_R_surf_S_cingul_Marginalis': df['fr_R_surf_S_cingul_Marginalis_y6'], 'fr_R_surf_S_circular_insula_ant': df['fr_R_surf_S_circular_insula_ant_y6'], 'fr_R_surf_S_circular_insula_inf': df['fr_R_surf_S_circular_insula_inf_y6'], 'fr_R_surf_S_circular_insula_sup': df['fr_R_surf_S_circular_insula_sup_y6'], 'fr_R_surf_S_collat_transv_ant': df['fr_R_surf_S_collat_transv_ant_y6'], 'fr_R_surf_S_collat_transv_post': df['fr_R_surf_S_collat_transv_post_y6'], 'fr_R_surf_S_front_inf': df['fr_R_surf_S_front_inf_y6'], 'fr_R_surf_S_front_middle': df['fr_R_surf_S_front_middle_y6'], 'fr_R_surf_S_front_sup': df['fr_R_surf_S_front_sup_y6'], 'fr_R_surf_S_interm_prim_Jensen': df['fr_R_surf_S_interm_prim_Jensen_y6'], 'fr_R_surf_S_intraparietandP_trans': df['fr_R_surf_S_intraparietandP_trans_y6'], 'fr_R_surf_S_oc_middleandLunatus': df['fr_R_surf_S_oc_middleandLunatus_y6'], 'fr_R_surf_S_oc_temp_lat': df['fr_R_surf_S_oc_temp_lat_y6'], 'fr_R_surf_S_oc_temp_medandLingual': df['fr_R_surf_S_oc_temp_medandLingual_y6'], 'fr_R_surf_S_orbital_H_Shaped': df['fr_R_surf_S_orbital_H_Shaped_y6'], 'fr_R_surf_S_orbital_lateral': df['fr_R_surf_S_orbital_lateral_y6'], 'fr_R_surf_S_orbital_med_olfact': df['fr_R_surf_S_orbital_med_olfact_y6'], 'fr_R_surf_S_parieto_occipital': df['fr_R_surf_S_parieto_occipital_y6'], 'fr_R_surf_S_pericallosal': df['fr_R_surf_S_pericallosal_y6'], 'fr_R_surf_S_postcentral': df['fr_R_surf_S_postcentral_y6'], 'fr_R_surf_S_precentral_inf_part': df['fr_R_surf_S_precentral_inf_part_y6'], 'fr_R_surf_S_precentral_sup_part': df['fr_R_surf_S_precentral_sup_part_y6'], 'fr_R_surf_S_suborbital': df['fr_R_surf_S_suborbital_y6'], 'fr_R_surf_S_subparietal': df['fr_R_surf_S_subparietal_y6'], 'fr_R_surf_S_temporal_inf': df['fr_R_surf_S_temporal_inf_y6'], 'fr_R_surf_S_temporal_sup': df['fr_R_surf_S_temporal_sup_y6'], 'fr_R_surf_S_temporal_transverse': df['fr_R_surf_S_temporal_transverse_y6'], 'fr_R_thick_G_Ins_lgandS_cent_ins': df['fr_R_thick_G_Ins_lgandS_cent_ins_y6'], 'fr_R_thick_G_cingul_Post_ventral': df['fr_R_thick_G_cingul_Post_ventral_y6'], 'fr_R_thick_G_cuneus': df['fr_R_thick_G_cuneus_y6'], 'fr_R_thick_G_front_inf_Opercular': df['fr_R_thick_G_front_inf_Opercular_y6'], 'fr_R_thick_G_front_inf_Orbital': df['fr_R_thick_G_front_inf_Orbital_y6'], 'fr_R_thick_G_front_inf_Triangul': df['fr_R_thick_G_front_inf_Triangul_y6'], 'fr_R_thick_G_front_middle': df['fr_R_thick_G_front_middle_y6'], 'fr_R_thick_G_front_sup': df['fr_R_thick_G_front_sup_y6'], 'fr_R_thick_G_insular_short': df['fr_R_thick_G_insular_short_y6'], 'fr_R_thick_G_oc_temp_lat_fusifor': df['fr_R_thick_G_oc_temp_lat_fusifor_y6'], 'fr_R_thick_G_oc_temp_med_Lingual': df['fr_R_thick_G_oc_temp_med_Lingual_y6'], 'fr_R_thick_G_oc_temp_med_Parahip': df['fr_R_thick_G_oc_temp_med_Parahip_y6'], 'fr_R_thick_G_occipital_middle': df['fr_R_thick_G_occipital_middle_y6'], 'fr_R_thick_G_occipital_sup': df['fr_R_thick_G_occipital_sup_y6'], 'fr_R_thick_G_orbital': df['fr_R_thick_G_orbital_y6'], 'fr_R_thick_G_pariet_inf_Angular': df['fr_R_thick_G_pariet_inf_Angular_y6'], 'fr_R_thick_G_pariet_inf_Supramar': df['fr_R_thick_G_pariet_inf_Supramar_y6'], 'fr_R_thick_G_parietal_sup': df['fr_R_thick_G_parietal_sup_y6'], 'fr_R_thick_G_postcentral': df['fr_R_thick_G_postcentral_y6'], 'fr_R_thick_G_precentral': df['fr_R_thick_G_precentral_y6'], 'fr_R_thick_G_precuneus': df['fr_R_thick_G_precuneus_y6'], 'fr_R_thick_G_rectus': df['fr_R_thick_G_rectus_y6'], 'fr_R_thick_G_temp_sup_G_T_transv': df['fr_R_thick_G_temp_sup_G_T_transv_y6'], 'fr_R_thick_G_temp_sup_Lateral': df['fr_R_thick_G_temp_sup_Lateral_y6'], 'fr_R_thick_G_temp_sup_Plan_polar': df['fr_R_thick_G_temp_sup_Plan_polar_y6'], 'fr_R_thick_G_temp_sup_Plan_tempo': df['fr_R_thick_G_temp_sup_Plan_tempo_y6'], 'fr_R_thick_G_temporal_inf': df['fr_R_thick_G_temporal_inf_y6'], 'fr_R_thick_G_temporal_middle': df['fr_R_thick_G_temporal_middle_y6'], 'fr_R_thick_GandS_cingul_Ant': df['fr_R_thick_GandS_cingul_Ant_y6'], 'fr_R_thick_GandS_cingul_Mid_Post': df['fr_R_thick_GandS_cingul_Mid_Post_y6'], 'fr_R_thick_GandS_frontomargin': df['fr_R_thick_GandS_frontomargin_y6'], 'fr_R_thick_GandS_occipital_inf': df['fr_R_thick_GandS_occipital_inf_y6'], 'fr_R_thick_GandS_paracentral': df['fr_R_thick_GandS_paracentral_y6'], 'fr_R_thick_GandS_subcentral': df['fr_R_thick_GandS_subcentral_y6'], 'fr_R_thick_GandS_transv_frontopol': df['fr_R_thick_GandS_transv_frontopol_y6'], 'fr_R_thick_Lat_Fis_ant_Horizont': df['fr_R_thick_Lat_Fis_ant_Horizont_y6'], 'fr_R_thick_Lat_Fis_post': df['fr_R_thick_Lat_Fis_post_y6'], 'fr_R_thick_Pole_occipital': df['fr_R_thick_Pole_occipital_y6'], 'fr_R_thick_Pole_temporal': df['fr_R_thick_Pole_temporal_y6'], 'fr_R_thick_S_calcarine': df['fr_R_thick_S_calcarine_y6'], 'fr_R_thick_S_central': df['fr_R_thick_S_central_y6'], 'fr_R_thick_S_cingul_Marginalis': df['fr_R_thick_S_cingul_Marginalis_y6'], 'fr_R_thick_S_circular_insula_ant': df['fr_R_thick_S_circular_insula_ant_y6'], 'fr_R_thick_S_circular_insula_inf': df['fr_R_thick_S_circular_insula_inf_y6'], 'fr_R_thick_S_circular_insula_sup': df['fr_R_thick_S_circular_insula_sup_y6'], 'fr_R_thick_S_collat_transv_ant': df['fr_R_thick_S_collat_transv_ant_y6'], 'fr_R_thick_S_collat_transv_post': df['fr_R_thick_S_collat_transv_post_y6'], 'fr_R_thick_S_front_inf': df['fr_R_thick_S_front_inf_y6'], 'fr_R_thick_S_front_middle': df['fr_R_thick_S_front_middle_y6'], 'fr_R_thick_S_front_sup': df['fr_R_thick_S_front_sup_y6'], 'fr_R_thick_S_interm_prim_Jensen': df['fr_R_thick_S_interm_prim_Jensen_y6'], 'fr_R_thick_S_intraparietandP_trans': df['fr_R_thick_S_intraparietandP_trans_y6'], 'fr_R_thick_S_oc_middleandLunatus': df['fr_R_thick_S_oc_middleandLunatus_y6'], 'fr_R_thick_S_oc_temp_lat': df['fr_R_thick_S_oc_temp_lat_y6'], 'fr_R_thick_S_oc_temp_medandLingual': df['fr_R_thick_S_oc_temp_medandLingual_y6'], 'fr_R_thick_S_orbital_H_Shaped': df['fr_R_thick_S_orbital_H_Shaped_y6'], 'fr_R_thick_S_orbital_lateral': df['fr_R_thick_S_orbital_lateral_y6'], 'fr_R_thick_S_orbital_med_olfact': df['fr_R_thick_S_orbital_med_olfact_y6'], 'fr_R_thick_S_parieto_occipital': df['fr_R_thick_S_parieto_occipital_y6'], 'fr_R_thick_S_pericallosal': df['fr_R_thick_S_pericallosal_y6'], 'fr_R_thick_S_postcentral': df['fr_R_thick_S_postcentral_y6'], 'fr_R_thick_S_precentral_inf_part': df['fr_R_thick_S_precentral_inf_part_y6'], 'fr_R_thick_S_precentral_sup_part': df['fr_R_thick_S_precentral_sup_part_y6'], 'fr_R_thick_S_suborbital': df['fr_R_thick_S_suborbital_y6'], 'fr_R_thick_S_subparietal': df['fr_R_thick_S_subparietal_y6'], 'fr_R_thick_S_temporal_inf': df['fr_R_thick_S_temporal_inf_y6'], 'fr_R_thick_S_temporal_sup': df['fr_R_thick_S_temporal_sup_y6'], 'fr_R_thick_S_temporal_transverse': df['fr_R_thick_S_temporal_transverse_y6'], 'fr_L_surf_G_Ins_lgandS_cent_ins': df['fr_L_surf_G_Ins_lgandS_cent_ins_y6'], 'fr_L_surf_G_cingul_Post_ventral': df['fr_L_surf_G_cingul_Post_ventral_y6'], 'fr_L_surf_G_cuneus': df['fr_L_surf_G_cuneus_y6'], 'fr_L_surf_G_front_inf_Opercular': df['fr_L_surf_G_front_inf_Opercular_y6'], 'fr_L_surf_G_front_inf_Orbital': df['fr_L_surf_G_front_inf_Orbital_y6'], 'fr_L_surf_G_front_inf_Triangul': df['fr_L_surf_G_front_inf_Triangul_y6'], 'fr_L_surf_G_front_middle': df['fr_L_surf_G_front_middle_y6'], 'fr_L_surf_G_front_sup': df['fr_L_surf_G_front_sup_y6'], 'fr_L_surf_G_insular_short': df['fr_L_surf_G_insular_short_y6'], 'fr_L_surf_G_oc_temp_lat_fusifor': df['fr_L_surf_G_oc_temp_lat_fusifor_y6'], 'fr_L_surf_G_oc_temp_med_Lingual': df['fr_L_surf_G_oc_temp_med_Lingual_y6'], 'fr_L_surf_G_oc_temp_med_Parahip': df['fr_L_surf_G_oc_temp_med_Parahip_y6'], 'fr_L_surf_G_occipital_middle': df['fr_L_surf_G_occipital_middle_y6'], 'fr_L_surf_G_occipital_sup': df['fr_L_surf_G_occipital_sup_y6'], 'fr_L_surf_G_orbital': df['fr_L_surf_G_orbital_y6'], 'fr_L_surf_G_pariet_inf_Angular': df['fr_L_surf_G_pariet_inf_Angular_y6'], 'fr_L_surf_G_pariet_inf_Supramar': df['fr_L_surf_G_pariet_inf_Supramar_y6'], 'fr_L_surf_G_parietal_sup': df['fr_L_surf_G_parietal_sup_y6'], 'fr_L_surf_G_postcentral': df['fr_L_surf_G_postcentral_y6'], 'fr_L_surf_G_precentral': df['fr_L_surf_G_precentral_y6'], 'fr_L_surf_G_precuneus': df['fr_L_surf_G_precuneus_y6'], 'fr_L_surf_G_rectus': df['fr_L_surf_G_rectus_y6'], 'fr_L_surf_G_temp_sup_G_T_transv': df['fr_L_surf_G_temp_sup_G_T_transv_y6'], 'fr_L_surf_G_temp_sup_Lateral': df['fr_L_surf_G_temp_sup_Lateral_y6'], 'fr_L_surf_G_temp_sup_Plan_polar': df['fr_L_surf_G_temp_sup_Plan_polar_y6'], 'fr_L_surf_G_temp_sup_Plan_tempo': df['fr_L_surf_G_temp_sup_Plan_tempo_y6'], 'fr_L_surf_G_temporal_inf': df['fr_L_surf_G_temporal_inf_y6'], 'fr_L_surf_G_temporal_middle': df['fr_L_surf_G_temporal_middle_y6'], 'fr_L_surf_GandS_cingul_Ant': df['fr_L_surf_GandS_cingul_Ant_y6'], 'fr_L_surf_GandS_cingul_Mid_Post': df['fr_L_surf_GandS_cingul_Mid_Post_y6'], 'fr_L_surf_GandS_frontomargin': df['fr_L_surf_GandS_frontomargin_y6'], 'fr_L_surf_GandS_occipital_inf': df['fr_L_surf_GandS_occipital_inf_y6'], 'fr_L_surf_GandS_paracentral': df['fr_L_surf_GandS_paracentral_y6'], 'fr_L_surf_GandS_subcentral': df['fr_L_surf_GandS_subcentral_y6'], 'fr_L_surf_GandS_transv_frontopol': df['fr_L_surf_GandS_transv_frontopol_y6'], 'fr_L_surf_Lat_Fis_ant_Horizont': df['fr_L_surf_Lat_Fis_ant_Horizont_y6'], 'fr_L_surf_Lat_Fis_post': df['fr_L_surf_Lat_Fis_post_y6'], 'fr_L_surf_Pole_occipital': df['fr_L_surf_Pole_occipital_y6'], 'fr_L_surf_Pole_temporal': df['fr_L_surf_Pole_temporal_y6'], 'fr_L_surf_S_calcarine': df['fr_L_surf_S_calcarine_y6'], 'fr_L_surf_S_central': df['fr_L_surf_S_central_y6'], 'fr_L_surf_S_cingul_Marginalis': df['fr_L_surf_S_cingul_Marginalis_y6'], 'fr_L_surf_S_circular_insula_ant': df['fr_L_surf_S_circular_insula_ant_y6'], 'fr_L_surf_S_circular_insula_inf': df['fr_L_surf_S_circular_insula_inf_y6'], 'fr_L_surf_S_circular_insula_sup': df['fr_L_surf_S_circular_insula_sup_y6'], 'fr_L_surf_S_collat_transv_ant': df['fr_L_surf_S_collat_transv_ant_y6'], 'fr_L_surf_S_collat_transv_post': df['fr_L_surf_S_collat_transv_post_y6'], 'fr_L_surf_S_front_inf': df['fr_L_surf_S_front_inf_y6'], 'fr_L_surf_S_front_middle': df['fr_L_surf_S_front_middle_y6'], 'fr_L_surf_S_front_sup': df['fr_L_surf_S_front_sup_y6'],  'fr_L_surf_S_intraparietandP_trans': df['fr_L_surf_S_intraparietandP_trans_y6'], 'fr_L_surf_S_oc_middleandLunatus': df['fr_L_surf_S_oc_middleandLunatus_y6'], 'fr_L_surf_S_oc_temp_lat': df['fr_L_surf_S_oc_temp_lat_y6'], 'fr_L_surf_S_oc_temp_medandLingual': df['fr_L_surf_S_oc_temp_medandLingual_y6'], 'fr_L_surf_S_orbital_H_Shaped': df['fr_L_surf_S_orbital_H_Shaped_y6'], 'fr_L_surf_S_orbital_lateral': df['fr_L_surf_S_orbital_lateral_y6'], 'fr_L_surf_S_orbital_med_olfact': df['fr_L_surf_S_orbital_med_olfact_y6'], 'fr_L_surf_S_parieto_occipital': df['fr_L_surf_S_parieto_occipital_y6'], 'fr_L_surf_S_postcentral': df['fr_L_surf_S_postcentral_y6'], 'fr_L_surf_S_precentral_inf_part': df['fr_L_surf_S_precentral_inf_part_y6'], 'fr_L_surf_S_precentral_sup_part': df['fr_L_surf_S_precentral_sup_part_y6'], 'fr_L_surf_S_suborbital': df['fr_L_surf_S_suborbital_y6'], 'fr_L_surf_S_subparietal': df['fr_L_surf_S_subparietal_y6'], 'fr_L_surf_S_temporal_inf': df['fr_L_surf_S_temporal_inf_y6'], 'fr_L_surf_S_temporal_sup': df['fr_L_surf_S_temporal_sup_y6']}


	free1, free2, free3, free4, free5, free6 = pd.DataFrame(frame_free1), pd.DataFrame(frame_free2), pd.DataFrame(frame_free3),  pd.DataFrame(frame_free4), pd.DataFrame(frame_free5), pd.DataFrame(frame_free6)
	df_free_lon = pd.concat([free1, free2, free3, free4, free5, free6])

	return df_free_lon

def scatterplot_2variables_in_df_(df2plt,xvar,yvar,figures_dir):
	"""scatterplot_2variables_in_df: scatter plot of 2 variables in dataframe
	Example: scatterplot_2variables_in_df(df,'siena_vel_12','siena_vaccloss_23')
	Args:df, xvar, yvar. 
	Out:
	"""
	def r2(x, y):
		return stats.pearsonr(x, y)[0] ** 2
	yearsx = xvar.split('_')[-1]; yearsy =  yvar.split('_')[-1]
	labelx = xvar.split('_')[0]; labely = yvar.split('_')[0]
	titlelabel = labelx + '-' + yearsx + '_' + labely + '-' + yearsy
	fig, ax = plt.subplots(figsize=(15,7))
	snsp = sns.jointplot(x=xvar, y=yvar, data=df2plt.replace([np.inf, -np.inf], np.nan), kind="reg", stat_func=r2);
	fig_file = os.path.join(figures_dir, 'joint_' + titlelabel + '.png')
	snsp.savefig(fig_file)


def remove_outliers_(df):
	"""
	"""
	df_orig = df.copy()
	low = .05
	high = .95
	#siena_cols = ['siena_vel_12','siena_vel_23','siena_vel_34','siena_vel_45','siena_vel_56','siena_vel_67']
	#siena_cols = ['siena_12','siena_23','siena_34','siena_45','siena_56','siena_67']
	fsl_cols = ['csf_rationat_visita1','gm_rationat_visita1','wm_rationat_visita1']
	df_fsl = df[fsl_cols]
	quant_df = df_fsl.quantile([low, high])
	
	print('Outliers: low= %.3f high= %.3f \n %s' %(low, high, quant_df))
	df_nooutliers = df_fsl[(df_fsl > quant_df.loc[low, fsl_cols]) & (df_fsl < quant_df.loc[high, fsl_cols])]
	df_outs = df_fsl[(df_fsl <= quant_df.loc[low, fsl_cols]) | (df_fsl >= quant_df.loc[high, fsl_cols])]
	df[fsl_cols] = df_nooutliers.to_numpy()
	# List of outliers
	reportfile = '/Users/jaime/github/papers/brain_age_estimation/outliers_Vol_tissues.txt'
	file_h= open(reportfile,"w+")
	print('Outliers Low : High %s %s' %(low, high))
	file_h.write('Outliers: low= %.3f high= %.3f \n %s \n' %(low, high, quant_df))
	for year in fsl_cols:
		outliers_y = df_outs.index[df_outs[year].notna() == True].tolist()
		file_h.write('\tOutliers Years :' + year + str(outliers_y) + '\n')
	return df

def convert_stringtofloat(dataframe):
	"""convert_stringtofloat: cast edad_ultimodx, edad_visita1,tpoi.j and siena_ij to float 
	Args:dataframe
	Out:dataframe
	"""
	# Change cx_cortov1 to float because all other dx corto are float (not strictly necessary)
	dataframe.dx_corto_visita1 = dataframe.dx_corto_visita1.astype(float)
	dataframe['edad_ultimodx'] = dataframe['edad_ultimodx'].str.replace(',','.').astype(float)
	dataframe['edad_visita1'] = dataframe['edad_visita1'].str.replace(',','.').astype(float)
	sv_toconvert = ['siena_12','siena_23','siena_34','siena_45','siena_56','siena_67','viena_12','viena_23','viena_34','viena_45','viena_56','viena_67']
	for ix in sv_toconvert:
		print('converting str to float in %s' % ix)
		dataframe[ix] = dataframe[ix].str.replace(',','.').astype(float)
	tpo_toconvert = ['tpo1.2','tpo1.3', 'tpo1.4','tpo1.5','tpo1.6','tpo1.7']
	for ix in tpo_toconvert:
		print('converting str to float in %s' % ix)
		dataframe[ix] = dataframe[ix].str.replace(',','.').astype(float)
	return dataframe

def cortical_parcels_lobes(df, year=None):
	"""https://surfer.nmr.mgh.harvard.edu/fswiki/CorticalParcellation
	"""

	lis = df.columns
	frontal = [s for s in lis if "front" in s]
	insula = [s for s in lis if "ins" in s]
	occipital = [s for s in lis if "oc" in s]
	parietal = [s for s in lis if "pariet" in s]
	precuneus = [s for s in lis if "precuneus" in s]
	#parietal = parietal + precuneus
	temporal = [s for s in lis if "temp" in s]
	cingulate = [s for s in lis if "cingul" in s]
	if year != None:

		frontal_y = [s for s in frontal if rf"_y{year}" in s]
		insula_y = [s for s in insula if rf"_y{year}" in s]
		occipital_y = [s for s in occipital if rf"_y{year}" in s]
		precuneus_y = [s for s in precuneus if rf"_y{year}" in s]
		parietal_y = [s for s in parietal if rf"_y{year}" in s]
		temporal_y = [s for s in temporal if rf"_y{year}" in s]
		cingulate_y = [s for s in cingulate if rf"_y{year}" in s]
	
	return frontal, insula, occipital, parietal, precuneus, temporal, cingulate


def getstring_corticals_aux(df):
	"""
	"""
	import re
	lis = df.columns
	pattern_y1 = re.compile(rf"fr_R_surf_\S+6\b")
	surf_y1 = list(filter(pattern_y1.match, lis))
	text = []
	for ele in surf_y1:
		#create sting to copy an paste
		
		string_in_string = '\'{}\': df[\'{}\']'.format(ele,ele)
		text.append(string_in_string)
	#pdb.set_trace()


def cortical_parcels(df,year=None):
	""" Select columns of cortical parcelation
	"""
	import re
	if year == None:
		year=1
	# fr_[R|L]_[surf|thick]_[S|G]_[ROI]_y[N] eg fr_R_surf_S_temporal_transverse_y6
	lis = df.columns
	#pattern_surf = re.compile(r"fr_[R|L]_surf_\S+y1\b")
	#surf = list(filter(pattern_surf.match, lis))
	pattern_surf_R = re.compile(rf"fr_R_surf_\S+y{year}\b")
	surf_R = list(filter(pattern_surf_R.match, lis))
	pattern_surf_L = re.compile(rf"fr_L_surf_\S+y{year}\b")
	surf_L = list(filter(pattern_surf_L.match, lis))
	# surf = surf_R + surf_L

	pattern_thick = re.compile(rf"fr_[R|L]_thick_\S+y{year}\b")
	thick = list(filter(pattern_thick.match, lis))
	
	pattern_thick_R = re.compile(rf"fr_R_thick_\S+y{year}\b")
	thick_R = list(filter(pattern_thick_R.match, lis))
	pattern_thick_L = re.compile(rf"fr_L_thick_\S+y{year}\b")
	thick_L = list(filter(pattern_thick_L.match, lis))
	thick == thick_R + thick_L
	
	pattern_sulcus = re.compile(rf"fr_[R|L]_surf_S_\S+y{year}\b")
	sulcus_surf = list(filter(pattern_sulcus.match, lis))
	pattern_giry = re.compile(rf"fr_[R|L]_surf_G_\S+y{year}\b")
	giry_surf = list(filter(pattern_giry.match, lis))
	
	pattern_sulcus_th = re.compile(rf"fr_[R|L]_thick_S_\S+y{year}\b")
	sulcus_surf_th = list(filter(pattern_sulcus_th.match, lis))
	pattern_giry_th = re.compile(rf"fr_[R|L]_thick_G_\S+y{year}\b")
	giry_surf_th = list(filter(pattern_giry_th.match, lis))

	return surf_R, surf_L, thick_R, thick_L, sulcus_surf, giry_surf, sulcus_surf_th, giry_surf_th

def add_tissue_ratio_cols(dataframe):
	"""
	"""
	total = dataframe['csf_volume_visita1'] + dataframe['gm_volume_visita1'] + dataframe['wm_volume_visita1']
	dataframe['csf_rationat_visita1'] = dataframe['csf_volume_visita1']/total
	dataframe['wm_rationat_visita1'] = dataframe['wm_volume_visita1']/total
	dataframe['gm_rationat_visita1'] = dataframe['gm_volume_visita1']/total
	return dataframe

def train_nonlinear(df, figures_dir):
	"""
	"""
	from sklearn.model_selection import train_test_split
	from sklearn import ensemble
	from sklearn.metrics import mean_squared_error

	XY = df[['csf_rationat_visita1', 'wm_rationat_visita1', 'gm_rationat_visita1','conversionmci','edad_visita1']]
	XY = XY.dropna()
	n_subjects = XY.shape[0]	
	#converttomci
	col = XY.iloc[:,-2].map({0:'b', 1:'r'})
	label = 'csf' + 'wm' + 'gm' 
	#label = 'gm' 
	#label = 'wm'
	#label = 'csf'
	if label == 'csf' + 'wm' + 'gm':
		X = XY.iloc[:,0:2]
	elif label == 'csf':
		X = XY.iloc[:,0].values.reshape(-1,1)
	elif label == 'wm':
		X = XY.iloc[:,1].values.reshape(-1,1)
	elif label == 'gm':
		X = XY.iloc[:,2].values.reshape(-1,1)
	y = XY.iloc[:,-1]
	# random split
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

	original_params = {'n_estimators': 1000, 'max_leaf_nodes': 4, 'max_depth': None, 'random_state': 2,
                   'min_samples_split': 5}

	plt.figure()

	for label, color, setting in [('No shrinkage', 'orange',
                               {'learning_rate': 1.0, 'subsample': 1.0}),
                              ('learning_rate=0.1', 'turquoise',
                               {'learning_rate': 0.1, 'subsample': 1.0}),
                              ('subsample=0.5', 'blue',
                               {'learning_rate': 1.0, 'subsample': 0.5}),
                              ('learning_rate=0.1, subsample=0.5', 'gray',
                               {'learning_rate': 0.1, 'subsample': 0.5}),
                              ('learning_rate=0.1, max_features=2', 'magenta',
                               {'learning_rate': 0.1, 'max_features': 2})]:
		params = dict(original_params)
		params.update(setting)
		# Fit regression model
		clf = ensemble.GradientBoostingRegressor(**params)
		clf.fit(X_train, y_train)
		mse = mean_squared_error(y_test, clf.predict(X_test))
		print("MSE: %.4f" % mse)
		# Plot training deviance
		test_score = np.zeros((params['n_estimators'],), dtype=np.float64)
		for i, y_pred in enumerate(clf.staged_predict(X_test)):
			test_score[i] = clf.loss_(y_test, y_pred)
		plt.figure(figsize=(12, 6))
		plt.subplot(1, 2, 1)
		plt.title('Deviance')
		plt.plot(np.arange(params['n_estimators']) + 1, clf.train_score_, 'b-',label='Training Set Deviance')
		plt.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-',label='Test Set Deviance')
		plt.legend(loc='upper right')
		plt.xlabel('Boosting Iterations')
		plt.ylabel('Deviance')
		plt.savefig(os.path.join(figures_dir, 'GradientBoost_' + label + '.png'))

def train_linear_regression(df, figures_dir):
	"""lm_volxintensity: Train a linear model on voxel intensity histogram
	"""
	from sklearn.model_selection import train_test_split
	from sklearn.linear_model import LinearRegression, RidgeCV
	#subjects with 2 visits
	
	XY = df[['csf_rationat_visita1', 'wm_rationat_visita1', 'gm_rationat_visita1','conversionmci','edad_visita1']]
	XY = XY.dropna()
	n_subjects = XY.shape[0]	
	#converttomci
	col = XY.iloc[:,-2].map({0:'b', 1:'r'})
	label = 'csf' + 'wm' + 'gm' 
	label = 'gm' 
	#label = 'wm'
	#label = 'csf'
	if label == 'csf' + 'wm' + 'gm':
		X = XY.iloc[:,0:2]
	elif label== 'csf':
		X = XY.iloc[:,0].values.reshape(-1,1)
	elif label == 'wm':
		X = XY.iloc[:,1].values.reshape(-1,1)
	elif label == 'gm':
		X = XY.iloc[:,2].values.reshape(-1,1)
	y = XY.iloc[:,-1]
	# random split
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

	#train, test = train_test_split(np.arange(n_subjects), test_size=0.2,random_state=0)
	print('Size of the test set : {} patients'.format(X_test.shape[0]))
	#model = RidgeCV()
	model = LinearRegression()
	model.fit(X_train, y_train)
	predictions = model.predict(X_test)
	true_values = y_test
	# Plot the results and compute the MAE

	fig, ax = plt.subplots()

	ax.scatter(predictions, true_values, s=10)
	#ax.scatter(predictions, true_values, s=10, c=col[X_test.index.values])
	#xmin, xmax, ymin, ymax = plt.axis()
	lims = [np.min([ax.get_xlim(), ax.get_ylim()]), np.max([ax.get_xlim(), ax.get_ylim()])]
	ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0)
	ax.set_aspect('equal')
	ax.set_xlim(lims)
	ax.set_ylim(lims)
	ax.grid()
	ax.set_xlabel('Predictions')
	ax.set_ylabel('True values')
	mae = np.mean(np.abs(predictions - true_values))
	ax.set_title(label + ' Mean Absolute Error : {:.2f} years'.format(mae))
	fig.savefig(os.path.join(figures_dir, 'scatter_predictions_' + label + '.png'))

	return

def tissue_histogram(imgpath, segmentationpath, figures_dir):
	"""tissue_histogram
	"""
	# Get subject if from path (works for FSL)
	subjectid = imgpath.split('pv_')[1].split('.anat')[0]
	# Load the files associated to raw data and segmentation

	image = nib.load(imgpath).get_data()
	segmentation = nib.load(segmentationpath).get_data()
	#Remove outside pixels
	skullstripped_image = image * (segmentation > 0)
	# visualize histogram
	bins = 200
	plt.figure(figsize=(9, 6))
	#plt.subplot(121)
	histogram_bins = plt.hist(skullstripped_image[skullstripped_image > 0], bins=bins, density=True);
	plt.subplot(111)
	# Use segmentation and superpose the histograms of CSF (blue), \
	# white matter (green) and grey matter (red) intensity values
	plt.hist([skullstripped_image[segmentation == i] for i in [1, 2, 3]], stacked=True, bins=200, density=True);
	plt.title(r'Histogram of voxel intensities:%s' %subjectid)
	figname = os.path.join(figures_dir, 'histogram_' + subjectid + '.png')
	plt.savefig(figname)
	return histogram_bins

def compute_SHAP(model, X):
	"""
	"""
	X100 = shap.utils.sample(X, 100) # 100 instances for use as the background distribution
	# Plot Partial dependence
	fig = plt.figure(figsize=(19, 15))
	shap.plots.partial_dependence('free_R_Thal', model.predict, X100, ice=False,model_expected_value=True, feature_expected_value=True)
	fig_name = os.path.join(figures_dir, 'PDP_rThal-Chrono' + '.png')
	plt.savefig(fig_name)

	# compute the SHAP values for the linear model
	explainer = shap.Explainer(model.predict, X100)
	shap_values = explainer(X)

	# make a standard partial dependence plot
	fig = plt.figure(figsize=(19, 15))
	sample_ind = 20
	shap.partial_dependence_plot("free_R_Thal", model.predict, X100, model_expected_value=True,feature_expected_value=True, ice=False,shap_values=shap_values[sample_ind:sample_ind+1,:])
	fig_name = os.path.join(figures_dir, 'PDP_rThal-Chrono2' + '.png')
	plt.savefig(fig_name)

	# the waterfall_plot shows how we get from shap_values.base_values to model.predict(X)[sample_ind]
	fig = plt.figure(figsize=(19, 15))
	shap.plots.waterfall(shap_values[sample_ind], max_display=14)
	fig_name = os.path.join(figures_dir, 'Cascade_rThal-Chrono2' + '.png')
	plt.savefig(fig_name)
	pdb.set_trace()	
	# the waterfall_plot shows how we get from shap_values.base_values to model.predict(X)[sample_ind]
	plt.figure()
	shap.plots.waterfall(shap_values[sample_ind], max_display=14)


def RVM_analysis(X_train,y_train, X_test, y_test, msg=None):
	"""
	"""
	from skrvm import RVR
	reg = RVR(kernel='linear')
	reg.fit(X_train, y_train)
	# Prediction with test data
	pdb.set_trace()


def feature_importance_Analysis(X_test, y_test, best_model, feature_importance):
	"""SHAP results by lobe, hemisphere and G,L. Get confusion Matrix
	"""
	from sklearn.metrics import ConfusionMatrixDisplay
	import xgboost as xgb
	print('CM for feature_importance_Analysis')

	
	dtest = xgb.DMatrix(X_test, label=y_test)

	# Scatter plot predictions
	df2plot = X_test[['education', 'apoe', 'sex']]
	df2plot['ytest'] = y_test
	df2plot['ypred'] = best_model.predict(dtest)
	ypp = pd.Series(best_model.predict(dtest), index=df2plot.index)
	df2plot['ypred'] = ypp
	df2plot['diff'] = df2plot['ytest']-df2plot['ypred']
	plot_predictions_grouped(df2plot)

	# Confusion Matrix of feature importance
	# Filter by hemisphere, lobe and SG. Get dataframes one for each condition
	# df_R, df_L, df_S, Df_G, df_Fro, df_Occ, df_ins, 
	ixR = feature_importance['col_name'].str.contains('fr_R_'); df_R= feature_importance[ixR]
	ixL = feature_importance['col_name'].str.contains('fr_L_'); df_L= feature_importance[ixL]
	ixS = feature_importance['col_name'].str.contains('_S_'); df_S= feature_importance[ixS]
	ixG = feature_importance['col_name'].str.contains('_G_'); df_G= feature_importance[ixG]

	ixF = feature_importance['col_name'].str.contains('front'); df_F= feature_importance[ixF]
	ixI = feature_importance['col_name'].str.contains('ins'); df_I= feature_importance[ixI]
	ixO = feature_importance['col_name'].str.contains('oc'); df_O= feature_importance[ixO]
	ixP = feature_importance['col_name'].str.contains('pariet'); df_P= feature_importance[ixP]
	ixT = feature_importance['col_name'].str.contains('temp'); df_T= feature_importance[ixT]
	ixU = feature_importance['col_name'].str.contains('precuneus'); df_U= feature_importance[ixU]
	ixC = feature_importance['col_name'].str.contains('cingul'); df_C= feature_importance[ixC]
	#
	total = feature_importance.mean()
	
	# hemisphere
	fig = plt.figure()
	ax = fig.add_subplot(111)
	labels = ['LH', 'RH']
	cm = np.empty([1, 2], dtype = float) # R and L
	cm[0,0] = df_L.mean()[0];  cm[0,1] = df_R.mean()[0]
	cm = cm/cm.sum()
	cax = ax.matshow(cm, interpolation='nearest')
	for (i, j), z in np.ndenumerate(cm):ax.text(j, i, '{:0.3f}'.format(z), ha='center', va='center',bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))
	fig.colorbar(cax)
	#ax.axes.xaxis.set_visible(False)
	ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)
	ax.axes.yaxis.set_visible(False)
	ax.set_xticklabels((['']+labels))
	ax.set_title('SHAP feature importance by Hemisphere', pad=30)
	fig_name = os.path.join(figures_dir, 'CM_hem' + '.png')
	plt.savefig(fig_name,bbox_inches='tight',dpi=400)
	
	# SG
	fig = plt.figure()
	ax = fig.add_subplot(111)
	labels = ['Sulci', 'Gyri']
	cm = np.empty([1, 2], dtype = float) # S and G
	cm[0,0] = df_S.mean()[0]; cm[0,1] = df_G.mean()[0]
	cm = cm/cm.sum()
	cax = ax.matshow(cm, interpolation='nearest')
	for (i, j), z in np.ndenumerate(cm):ax.text(j, i, '{:0.3f}'.format(z), ha='center', va='center',bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))
	fig.colorbar(cax)
	ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)
	ax.axes.yaxis.set_visible(False)
	ax.set_xticklabels((['']+labels))
	ax.set_title('SHAP feature importance by Sulci or Gyri', pad=30)
	fig_name = os.path.join(figures_dir, 'CM_SG' + '.png')
	plt.savefig(fig_name,bbox_inches='tight',dpi=400)

	# lobe
	fig = plt.figure()
	ax = fig.add_subplot(111)
	labels = ['Fro', 'Temp', 'Par', 'Occ','Ins']
	cm = np.empty([1, 5], dtype = float) #5 cols: fro, ins, oc, par, temp
	cm[0,0] = df_F.mean()[0]; cm[0,1] = df_T.mean()[0]; cm[0,2] = df_O.mean()[0]
	cm[0,3] = df_P.mean()[0]; cm[0,4] = df_I.mean()[0];
	cm = cm/cm.sum()
	cax = ax.matshow(cm, interpolation='nearest')
	for (i, j), z in np.ndenumerate(cm):ax.text(j, i, '{:0.3f}'.format(z), ha='center', va='center',bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))
	fig.colorbar(cax)
	ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)
	ax.axes.yaxis.set_visible(False)
	ax.set_xticklabels((['']+labels))
	ax.set_title('SHAP feature importance by Lobe', pad=30)
	fig_name = os.path.join(figures_dir, 'CM_lobe' + '.png')
	plt.savefig(fig_name,bbox_inches='tight',dpi=400)

	#hemisphere and SG
	fig = plt.figure()
	ax = fig.add_subplot(111)
	labelsY = ['LH', 'RH']
	labelsX =  ['Sulci', 'Gyri']
	cm = np.empty([2, 2], dtype = float) #rows R,L cols S, G
	ixRS = df_R['col_name'].str.contains('_S_'); dfRS = df_R[ixRS]
	ixRG = df_R['col_name'].str.contains('_G_'); dfRG = df_R[ixRG]
	ixLS = df_L['col_name'].str.contains('_S_'); dfLS = df_L[ixLS]
	ixLG = df_L['col_name'].str.contains('_G_'); dfLG = df_L[ixLG]	
	cm[0,0] = dfLS.mean()[0]
	cm[0,1] = dfLG.mean()[0]
	cm[1,0] = dfRS.mean()[0]
	cm[1,1] = dfRG.mean()[0]
	cm = cm/cm.sum()
	cax = ax.matshow(cm, interpolation='nearest')
	for (i, j), z in np.ndenumerate(cm):ax.text(j, i, '{:0.3f}'.format(z), ha='center', va='center',bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))
	fig.colorbar(cax)
	ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)
	ax.set_xticklabels((['']+labelsX))
	ax.set_yticklabels((['']+labelsY))
	ax.set_title('SHAP feature importance by Hemisphere & Cortical Sulcus Gyri', pad=30)
	fig_name = os.path.join(figures_dir, 'CM_RL-SG' + '.png')
	plt.savefig(fig_name,bbox_inches='tight',dpi=400)

	#Lobe and SG
	fig = plt.figure()
	ax = fig.add_subplot(111)
	labelsY = ['Fro', 'Temp', 'Par', 'Occ','Ins']
	labelsX =  ['Sulci', 'Gyri']
	cm = np.empty([5, 2], dtype = float)
	ixFS = df_F['col_name'].str.contains('_S_'); dfFS = df_F[ixFS]
	ixFG = df_F['col_name'].str.contains('_G_'); dfFG = df_F[ixFG]
	ixTS = df_T['col_name'].str.contains('_S_'); dfTS = df_T[ixTS]
	ixTG = df_T['col_name'].str.contains('_G_'); dfTG = df_T[ixTG]
	ixPS = df_P['col_name'].str.contains('_S_'); dfPS = df_P[ixPS]
	ixPG = df_P['col_name'].str.contains('_G_'); dfPG = df_P[ixPG]
	ixOS = df_O['col_name'].str.contains('_S_'); dfOS = df_O[ixOS]
	ixOG = df_O['col_name'].str.contains('_G_'); dfOG = df_O[ixOG]
	ixIS = df_I['col_name'].str.contains('_S_'); dfIS = df_I[ixIS]
	ixIG = df_I['col_name'].str.contains('_G_'); dfIG = df_I[ixIG]
	cm[0,0] = dfFS.mean()[0];cm[0,1] = dfFG.mean()[0]
	cm[1,0] = dfTS.mean()[0]; cm[1,1] = dfTG.mean()[0]
	cm[2,0] = dfPS.mean()[0];cm[2,1] = dfPG.mean()[0]
	cm[3,0] = dfOS.mean()[0]; cm[3,1] = dfOG.mean()[0]
	cm[4,0] = dfIS.mean()[0]; cm[4,1] = dfIG.mean()[0]
	cm = cm/cm.sum()
	cax = ax.matshow(cm, interpolation='nearest')
	for (i, j), z in np.ndenumerate(cm):ax.text(j, i, '{:0.3f}'.format(z), ha='center', va='center',bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))
	fig.colorbar(cax)
	ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)
	ax.set_xticklabels((['']+labelsX))
	ax.set_yticklabels((['']+labelsY))
	ax.set_title('SHAP feature importance by Lobe & Cortical Sulcus Gyri', pad=30)
	fig_name = os.path.join(figures_dir, 'CM_Lobe-SG' + '.png')
	plt.savefig(fig_name,bbox_inches='tight',dpi=400)

	# Lobe and Hemisphere
	fig = plt.figure()
	ax = fig.add_subplot(111)
	labelsY = ['Fro', 'Temp', 'Par', 'Occ','Ins']
	labelsX =  ['LH', 'RH']
	cm = np.empty([5, 2], dtype = float)
	ixFL = df_F['col_name'].str.contains('_L_'); dfFL = df_F[ixFL]
	ixFR = df_F['col_name'].str.contains('_R_'); dfFR = df_F[ixFR]
	ixTL = df_T['col_name'].str.contains('_L_'); dfTL = df_T[ixTL]
	ixTR = df_T['col_name'].str.contains('_R_'); dfTR = df_T[ixTR]
	ixPL = df_P['col_name'].str.contains('_L_'); dfPL = df_P[ixPL]
	ixPR = df_P['col_name'].str.contains('_R_'); dfPR = df_P[ixPR]
	ixOL = df_O['col_name'].str.contains('_L_'); dfOL = df_O[ixOL]
	ixOR = df_O['col_name'].str.contains('_R_'); dfOR = df_O[ixOR]
	ixIL = df_I['col_name'].str.contains('_L_'); dfIL = df_I[ixIL]
	ixIR = df_I['col_name'].str.contains('_R_'); dfIR = df_I[ixIR]
	cm[0,0] = dfFL.mean()[0];cm[0,1] = dfFR.mean()[0]
	cm[1,0] = dfTL.mean()[0]; cm[1,1] = dfTR.mean()[0]
	cm[2,0] = dfPL.mean()[0];cm[2,1] = dfPR.mean()[0]
	cm[3,0] = dfOL.mean()[0]; cm[3,1] = dfOR.mean()[0]
	cm[4,0] = dfIL.mean()[0]; cm[4,1] = dfIR.mean()[0]
	cm = cm/cm.sum()
	cax = ax.matshow(cm, interpolation='nearest')
	for (i, j), z in np.ndenumerate(cm):ax.text(j, i, '{:0.3f}'.format(z), ha='center', va='center',bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))
	fig.colorbar(cax)
	ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)
	ax.set_xticklabels((['']+labelsX))
	ax.set_yticklabels((['']+labelsY))
	ax.set_title('SHAP feature importance by Lobe & Hemisphere', pad=30)
	fig_name = os.path.join(figures_dir, 'CM_Lobe-Hem' + '.png')
	plt.savefig(fig_name,bbox_inches='tight',dpi=400)

	return df2plot


def XGBoost_SHAP(model, X_train,y_train, X_test, y_test, msg=None):
	""" return dataframe with ordered feature importance
	""" 
	# Rename X_text columns

	explainer = shap.TreeExplainer(model)
	shap_values = explainer.shap_values(X_test)
	print(' The SHAP explainer EV explainer.expected_value == %.3f' %explainer.expected_value)

	vals= np.abs(shap_values).mean(0)
	feature_importance = pd.DataFrame(list(zip(X_train.columns,vals)),columns=['col_name','feature_importance_vals'])
	feature_importance.sort_values(by=['feature_importance_vals'],ascending=False,inplace=True)
	feature_importance.head()

	fig = plt.figure()
	ax = shap.summary_plot(shap_values, X_test, plot_type="bar")
	fig_name = os.path.join(figures_dir, 'SHAP_Importance_XGBooost_' + msg + '.png')
	fig.savefig(fig_name, bbox_inches='tight',dpi=400)

	fig = plt.figure()
	plt.rcParams['ytick.labelsize'] = 8
	shap.summary_plot(shap_values, X_test)
	fig_name = os.path.join(figures_dir, 'SHAP_BF_Importance_XGBooost_' + msg + '.png')
	plt.savefig(fig_name, bbox_inches='tight',dpi=400)

	feat1 = feature_importance.iloc[0,0]
	feat2 = feature_importance.iloc[1,0]
	feat3 = feature_importance.iloc[2,0]
	fig = plt.figure()
	shap.dependence_plot(feat1, shap_values, X_test)
	fig_name = os.path.join(figures_dir, 'feat1_SHAP_BF_Importance_XGBooost_' + msg + '.png')
	plt.savefig(fig_name, bbox_inches='tight',dpi=400)	
	fig = plt.figure()
	shap.dependence_plot(feat2, shap_values, X_test)
	fig_name = os.path.join(figures_dir, 'feat2_SHAP_BF_Importance_XGBooost_' + msg + '.png')
	plt.savefig(fig_name, bbox_inches='tight',dpi=400)	
	fig = plt.figure()
	shap.dependence_plot(feat3, shap_values, X_test)
	fig_name = os.path.join(figures_dir, 'feat3_SHAP_BF_Importance_XGBooost_' + msg + '.png')
	plt.savefig(fig_name, bbox_inches='tight',dpi=400)	
	
	# Scatter Plots
	shap_values2 = explainer(X_test)
	fig = plt.figure()
	shap.plots.scatter(shap_values2[:,'fr_BrainSegVol_to_eTIV'])
	fig_name = os.path.join(figures_dir, 'eTIV_' + msg + '.png')
	plt.savefig(fig_name, bbox_inches='tight',dpi=400)

	# Which features interact with the most important feature
	inds = shap.utils.potential_interactions(shap_values2[:, 'fr_BrainSegVol_to_eTIV'], shap_values2)
	# make plots colored by each of the top three possible interacting features
	for i in range(feature_importance.shape[0]):
		fig = plt.figure();
		shap.plots.scatter(shap_values2[:,"fr_BrainSegVol_to_eTIV"],color=shap_values2[:,inds[i]]);
		fig_name = os.path.join(figures_dir, 'eTIV_vs' + str(i) + msg + '.png');
		plt.savefig(fig_name, bbox_inches='tight',dpi=400);	
	
	return feature_importance



def XGBoost_permut_importance(model, X_train,y_train, X_test, y_test, msg=None):
	""" Uses feature_importances_ works for XGBRegressr not for xgb
	"""
	from sklearn.inspection import permutation_importance
	howmany= 12
	
	fig, ax = plt.subplots(figsize=(30, 30))

	sorted_idx = model.feature_importances_.argsort()
	features = model.get_booster().feature_names
	features_array = np.array(features)

	plt.barh(features_array[sorted_idx][:howmany], model.feature_importances_[sorted_idx][:howmany])
	plt.xlabel("Xgboost Feature Importance")
	fig_name = os.path.join(figures_dir, 'Importance_XGBooost_'+ 'Nb_' + str(howmany) + msg + '.png')
	plt.savefig(fig_name)

	print('The '+ str(howmany) + ' most important XGB based features:');print(features_array[sorted_idx][:howmany])
	# Permutation based Importance
	#Randomly shuffle each feature and compute the change in the model’s performance
	print('Running permutation_importance....')
	fig, ax = plt.subplots(figsize=(30, 20))

	perm_importance = permutation_importance(model, X_test, y_test)
	sorted_idx = perm_importance.importances_mean.argsort()
	plt.barh(features_array[sorted_idx][:howmany], perm_importance.importances_mean[sorted_idx][:howmany])
	plt.xlabel("Permutation Importance")
	fig_name = os.path.join(figures_dir, 'XGBooost_PermImp_'+ 'Nb_' + str(howmany) + msg + '.png')
	plt.savefig(fig_name)
	print('The '+ str(howmany) + ' most  important permutation based features:');print(features_array[sorted_idx][:howmany])
	#xgb.plot_importance(model)

def plot_predictions_grouped(df):
	"""
	"""
	# scatter plot age obs vs pred
	plt.figure()
	ax = sns.scatterplot(data=df, x="ytest", y="ypred", hue="sex")
	ax.set_xlabel('Age Observed'); ax.set_ylabel('Age Predicted')
	leg_handles = ax.get_legend_handles_labels()[0]
	ax.legend(leg_handles, ['Male', 'Female'])
	fig_name = os.path.join(figures_dir, 'XGBooost_preds_sex.png')
	plt.savefig(fig_name,bbox_inches='tight',dpi=400)
	
	# regression plot age obs vs pred
	plt.figure()
	ax = sns.regplot(data=df, x="ytest", y="ypred")
	ax.set_xlabel('Age Observed'); ax.set_ylabel('Age Predicted')
	#leg = plt.legend()
	#leg.get_texts()[0].set_text('Male'); leg.get_texts()[1].set_text('Female')
	fig_name = os.path.join(figures_dir, 'XGBooost_reg_preds_sex.png')
	plt.savefig(fig_name,bbox_inches='tight',dpi=400)

	fig, ax = plt.subplots()
	sns.lmplot(data=df, x="ytest", y="ypred", hue="sex",markers=["o", "x"], palette="Set1", legend=False)
	ax.set_xlabel('Age Observed'); ax.set_ylabel('Age Predicted')
	leg = plt.legend()
	leg.get_texts()[0].set_text('Male'); leg.get_texts()[1].set_text('Female')
	fig_name = os.path.join(figures_dir, 'XGBooost_lmplot_preds_sex.png')
	plt.savefig(fig_name,bbox_inches='tight',dpi=400)

	fig, ax = plt.subplots()
	sns.lmplot(data=df, x="ytest", y="ypred", hue="sex",col="education", markers=["o", "x"], palette="Set1", legend=False)
	ax.set_xlabel('Age Observed'); ax.set_ylabel('Age Predicted')
	leg = plt.legend()
	leg.get_texts()[0].set_text('Male'); leg.get_texts()[1].set_text('Female')
	fig_name = os.path.join(figures_dir, 'XGBooost_edulmplot_preds_sex.png')
	plt.savefig(fig_name,bbox_inches='tight',dpi=400)


	plt.figure()
	ax = sns.scatterplot(data=df, x="ytest", y="ypred", hue="education")
	ax.set_xlabel('Age Observed'); ax.set_ylabel('Age Predicted')
	leg_handles = ax.get_legend_handles_labels()[0]
	ax.legend(leg_handles, ['No School', 'Primary', 'Secondary', ' University'])
	fig_name = os.path.join(figures_dir, 'XGBooost_preds_edu.png')
	plt.savefig(fig_name,bbox_inches='tight',dpi=400)

	plt.figure()
	ax = sns.scatterplot(data=df, x="ytest", y="ypred", hue="apoe")
	ax.set_xlabel('Age Observed'); ax.set_ylabel('Age Predicted')
	leg_handles = ax.get_legend_handles_labels()[0]
	ax.legend(leg_handles, ['e2,e3', 'e4', 'e4e4'])
	fig_name = os.path.join(figures_dir, 'XGBooost_preds_apoe.png')
	plt.savefig(fig_name,bbox_inches='tight',dpi=400)

	# Histogramas
	fig, ax = plt.subplots()
	line1 = sns.distplot(df['ypred'], label='Age Pred')
	line2  = sns.distplot(df['ytest'], label='Age Obs')
	plt.legend()
	#first_legend = ax.legend(handles =[line1], loc ='upper center')
	ax.set_xlabel('Age Obs and Age Predicted KDE')
	#ax.add_artist(first_legend)
	#ax.legend(handles =[line2], loc ='lower center')
	fig_name = os.path.join(figures_dir, 'XGBooost_histo.png')
	plt.savefig(fig_name,bbox_inches='tight',dpi=400)

	fig = plt.subplots()
	with sns.axes_style('white'):
		ax = sns.jointplot("ypred", "ytest", df, kind='hex')
	fig_name = os.path.join(figures_dir, 'XGBooost_hex.png')
	plt.savefig(fig_name,bbox_inches='tight',dpi=400)

	plt.figure()
	with sns.axes_style(style='ticks'):
		g = sns.factorplot("education", "diff", "sex", data=df, kind="box")
		g.set_axis_labels("education", "Age Obs- Age Pred");
	fig_name = os.path.join(figures_dir, 'XGBooost_Factor.png')
	plt.savefig(fig_name)

	plt.figure()
	sns.jointplot("ytest", "ypred", data=df, kind='reg');
	fig_name = os.path.join(figures_dir, 'XGBooost_reg.png')
	plt.savefig(fig_name)

	plt.figure()
	sns.distplot(df['diff'], kde=False);
	plt.axvline(-2, color="k", linestyle="--");
	plt.axvline(2, color="k", linestyle="--");
	fig_name = os.path.join(figures_dir, 'XGBooost_Hislines.png')
	plt.savefig(fig_name)

	plt.figure()
	ax= sns.violinplot("sex", "diff", data=df,palette=["lightblue", "lightpink"]);
	ax.set_xticklabels(['M', 'F'])
	fig_name = os.path.join(figures_dir, 'XGBooost_sexdiffs.png')
	plt.savefig(fig_name)

	plt.figure()
	g = sns.lmplot('ytest', 'ypred', col='sex', data=df,markers=".", scatter_kws=dict(color='c'))
	g.map(plt.axhline, y=df['ytest'].mean(), color="k", ls=":");
	fig_name = os.path.join(figures_dir, 'XGBooost_sexscats.png')
	plt.savefig(fig_name)

	plt.figure()	
	r_, p_ = stats.pearsonr(df['ytest'], df['ypred'])
	g = sns.jointplot(x=df['ytest'], y=df['ypred'], kind='reg', color='royalblue')
	g.ax_joint.annotate(f'$\\rho = {r_:.3f}, p = {p_:.3f}$',xy=(0.1, 0.9), xycoords='axes fraction',ha='left', va='center',bbox={'boxstyle': 'round', 'fc': 'powderblue', 'ec': 'navy'})
	g.ax_joint.scatter(df['ytest'], df['ypred'])
	g.set_axis_labels(xlabel='AgeObs', ylabel='AgePred', size=15)
	plt.tight_layout()
	fig_name = os.path.join(figures_dir, 'XGBooost_lins.png')
	plt.savefig(fig_name)

	# obs - pred >2 older than predicted 
	oldertanpred = df['diff']>2; df_otp = df[oldertanpred] #188
	youngertanpred = df['diff']<-2; df_ytp = df[youngertanpred] #195
	from scipy.stats import ttest_ind
	males = df_otp[df_otp['sex']==0]
	females = df_otp[df_otp['sex']==1]
	ttest_ind(males['diff'], females['diff'])

	males = df_ytp[df_ytp['sex']==0]
	females = df_ytp[df_ytp['sex']==1]
	ttest_ind(males['diff'], females['diff'])



def scatterplot_predictions(model, ypred, yobs, df=None):
	"""
	"""


	fig, axes = plt.subplots(1, 1, figsize=(16, 8))
	plt.scatter(yobs, ypred, c='blue')
	plt.xlabel('Observed Age')
	plt.ylabel('PPredicted Age')
	plt.tight_layout()
	fig_name = os.path.join(figures_dir, 'XGBooost_scatter.png')
	plt.savefig(fig_name)
	# Plot prediction errors
	_, ax = plt.subplots()
	ax.scatter(x = range(0, yobs.size), y=yobs, c = 'blue', label = 'Actual', alpha = 0.3)
	ax.scatter(x = range(0, yobs.size), y=ypred, c = 'red', label = 'Predicted', alpha = 0.3)
	plt.title('Actual and predicted age')
	plt.xlabel('Observations')
	plt.ylabel('Age')
	plt.legend()
	fig_name = os.path.join(figures_dir, 'XGBooost_scatter_errors.png')
	plt.savefig(fig_name)
	# Plot errors histogram
	diff = yobs - ypred
	diff = pd.DataFrame(diff, columns = ['Age PvsO'])

	diff.hist(bins = 40)
	plt.title('Histogram of prediction errors')
	plt.xlabel('Age prediction error obs-pred')
	plt.ylabel('Frequency')
	fig_name = os.path.join(figures_dir, 'XGBooost_scatter_hist.png')
	plt.savefig(fig_name)
	# ID who look younger and who look older and study thei distro properties
	diff.set_index(df.index, inplace=True)
	df['error'] = diff
	# Study properties of dataset based on error : younger older 
	pdb.set_trace()	

def evaluate_xgboost(xgHyper, params, X_test, y_test):
	""" Metrics of XGBoost Regressor best model
	""" 
	import xgboost as xgb
	from sklearn.metrics import explained_variance_score, max_error, mean_absolute_error, mean_squared_error, median_absolute_error, mean_absolute_percentage_error, r2_score
	from sklearn.dummy import DummyRegressor


	dtest = xgb.DMatrix(X_test, label=y_test)
	y_pred =xgHyper.predict(dtest)
	print('Best Model MAE %.3f' %mean_absolute_error(y_pred, y_test))
		
	evs = explained_variance_score(y_test, y_pred)
	mxe = max_error(y_test, y_pred)
	mae = mean_absolute_error(y_test, y_pred)
	mse = mean_squared_error(y_test, y_pred)
	# mean_absolute_error, may ignore the small magnitude values and only reflected the error in prediction of highest magnitude value. 
	# MAPE solves this because it calculates relative percentage error with respect to actual output.
	mape = mean_absolute_percentage_error(y_test, y_pred)
	# robust to outliers.
	medae = median_absolute_error(y_test, y_pred)
	# proportion of variance (of y) that has been explained by the independent variables in the model.
	# indication of goodness of fit and therefore a measure of how well unseen samples are likely to be predicted by the model, through the proportion of explained variance.
	# Best R2 1.0 and it can be negative. A constant model that always predicts the expected value of y, disregarding the input features, would get a R² score of 0.0.
	r2 = r2_score(y_test, y_pred)
	print('Best Model Metrics Results. Explained Variance= %.3f, Max Error = %.3f, MeanAbsErr=%.3f, MeanSqErr= %.3f' %(evs, mxe, mae, mse))
	print('\t MeanAbsPercErr= %.3f, median_absolute_error = %.3f, r2_score=%.3f' %(mape, medae, r2))
	evalmetrics = dict(evs=evs, mae=mae, mxe=mxe, mse=mse, mape=mape,medae=medae,r2=r2)
	return evalmetrics

def DummyRegressor_metrics(xgmodel, X_train,y_train, X_test, y_test):
	"""
	"""

	from sklearn.dummy import DummyRegressor
	from sklearn.metrics import mean_squared_error, r2_score, median_absolute_error,mean_absolute_error
	import xgboost as xgb

	print('DummyRegressor_metrics compared to non dummy model')
	lm_dummy_mean = DummyRegressor(strategy = 'mean').fit(X_train, y_train)
	lm_dummy_median = DummyRegressor(strategy = 'median').fit(X_train, y_train)
	y_predict_dummy_mean = lm_dummy_mean.predict(X_test)
	y_predict_dummy_median = lm_dummy_median.predict(X_test)

	dtrain = xgb.DMatrix(X_train,label=y_train)
	dtest = xgb.DMatrix(X_test, label=y_test)
	y_predict = xgmodel.predict(dtest)


	print("Mean squared error (dummy): {:.3f}".format(mean_squared_error(y_test,y_predict_dummy_mean)))
	print("Mean squared error (XGB best model): {:.3f}".format(mean_squared_error(y_test, y_predict)))
	print("Median absolute error (dummy): {:.3f}".format(median_absolute_error(y_test, y_predict_dummy_median)))
	print("Median absolute error (XGB best model): {:.3f}".format(median_absolute_error(y_test, y_predict)))
	print("Mean absolute error (dummy): {:.3f}".format(mean_absolute_error(y_test, y_predict_dummy_median)))
	print("Mean absolute error (XGB best model): {:.3f}".format(mean_absolute_error(y_test, y_predict)))

	print("r2_score (dummy mean): {:.3f}".format(r2_score(y_test, y_predict_dummy_mean)))
	print("r2_score (dummy median): {:.3f}".format(r2_score(y_test, y_predict_dummy_median)))
	print("r2_score (XGB best model): {:.3f}".format(r2_score(y_test, y_predict)))
	dummy_metrics = dict(mse=mean_squared_error(y_test,y_predict_dummy_mean),mae=median_absolute_error(y_test, y_predict_dummy_median),r2mean=r2_score(y_test, y_predict_dummy_mean))
	return dummy_metrics

def XGBoost_regressorHyper(X_train,y_train, X_test, y_test, msg=None):
	""" https://blog.cambridgespark.com/hyperparameter-tuning-in-xgboost-4ff9100a3b2f
	"""
	# Hyperparameter tuning in XGBoost
	import xgboost as xgb
	from sklearn.metrics import mean_absolute_error,r2_score

	dtrain = xgb.DMatrix(X_train,label=y_train)
	dtest = xgb.DMatrix(X_test, label=y_test)
	
	# MAE same scale as target, so easy to interpret
	# build baseline dumb model that learns the eman
	mean_train = np.mean(y_train)
	# Get predictions on test set
	baseline_preds = np.ones(y_test.shape) * mean_train
	# Compute MAE
	mae_baseline = mean_absolute_error(y_test, baseline_preds)
	print("Baseline MAE is {:.2f}".format(mae_baseline))
	
	# Search Hyperparameters
	#https://coderzcolumn.com/tutorials/machine-learning/xgboost-an-in-depth-guide-python
	# lower eta model more robust to overfitting but requires more rounds
	# subsample .5 prevent overfitting but 1 best results
	params = {'eval_metric':"mae", 'max_depth':8, 'min_child_weight': 5,'eta':.01, 'subsample':0.85,'colsample_bytree': .85, 'tree_method':'hist','objective':'reg:squarederror'} #'reg:tweedie'
	# define a test dataset and a metric that is used to assess performance at each round
	# If performance haven’t improved for N rounds (N is defined by the variable early_stopping_round), we stop the training
	#params['eval_metric'] = "mae"
	num_boost_round = 1499
	model = xgb.train(params, dtrain,num_boost_round=num_boost_round, evals=[(dtrain, "train"), (dtest, "Test")],early_stopping_rounds=10)
	print("(Default params) MAE: {:.2f} with {} rounds".format(model.best_score,model.best_iteration+1))
	print("\nTrain MAE : ",model.eval(dtrain))
	print("Test  MAE : ", model.eval(dtest))
	print("Train R2 Score : %.2f"%r2_score(y_train, model.predict(dtrain)))
	print("\nTest  R2 Score : %.2f"%r2_score(y_test, model.predict(dtest)))
	
	# tune hyperparameters with CV
	print('\n Running XGBoost now using Cross Validation ...')
	# Not need to pass dtest, because it is automaticall selected
	nfold = 5
	cv_results = xgb.cv(params, dtrain,num_boost_round=num_boost_round, seed=1, nfold=nfold, metrics={'mae'},early_stopping_rounds=10, verbose_eval=1)
	print('(CV) Train min MAE %.4f' %cv_results['train-mae-mean'].min())
	print('(CV) Test min MAE %.4f' %cv_results['test-mae-mean'].min())
	

	# DELETE to run faster
	####
	# minimize the MAE on cross-validation
	hypertunning = False
	if hypertunning == True:
		print('\n Running XGBoost now using Cross Validation ...')
		print("\n Learning max_depth, min_child_weight....")
		min_mae = float("Inf")
		best_params = None
		gridsearch_params = [(eta,gamma,subsample,colsample_bytree,max_depth, min_child_weight)
		for eta in np.arange(.01,.1, 0.5)
		for gamma in range(0,2,1)
		for subsample in np.arange(0.75,.9, 0.1) #0.25,1, 0.25
		for colsample_bytree in np.arange(0.75,.9, 0.1) #0.25,1, 0.25
		for max_depth in range(8,10,2)
		for min_child_weight in range(6,8, 2)]
		# Find best combination of complexity params max_depth, min_child_weight
		for eta,gamma, subsample,colsample_bytree,max_depth, min_child_weight in gridsearch_params:
			print("CV with eta={}, gamma={}, subsample={}, colsample_bytree={}, max_depth={}, min_child_weight={}".format(eta,gamma,subsample,colsample_bytree,max_depth,min_child_weight))
			# update params
			params['min_child_weight'] = min_child_weight
			params['max_depth'] = max_depth
			##
			params['gamma'] = gamma
			params['subsample'] = subsample
			params['colsample_bytree'] = colsample_bytree
			params['eta'] = eta

			# Run CV
			cv_results = xgb.cv(params, dtrain,num_boost_round=num_boost_round, seed=1,nfold=nfold,metrics={'mae'}, early_stopping_rounds=10)
			# update Best MAE
			mean_mae = cv_results['test-mae-mean'].min()
			boost_rounds = cv_results['test-mae-mean'].argmin()
			print("\tMAE {} for {} rounds".format(mean_mae, boost_rounds))
			if mean_mae < min_mae:
				min_mae = mean_mae
				#best_params = (max_depth,min_child_weight)
				best_params = (gamma, subsample, colsample_bytree, max_depth,min_child_weight, eta)

		#%%time
		min_mae = mean_mae
		best_params = (gamma, subsample, colsample_bytree, max_depth, min_child_weight, eta)
		params['gamma'] = best_params[0]; params['subsample'] = best_params[1]; params['colsample_bytree'] = best_params[2]; 
		params['max_depth'] = best_params[3]; params['min_child_weight'] = best_params[4]; params['eta'] = best_params[5]; 
		print("Best params: {}, MAE: {}".format(best_params, min_mae))

		# print("\n Learning ETA....")
		# # learning rate eta
		# for eta in [.05, .01]:
		# 	print("CV with eta={}".format(eta))
		# 	params['eta'] = eta
		# 	cv_results = xgb.cv(params, dtrain,num_boost_round=num_boost_round, seed=1,nfold=nfold,metrics={'mae'}, early_stopping_rounds=10)
		# 	mean_mae = cv_results['test-mae-mean'].min()
		# 	boost_rounds = cv_results['test-mae-mean'].argmin()
		# 	print("\tMAE {} for {} rounds\n".format(mean_mae, boost_rounds))
		# 	if mean_mae < min_mae:
		# 		min_mae = mean_mae
		# 		best_params = eta
		# print("Best params: {}, MAE: {}".format(best_params, min_mae))
		# params['eta='] = best_params #0.1
	

	# Finally Retrain with best params
	model = xgb.train(params, dtrain,num_boost_round=num_boost_round,evals=[(dtrain, "train"), (dtest, "Test")],early_stopping_rounds=10)
	print("Best MAE: {:.2f} in {} rounds".format(model.best_score, model.best_iteration+1))
	# Best MAE: 2.21 in 1999 rounds Best Model MAE 2.049
	num_boost_round = model.best_iteration + 1
	
	best_model = xgb.train(params, dtrain,num_boost_round=num_boost_round,evals=[(dtrain, "train"), (dtest, "Test")])
	print('HYPER Params tunning Best Model MAE %.3f' %mean_absolute_error(best_model.predict(dtest), y_test))
	model_n = 'best_model_26072021_histtwed_' + msg +  '.model'
	best_model.save_model("best_model_27072021_histtwed_ALL.model")
	return best_model, params
	#return best_model
	
	# To load the model
	#loaded_model = xgb.Booster()
	#loaded_model.load_model("best_model_09072021.model")
	#loaded_model.predict(dtest)


def XGBoost_regressor(X_train,y_train, X_test, y_test, msg=None):
	"""
	"""
	import xgboost as xgb
	from sklearn.metrics import mean_squared_error
	from sklearn.model_selection import KFold
	from sklearn.model_selection import cross_val_score

	model = xgb.XGBRegressor()
	model.fit(X_train,y_train)
	# Training scores
	score = model.score(X_train,y_train)  
	print("Training score: ",msg,  score)
	print("Testing score: ",msg,  model.score(X_test,y_test))

	# cross validation to evaluate trainign score
	kfold = KFold(n_splits=10, shuffle=True,random_state=1)
	scores = cross_val_score(model, X_train, y_train, cv=kfold)
	print("K-fold CV average score: %.2f" % scores.mean())

	# Make predictions for test data
	y_pred = model.predict(X_test)
	# Study accuracy of model predictions
	mse = mean_squared_error(y_pred, y_test)
	print("MSE: %.2f" % (mse))
	
	# Plot predictions
	fig, axes = plt.subplots(1, 1, figsize=(16, 8))
	x_ax = range(len(y_test))
	plt.plot(x_ax, y_test, label="original")
	plt.plot(x_ax, y_pred, label="predicted")
	plt.title("Chrono Age "+ msg)
	plt.legend()
	plt.tight_layout()
	fig_name = os.path.join(figures_dir, 'XGBooost_preds_'+ msg + '.png')
	plt.savefig(fig_name)
	
	# plot trees
	fig, ax = plt.subplots(figsize=(30, 30))
	xgb.plot_tree(model,ax=ax)
	#plt.rcParams['figure.figsize'] = [50, 10]
	fig_name = os.path.join(figures_dir, 'XGBooost_1tree_'+ msg + '.png')
	plt.savefig(fig_name)
	fig, axes = plt.subplots(1, 1)
	xgb.plot_tree(model, num_trees=0, rankdir='LR')
	fig_name = os.path.join(figures_dir, 'XGBooost_2tree_'+ msg + '.png')
	plt.savefig(fig_name)

	return model
	
def evaluation_metrics(model,X_train,y_train, X_test, y_test, msg=None):
	"""
	"""	#https://coderzcolumn.com/tutorials/machine-learning/scikit-plot-visualizing-machine-learning-algorithm-results-and-performance

	#from sklearn.metrics import confusion_matrix
	from sklearn.metrics import explained_variance_score, max_error,mean_absolute_error,mean_squared_error, mean_absolute_percentage_error,median_absolute_error,r2_score
	#print('Evaluation metric for TEST set %s' %msg)
	y_pred = model.predict(X_test)
	
	evs = explained_variance_score(y_test, y_pred)
	# The worst case
	mxe = max_error(y_test, y_pred)
	mae = mean_absolute_error(y_test, y_pred)
	mse = mean_squared_error(y_test, y_pred)
	# mean_absolute_error, may ignore the small magnitude values and only reflected the error in prediction of highest magnitude value. 
	# MAPE solves this because it calculates relative percentage error with respect to actual output.
	mape = mean_absolute_percentage_error(y_test, y_pred)
	
	# robust to outliers.
	medae = median_absolute_error(y_test, y_pred)
	# proportion of variance (of y) that has been explained by the independent variables in the model.
	# indication of goodness of fit and therefore a measure of how well unseen samples are likely to be predicted by the model, through the proportion of explained variance.
	# Best R2 1.0 and it can be negative. A constant model that always predicts the expected value of y, disregarding the input features, would get a R² score of 0.0.
	r2 = r2_score(y_test, y_pred)
	print('Linear Model Metrics Results. Explained Variance= %.3f, Max Error = %.3f, MeanAbsErr=%.3f, MeanSqErr= %.3f' %(evs, mxe, mae, mse))
	print('\t MeanAbsPercErr= %.3f, median_absolute_error = %.3f, r2_score=%.3f' %(mape, medae, r2))
	evalmetrics = dict(evs=evs, mae=mae, mxe=mxe, mse=mse, mape=mape,medae=medae,r2=r2)
	return evalmetrics


def PLS_analysis(X_train,y_train, X_test, y_test, msg=None):
	"""
	"""
	print('Running PLS analysis.....\n')
	from sklearn.decomposition import PCA
	from sklearn.cross_decomposition import PLSRegression
	from sklearn.model_selection import cross_val_predict
	from sklearn.metrics import mean_squared_error, r2_score

	pls = PLSRegression(n_components=32)
	
	pls.fit(X_train, y_train)

	score_pls = round(pls.score(X_test, y_test),4)
	print(f"PLS r-squared {score_pls:.3f}", msg)

	# Plot  Partial Least Squares Regression (PLS)  regression, just one component
	fig, axes = plt.subplots(1, 1, figsize=(10, 3))
	axes.scatter(pls.transform(X_test)[:,0], y_test, alpha=.3, label='ground truth')
	axes.scatter(pls.transform(X_test)[:,0], pls.predict(X_test), alpha=.3,label='predictions')
	axes.set(xlabel='Projected data onto first PLS component',ylabel='y', title='PLS')
	plt.text(3, 90, 'PLS r-squared=' + str(score_pls), color="red", fontsize=11)
	axes.legend()
	plt.tight_layout()
	fig_name = os.path.join(figures_dir, 'PLS_scatter_preds_'+ msg + '.png')
	plt.savefig(fig_name)
	
	# Cross validation

	return pls



def PCR_analysis(X_train,y_train, X_test, y_test, msg=None):
	""" PCA train and test and do cv
	"""


def PCA_analysis(X_train,y_train, X_test, y_test, msg=None):
	""" PCA train and test and do cv
	"""
	from sklearn.decomposition import PCA
	from sklearn.linear_model import LinearRegression
	from sklearn.model_selection import RepeatedKFold
	from sklearn.metrics import mean_squared_error
	from sklearn.preprocessing import scale 
	from sklearn import model_selection

	regr = LinearRegression()
	pca= PCA()
	# scaled train set
	X_scaled_train = pca.fit_transform(scale(X_train))
	n = len(X_scaled_train)
	# 10-fold CV, with shuffle
	kf_10 = model_selection.KFold( n_splits=10, shuffle=True, random_state=1)

	mse = []
	# Calculate MSE with only the intercept (no principal components in regression)
	score = -1*model_selection.cross_val_score(regr, np.ones((n,1)), y_train.ravel(), cv=kf_10, scoring='neg_mean_squared_error').mean()    
	mse.append(score)
	# Calculate MSE using CV for the 19 principle components, adding one component at the time.
	for i in np.arange(1, 30):
		score = -1*model_selection.cross_val_score(regr, X_scaled_train[:,:i], y_train.ravel(), cv=kf_10, scoring='neg_mean_squared_error').mean()
		mse.append(score)

	# plot MSE per nb of components
	fig = plt.figure()
	ax = fig.add_subplot(111)
	plt.plot(np.array(mse), '-v')
	plt.xlabel('Number of principal components in regression')
	plt.ylabel('MSE')
	plt.title('Chronological Age PCA')
	plt.xlim(xmin=-1);
	fig_name = os.path.join(figures_dir, 'PCA_'+ msg + '.png')
	plt.savefig(fig_name)

	#
	fig = plt.figure()
	ax = fig.add_subplot(111)
	plt.plot(np.cumsum(pca.explained_variance_ratio_))
	plt.xlim(0,60,1)
	plt.xlabel('Number of components')
	plt.ylabel('Cumulative explained variance')
	fig_name = os.path.join(figures_dir, 'PCA_cumulative_'+ msg + '.png')
	plt.savefig(fig_name,bbox_inches='tight',dpi=400)

	k = 13 # with 13 components we get minimum error already
	# check new performance with only k components
	X_reduced_test = pca.transform(scale(X_test))[:,:k]

	# Train regression model on training data 
	regr = LinearRegression()
	regr.fit(X_scaled_train[:,:k], y_train)

	# Prediction with test data
	pred = regr.predict(X_reduced_test)
	mse_test = mean_squared_error(y_test, pred)
	return regr



def build_lm_tt_cv(X_train,y_train, X_test, y_test, msg=None):
	""" build linear regression model, train and test and do cv
	"""
	import sklearn
	from sklearn.linear_model import LassoCV
	from sklearn.model_selection import RepeatedKFold
	
	print('build_lm train and test sets and x validation %s' %(msg))
	model = sklearn.linear_model.LinearRegression(normalize=False)
	model = model.fit(X_train, y_train)
	predictions = model.predict(X_test)
	score = round(model.score(X_test, y_test),4)
	print("Score:%.4f" %(score))
	# plot predictions
	fig = plt.figure()
	ax = fig.add_subplot(111)
	plt.scatter(y_test, predictions)
	#plt.xlabel("True Values")
	#plt.ylabel("Predictions")
	ax.set_title('True age', fontsize=12)
	ax.set_ylabel(r'Predicted age', fontsize=12) #ax.set_xlabel(' ')
	plt.text(86, 82, 'Score=' + str(score), color="red", fontsize=11)
	fig_name = os.path.join(figures_dir, 'scatter_y1_predicted'+ msg + '.png')
	plt.savefig(fig_name)
	

	cv = RepeatedKFold(n_splits=100, n_repeats=3, random_state=1)
	# define model
	model_lasso = LassoCV(alphas=np.arange(0, 1, 0.01), cv=cv, n_jobs=-1)
	# fit model
	model_lasso = model_lasso.fit(X_train, y_train)
	# summarize chosen configuration
	print('alpha: %f' % model_lasso.alpha_)
	score = round(model_lasso.score(X_test, y_test),4)
	print("Lasso Score:%.4f" %(score))

	pdb.set_trace()



def build_lm(X,y):
	""" build linear regression molde
	"""

	import sklearn
	
	model = sklearn.linear_model.LinearRegression(normalize=False)
	model.fit(X,y)
	print("Model coefficients:\n")
	for i in range(X.shape[1]):
		print(X.columns[i], "=", model.coef_[i].round(4))
		#the magnitude of a coefficient is not necessarily a good measure of a feature’s importance in a linear model.
	sco = model.score(X,y)	
	print('lm score =%.3f' %sco)
	pdb.set_trace()	
	compute_SHAP(model, X)

def life_expectancy():
	"""
	"""
	# load WHO longevity data
	# http://apps.who.int/gho/data/node.main.688
	who_list = pd.read_csv('http://apps.who.int/gho/athena/data/GHO/WHOSIS_000001,WHOSIS_000015?filter=COUNTRY:*&x-sideaxis=COUNTRY;YEAR&x-topaxis=GHO;SEX&profile=verbose&format=csv')

	who_list.to_csv('WHOSIS_000001,WHOSIS_000015.csv')

def model_residuals_analysis(df, msg=None):
	""" df2plot df containing predictions and observations
	"""
	#import statsmodels
	import scipy as sp

	#df['diff'] = df['ytest'] - df['ypred']

	# scatter plot of residuals. Expect to see random
	fig, ax = plt.subplots()
	df.plot(x = "ypred", y = "diff",kind = "scatter")
	ax.set_xlabel('Age Predictions')
	ax.set_ylabel('Residulas (Obs-Pred)')
	ax.set_title("Residual plot XGBoost")
	fig_name = os.path.join(figures_dir, 'scatter_residuals'+ msg + '.png')
	plt.savefig(fig_name,bbox_inches='tight',dpi=400)
	# Normality
	fig, ax = plt.subplots(figsize=(6,2.5))
	_, (__, ___, rr) = sp.stats.probplot(df['diff'], plot=ax, fit=True)
	fig_name = os.path.join(figures_dir, 'Normality_residuals'+ msg + '.png')
	plt.savefig(fig_name,bbox_inches='tight',dpi=400)
	print('Fit for normality:')
	print(rr**2)



def isoforest_outlier_detection(df_free_lon):
	"""
	"""
	# Delete contamination on entire DF	
	free_totalr = df_free_lon.shape[0]
	contamination = [0.01, 0.05, 0.1, 0.25, 'auto']
	ix_contlabel = -1
	cont_label = str(contamination[ix_contlabel])
	free_datasets, indexes, free_counts = [], [], []
	for cont in contamination:
		print('outlier_detection_isoforest contamination =%s \n' %cont)
		df_free_lon_post = outlier_detection_isoforest_cortical(df_free_lon, cont)
		print('Sick PRE Iso %d POST Iso = %d' %(sum(df_free_lon['dx_visita']>0),sum(df_free_lon_post['dx_visita']>0)))
		print('Removed healthys = %d Sicks ==%d' %(sum(df_free_lon['dx_visita']==0) - sum(df_free_lon_post['dx_visita']==0), sum(df_free_lon['dx_visita']>0) - sum(df_free_lon_post['dx_visita']>0)))

		# percentange of remaining cases
		free_removed_pc =1 - (df_free_lon_post.shape[0] - free_totalr)/100
		free_datasets.append(df_free_lon_post)
		indexes.append(cont), free_counts.append(free_removed_pc)

	df_plot = plot_removed_foriso_cortical(indexes, free_counts)
	print('Bar plot removed_foriso at PCremovedIsoforest_ISO\n')
	# Longitudinal with contamination filter
	# Select auto contamination -1 for auto
	df_free_lon = free_datasets[ix_contlabel]
	## EDA  box plot and correlation matrix
	print('Box and Corr plots for ALL FS ...\n')
	# All regions and conditions
	label = 'free' + cont_label + '_ALL'
	return df_free_lon

def lm_impfeatures_analysis(impfeats, X, y):
	"""
	"""
	# ANOVA of important features
	# regression OLS model important features
	import statsmodels.api as sm
	from statsmodels.sandbox.regression.predstd import wls_prediction_std
	X_imp = X[impfeats]

	model = sm.OLS(y, X_imp)
	results = model.fit()
	print(results.summary())
	print('Parameters: ', results.params)
	print('R2: ', results.rsquared)
	# ANOVA test (1-way) or one factor at a time
	fig, ax = plt.subplots()
	X['age']=y
	ax = sns.boxplot(x='sex', y='age', data=X, color='#99c2a2')
	fig_name = os.path.join(figures_dir, 'sex_age_OLS' + '.png')
	plt.savefig(fig_name,bbox_inches='tight',dpi=400)

def plot_heatmap_corr(df):
	"""
	"""
	fig, ax = plt.subplots()
	heatmap = sns.heatmap(df.corr(method='spearman'),vmin=-1, vmax=1,annot=True)
	heatmap.set_title('Correlation imp. feat.', fontdict={'fontsize':12}, pad=12);
	labels = ['Brain2ICV', 'rHipp', 'rOcTmpMeddLing', 'rTempTrav', 'lHipp', 'Age']
	ax.set_xticklabels((labels),rotation=45)
	ax.set_yticklabels((labels))
	fig_name = os.path.join(figures_dir, 'corr_spearman_heatmap.png')
	plt.savefig(fig_name, dpi=400, bbox_inches='tight')
	#Scatter plots 
	plt.figure()
	ax = sns.scatterplot(data=df, x="fr_BrainSegVol_to_eTIV", y="age")
	ax.set_xlabel('Age Brain2ICV'); ax.set_ylabel('Age')
	fig_name = os.path.join(figures_dir, 'scatter_b2ICV_age.png')
	plt.savefig(fig_name,bbox_inches='tight',dpi=400)

	plt.figure()
	ax = sns.scatterplot(data=df, x="free_R_Hipp", y="age")
	ax.set_xlabel('rHipp'); ax.set_ylabel('Age')
	fig_name = os.path.join(figures_dir, 'scatter_rHipp_age.png')
	plt.savefig(fig_name,bbox_inches='tight',dpi=400)

def remove_manual_outliers(df, stds=None):
	""" stds: number of standard deviations to remove points above and beyond that
	"""
	#exclude 'education', 'handlat', 'apoe', 'nvisita', 'y_last', 'dx_last','dx_visita', 'age_lastdx', 'age', 'sex'
	cols = df.columns[10:]
	#cols =['fr_BrainSegVol_to_eTIV']
	#pdb.set_trace()
	df_manual = df.copy(deep=True)
	for col in cols:
		cut_off = df_manual[col].std()*stds
		print('Cutoff Outliers for %s =%.3f' %(col,cut_off))
		lower, upper = df[col].mean() - cut_off, df[col].mean() + cut_off
		#df_manual = df[(df[col] > lower) and (df[col] < upper)]
		df_manual = df_manual[(df_manual[col]<upper) & (df_manual[col]>lower)]
	return df_manual

##################################################################################
##################################################################################
##################################################################################
##################################################################################
##################################################################################
def main():
	#https://medium.com/owkin/a-machine-learning-survival-kit-for-doctors-97982d69a375
	#https://colab.research.google.com/drive/1SWUcKT6bJLaTlxVLMvjHgLOD7Rdy3MQY#scrollTo=emyzNaHln9qa&forceEdit=true&offline=true&sandboxMode=true

	print('Code for Brain Age vs Chronological age \n')

	plt.close('all')
	# Open csv dataset
	dataframe = pd.read_csv(csv_path, sep=';') 
	dataframe_orig = dataframe.copy()
	
	# FLS Free col names of subcortical structures
	#fsl_cols = ['R_Thal_visita1','L_Puta_visita1','L_Amyg_visita1','R_Pall_visita1','L_Caud_visita1',\
	#'L_Hipp_visita1','R_Hipp_visita1','L_Accu_visita1','R_Puta_visita1','BrStem_visita1',\
	#'R_Caud_visita1','R_Amyg_visita1','L_Thal_visita1','L_Pall_visita1','R_Accu_visita1']
	free_cols = ['fr_Right_Thalamus_Proper_y1','fr_Left_Putamen_y1','fr_Left_Amygdala_y1',\
	'fr_Right_Pallidum_y1','fr_Left_Caudate_y1','fr_Left_Hippocampus_y1','fr_Right_Hippocampus_y1',\
	'fr_Left_Accumbens_area_y1','fr_Right_Putamen_y1','fr_Right_Caudate_y1','fr_Right_Amygdala_y1',\
	'fr_Left_Thalamus_Proper_y1','fr_Left_Pallidum_y1','fr_Right_Accumbens_area_y1']

	## Create string that i copy and paste in convertdf_into_longitudinal_cortical
	#getstring_corticals_aux(dataframe)
	####

	# Free of cortical parcellations
	surf_R1, surf_L1, thick_R1, thick_L1, sulcus_surf1, giry_surf1, sulcus_suth1, giry_th1 = cortical_parcels(dataframe,1)
	surf_R2, surf_L2, thick_R2, thick_L2, sulcus_surf2, giry_surf2, sulcus_suth2, giry_th2= cortical_parcels(dataframe,2)
	surf_R3, surf_L3, thick_R3, thick_L3, sulcus_surf3, giry_surf3, sulcus_suth3, giry_th3 = cortical_parcels(dataframe,3)
	surf_R4, surf_L4, thick_R4, thick_L4, sulcus_surf4, giry_surf4, sulcus_suth4, giry_th4 = cortical_parcels(dataframe,4)
	surf_R5, surf_L5, thick_R5, thick_L5, sulcus_surf5, giry_surf5, sulcus_suth5, giry_th5 = cortical_parcels(dataframe,5)
	surf_R6, surf_L6, thick_R6, thick_L6, sulcus_surf6, giry_surf6, sulcus_suth6, giry_th6 = cortical_parcels(dataframe,6)
	
	surf = surf_R1 + surf_L1 + surf_R2 + surf_L2 + surf_R3 + surf_L3 + surf_R4 + surf_L4 + surf_R5 + surf_L5 + surf_R6 + surf_L6; 
	thick = thick_R1 + thick_L1 + thick_R2 + thick_L2 + thick_R3 + thick_L3 + thick_R4 + thick_L4+ thick_R5 + thick_L5+ thick_R6 + thick_L6;

	# Rename for longitudinal df to be created
	#fsl_lon_cols = ['fsl_R_Thal', 'fsl_L_Thal', 'fsl_R_Puta', 'fsl_L_Puta','fsl_R_Amyg', 'fsl_L_Amyg', 'fsl_R_Pall', 'fsl_L_Pall', 'fsl_R_Caud','fsl_L_Caud', 'fsl_R_Hipp', 'fsl_L_Hipp', 'fsl_R_Accu', 'fsl_L_Accu']
	free_lon_cols = ['free_R_Thal', 'free_L_Thal', 'free_R_Puta','free_L_Puta', 'free_R_Amyg', 'free_L_Amyg', 'free_R_Pall','free_L_Pall', 'free_R_Caud', 'free_L_Caud', 'free_R_Hipp','free_L_Hipp', 'free_R_Accu', 'free_L_Accu']
	free_lon_cols = free_lon_cols + surf
	# Rename for RH and LH
	#fsl_R_cols = [s for s in fsl_lon_cols if "_R_" in s]
	#fsl_L_cols = [s for s in fsl_lon_cols if "_L_" in s]
	free_R_cols = [s for s in free_lon_cols if "_R_" in s]
	free_L_cols = [s for s in free_lon_cols if "_L_" in s]
	# Get longitudinal dataframe
	print('Building Longitudinal df MxN from NXM including Cortical')
	
	parceltype = 'thick' #'surf' None is 'both'
	parcelsyears_thick = [thick_R1 + thick_L1, thick_R2 + thick_L2, thick_R3 + thick_L3, thick_R4 + thick_L4, thick_R5 + thick_L5, thick_R6 + thick_L6]
	parcelsyears_surf = [surf_R1 + surf_L1, surf_R2 + surf_L2, surf_R3 + surf_L3, surf_R4 + surf_L4, surf_R5 + surf_L5, surf_R6 + surf_L6]
	if parceltype == 'thick':
		parcelsyears= parcelsyears_thick
		df_free_lon = convertdf_into_longitudinal_cortical(dataframe_orig, parcelsyears, 'thick')	
	elif parceltype == 'surf':
 		parcelsyears = parcelsyears_surf
 		df_free_lon = convertdf_into_longitudinal_cortical(dataframe_orig, parcelsyears, 'surf')
	elif parceltype == None:
		parcelsyears_thick.append(parcelsyears_surf);parcelsyears = parcelsyears_thick
		df_free_lon = convertdf_into_longitudinal_cortical(dataframe_orig, parcelsyears)
	# check if there are repeated elements in the list
	lis = [k for k,v in Counter(df_free_lon.columns).items() if v>1]
	print('REPEATED parcels must be 0 in List is : %d' %len(lis))
		
	# Remove rows with some NaNs missing volumetry for the due year
	df_free_lon.dropna(axis=0, how='any', inplace=True)

	#df_fsl_lon = df_fsl_lon.reset_index(drop=True)
	df_free_lon = df_free_lon.reset_index(drop=True)
	#df_fsl_lon_all = df_fsl_lon.copy()
	#df_free_lon_all = df_free_lon.copy()

	#Manual Outlier number of standard deviations (eg 4)
	nbrows_all = df_free_lon.shape[0]
	print('Before Manual Outlier Remove %d' %(df_free_lon.shape[0]))
	df_free_lon= remove_manual_outliers(df_free_lon, 5)
	print('After Manual Outlier Remove %d - %d = %d' %(nbrows_all,df_free_lon.shape[0], nbrows_all - df_free_lon.shape[0]))
	
	# Select uncontaminated df based on condition : df_iso OR df_free_lon
	isoforest = True
	if isoforest == True:
		# This will not work because NaNs
		df_iso = isoforest_outlier_detection(df_free_lon)
		print('REMOVED %d - %d rows ' %(df_free_lon.shape[0],df_iso.shape[0]))
		df_free_lon = df_iso.copy()

	# Get labels by cortical parcel
	frontal, insula, occipital, parietal, precuneus, temporal, cingulate = cortical_parcels_lobes(df_free_lon)

	# Select based on DX
	# Only Healthy
	df_free_H_lon = df_free_lon.loc[df_free_lon['dx_visita']==0]
	df_free_H_lon = df_free_H_lon.reset_index(drop=True)

	# Non AD (H or MCI)
	df_free_NAD_lon = df_free_lon.loc[df_free_lon['dx_visita'].isin([0,1])]
	df_free_NAD_lon = df_free_NAD_lon.reset_index(drop=True)

	# Only MCIs
	df_free_MCI_lon = df_free_lon.loc[df_free_lon['dx_visita']==1]
	df_free_MCI_lon = df_free_MCI_lon.reset_index(drop=True)

	# Only sick ones
	df_free_AD_lon = df_free_lon.loc[df_free_lon['dx_visita']==2]
	df_free_AD_lon = df_free_AD_lon.reset_index(drop=True)

	# Select type of columns: confounders subc and cort
	allcols = df_free_H_lon.columns
	regx = re.compile(rf"free_\S")
	subcort = list(filter(regx.match, allcols))
	regx = re.compile(rf"fr_\S")
	cort = list(filter(regx.match, allcols))
	# HeatMap correlatX[on
	feat_cols2plot = ['fr_BrainSegVol_to_eTIV', 'free_R_Hipp', 'fr_R_thick_S_oc_temp_medandLingual','fr_R_thick_S_temporal_transverse','free_L_Hipp']
	feat_cols2plot.append('age')
	plot_heatmap_corr(df_free_H_lon[feat_cols2plot])

	### Build MODEL
	# Build lm to predict chrono age
	# Cols = ('handlat', 'apoe', 'y_last', 'dx_last', 'dx_visita', 'age_lastdx','age', 'sex') + subcortica + cortical
	#pdb.set_trace()
	conf = ['education', 'apoe', 'sex']
	colsX = conf + subcort + cort
	# cort can be precuneus, frontal etc.

	#colsX= ['free_R_Thal', 'free_L_Thal', 'free_R_Puta','free_L_Puta', 'free_R_Amyg', 'free_L_Amyg', 'free_R_Pall','free_L_Pall', 'free_R_Caud', 'free_L_Caud', 'free_R_Hipp','free_L_Hipp', 'free_R_Accu', 'free_L_Accu']
	# age at the actual exam, 'age_lastdx' last visit
	# X, y features and ages all years YS: Do function select thick || surface from free
	X = df_free_H_lon[colsX]; y= df_free_H_lon['age'];

	# X_yi yi by year of visit
	X_y1 = df_free_H_lon[colsX].loc[df_free_H_lon['nvisita']==1];X_y2 = df_free_H_lon[colsX].loc[df_free_H_lon['nvisita']==2];X_y3 = df_free_H_lon[colsX].loc[df_free_H_lon['nvisita']==3]
	X_y4 = df_free_H_lon[colsX].loc[df_free_H_lon['nvisita']==4];X_y5 = df_free_H_lon[colsX].loc[df_free_H_lon['nvisita']==5];X_y6 = df_free_H_lon[colsX].loc[df_free_H_lon['nvisita']==6]
	y_y1= df_free_H_lon['age'].loc[df_free_H_lon['nvisita']==1];y_y2= df_free_H_lon['age'].loc[df_free_H_lon['nvisita']==2];y_y3= df_free_H_lon['age'].loc[df_free_H_lon['nvisita']==3];
	y_y4= df_free_H_lon['age'].loc[df_free_H_lon['nvisita']==4];y_y5= df_free_H_lon['age'].loc[df_free_H_lon['nvisita']==5];y_y6= df_free_H_lon['age'].loc[df_free_H_lon['nvisita']==6];

	# Split total dataset X,y in train , test and validation
	split_size = [.33, .25, .20]
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_size[1], random_state=42)
	msg = 'lm_y_ALLyears' + parceltype 
	regr = PLS_analysis(X_train,y_train, X_test, y_test, msg)
	lm_eval_metrics = evaluation_metrics(regr, X_train,y_train, X_test, y_test, msg)

	# Load model 
	#import xgboost as xgb
	#from sklearn.metrics import mean_absolute_error
	#loaded_model = xgb.Booster()
	#loaded_model.load_model("best_model_09072021.model")
	#loaded_model.predict(dtest)
	msg = 'xgb_y_ALLyears' + parceltype 
	xgHyper, params = XGBoost_regressorHyper(X_train,y_train, X_test, y_test, msg)
	eval_metrics = evaluate_xgboost(xgHyper, params, X_test, y_test)
	dummy_metrics = DummyRegressor_metrics(xgHyper, X_train,y_train, X_test, y_test)
	# Comapre metrics : linear(), xgb (eval_metrics) and dummy (dummy_metrics) 
	# Feature importance
	feature_importance = XGBoost_SHAP(xgHyper, X_train,y_train, X_test, y_test, msg)
	df2plot = feature_importance_Analysis(X_test, y_test, xgHyper, feature_importance)	
	
	# Residual analysis
	model_residuals_analysis(df2plot, 'xgb_hyper')
	# PCA Analisis
	PCA_analysis(X_y1,y_y1, X_y6, y_y6, 'y1y2')
	PCA_analysis(X_train,y_train, X_test, y_test, 'xgb_hyper')
	# anovas of the most important features
	print('Feature Importance:', feature_importance['col_name'])
	f1 = 'fr_BrainSegVol_to_eTIV'
	f2 = 'free_R_Hipp'	
	impfeats = [f1,f2]

	lm_impfeatures_analysis(impfeats, X_test, y_test)
	# pred in tyears y12...y16
	pdb.set_trace()
	







	#####################################################################################
	# PCA analysis
	print(' PLS ....\n\n')

	regry12 = PLS_analysis(X_y1,y_y1, X_y2, y_y2, 'y1-y2')
	evaluation_metrics(regry12,X_y1,y_y1, X_y2, y_y2, 'y1-y2')
	# XGB
	print(' XGB ....\n\n')
	xg = XGBoost_regressor(X_y1,y_y1, X_y2, y_y2, 'y1-y2')
	evaluation_metrics(xg,X_y1,y_y1, X_y2, y_y2, 'y1-y2')
	print(' xgHyper Y1-Y6 ....\n\n')
	xgHyper = XGBoost_regressorHyper(X_y1,y_y1, X_y6, y_y6, 'y1-y6')
	evaluation_metrics(xgHyper,X_y1,y_y1, X_y6, y_y6, 'y1-y6')
	pdb.set_trace()

	regr = PLS_analysis(X_y1,y_y1, X_y3, y_y3, 'y1-y3')
	regr = PLS_analysis(X_y1,y_y1, X_y4, y_y4, 'y1-y4')
	regr = PLS_analysis(X_y1,y_y1, X_y5, y_y5, 'y1-y5')
	regr = PLS_analysis(X_y1,y_y1, X_y6, y_y6, 'y1-y6')
	
	pdb.set_trace()
	# Boosting algorithm
	xg = XGBoost_regressor(X_y1,y_y1, X_y2, y_y2, 'y1-y2')
	evaluation_metrics(regr,X_y1,y_y1, X_y2, y_y2, 'y1-y2')
	# Model in XGBoost_importance must be XGBRegressor have attribute 'feature_importances_'
	XGBoost_importance(xg, X_y1,y_y1, X_y2, y_y2, 'y1-y2')
	# 
	XGBoost_SHAP(xg, X_y1,y_y1, X_y2, y_y2, 'y1-y2')	
	pdb.set_trace()

	# sparse Bayesian analogue to the Support Vector Machine
	sparse_rvm = RVM_analysis(X_y1,y_y1, X_y2, y_y2, 'y1-y2')

	pdb.set_trace()
	#regr = PCR_analysis(X_y1,y_y1, X_y2, y_y2, 'y1-y2')
	# PLS select components
	# X_train, y_ytain, X_test, y_yest
	build_lm_tt_cv(X_y1,y_y1, X_y2, y_y2, 'y1-y2')
	build_lm(X,y)
	# Rank features in importance using shapley
	#https://shap.readthedocs.io/en/latest/example_notebooks/overviews/An%20introduction%20to%20explainable%20AI%20with%20Shapley%20values.html


	# dataframe = convert_stringtofloat(dataframe)
	# dataframe_ = add_tissue_ratio_cols(dataframe)
	# # remove outliers csf|wm|gm_ratio_nat low, high 0.01, 0.99
	# dataframe = remove_outliers_(dataframe)
	# dataframe_ = dataframe_[np.isfinite(dataframe_['conversionmci'])]
	# train_linear_regression(dataframe_, figures_dir)
	# train_nonlinear(dataframe_, figures_dir)
	# pdb.set_trace()
	# # Scatter Plots
	# plt.figure(figsize=(15, 8))
	# col = dataframe_.conversionmci.dropna().map({0:'b', 1:'brown'})
	# plt.subplot(131)
	# plt.scatter(dataframe_.csf_rationat_visita1,dataframe_.edad_visita1, c=col, s=3, alpha=0.8)
	# plt.title('CSF~Age')
	# plt.subplot(132)
	# plt.scatter(dataframe_.gm_rationat_visita1,dataframe_.edad_visita1, c=col, s=3, alpha=0.8)
	# plt.title('GM~Age')
	# plt.subplot(133)
	# plt.scatter(dataframe_.wm_rationat_visita1,dataframe_.edad_visita1, c=col, s=3, alpha=0.8)
	# plt.title('WM~Age')
	# plt.savefig(os.path.join(figures_dir, 'scatter_tissue-age.png'))
	# xvar = 'gm_rationat_visita1'; yvar = 'wm_rationat_visita1'
	# scatterplot_2variables_in_df_(dataframe_,xvar,yvar,figures_dir)
	# xvar = 'gm_rationat_visita1'; yvar = 'csf_rationat_visita1'
	# scatterplot_2variables_in_df_(dataframe_,xvar,yvar,figures_dir)
	# xvar = 'gm_rationat_visita1'; yvar = 'edad_visita1'
	# scatterplot_2variables_in_df_(dataframe_,xvar,yvar,figures_dir)


	# # Load the files associated to raw data and segmentation
	
	# imgpath =  '/Users/jaime/vallecas/data/test/fsl/pv_0616_y2.anat/T1_biascorr_brain.nii.gz'
	# segmentationpath = '/Users/jaime/vallecas/data/test/fsl/pv_0616_y2.anat/T1_fast_seg.nii.gz'
	# print('Calling Tissue histogram imgpath:%s' %imgpath)
	# histobins = tissue_histogram(imgpath, segmentationpath, figures_dir)
	# print('Tissue histogram voxel intensities:%s DONE!' %imgpath)
	# pdb.set_trace()


if __name__ == "__name__":
	
	main()