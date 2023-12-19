#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
plot_spectre_2D_casys.py
:author: A. Jouzeau, M. Dalila
:creation date : 09-10-2017
:last modified : 08-11-2019

python 2.7 test OK python 3.X pas de test
Copyright 2017, CLS.

This program is free software: you can redistribute it and/or modify it
under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License,
or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser
General Public License for more details (http://www.gnu.org/licenses/).
"""

from math import pi
import numpy as np
import argparse
import netCDF4 as netcdf
import datetime
from matplotlib import pyplot as plt
from matplotlib.patches import Patch


def pol2cart(rho, phi):
    """
    Polar => Carthesian
    Usage:
        (x,y) = pol2cart(rho, phi)
    With:
        :param      rho     :distance to center     :NA     :NA     :E
        :param      phi     :angle                  :NA     :NA     :E
        :type       rho     :float
        :returns    (x,y)   :coordinates in carthesian
        :rtype      (x,y)   :tuple of floats
    """
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)


def process_pcomb(p_comb, kspectra, parts_comb,
                  posneg=0, box=562,
                  n_k=32,
                  n_phi=24,
                  n_partitions=3):
    """
    Preprocess data to plot 2D direction spectrum for all incidences combined
    Usage:
        specm_comb, mask_comb  = process_pcomb(p_comb, kspectra, parts_comb,
                                                posneg, box,n_k, n_phi, n_partitions)
    With:
        :param p_comb       :2D combined measure spectrum   :L2     :p_combined     :E
        :param kspectra     :wave number vector             :L2     :k_spectra      :E
        :param parts_comb   :mask of detected partitions    :L2     :mask_combined  :E
        :param posneg       :left/right                     :L2     :n_posneg       :E
        :param box          :box number                     :NA     :NA             :E
        :param n_k          :length wave number             :L2     :nk             :E
        :param n_phi        :2*len(phi original)            :NA     :NA             :E
        :param n_partitions :number of detected partitions  :NA     :NA             :E

        :type p_comb        :numpy ma array
        :type kspectra      :float
        :type parts_comb    :float
        :type posneg        :int
        :type box           :int
        :type n_k           :int
        :type n_phi         :int
        :type n_partitions  :int

        :returns    specm_comb  :Mean modulation spectrum combining all incidence beams from 0:360°
        :returns    mask_comb   :Mask for all partitions detected
        :rtype      specm       :numpy.ma
        :rtype      mask_comb   :numpy.ma
    """
    # integer part of number
    original_mask = np.int32(parts_comb[:, :, :, :, :])
    # spectre built with phi = 0° at the north
    spec = np.ma.zeros((n_phi, n_k))
    # spec == pp_mean[:, phi,:...] only if p_combined not masked else masked
    # Looking for places where data is masked -- for box = n°box => phi = 0:360:15
    spec[0:6, :] = np.ma.masked_where(np.ma.getmask(
                   p_comb[:, 5::-1, posneg, box].T) == True,
                   p_comb[:, 5::-1, posneg, box].T)
    spec[6:12, :] = np.ma.masked_where(np.ma.getmask(
                    p_comb[:, 11:5:-1, posneg, box].T) == True,
                    p_comb[:, 11:5:-1, posneg, box].T)
    spec[12:18, :] = np.ma.masked_where(np.ma.getmask(
                     p_comb[:, 5::-1, posneg, box].T) == True,
                     p_comb[:, 5::-1, posneg, box].T)
    spec[18:24, :] = np.ma.masked_where(np.ma.getmask(
                     p_comb[:, 11:5:-1, posneg, box].T) == True,
                     p_comb[:, 11:5:-1, posneg, box].T)

    # limits inf, limits lambda < 500m
    specm_comb = np.ma.masked_where(spec < -1.e8, spec)
    specm_comb_lim = np.zeros_like(specm_comb)
    # limit Wavelength
    limit_k = (2*pi)/500.0
    for i in range(specm_comb.shape[0]):
        specm_comb_lim[i] = np.ma.masked_where(kspectra[:] < limit_k, specm_comb[i])

    # mask partitions with 180° ambiguity and phi = 0° at the North
    mask_comb = np.zeros((n_partitions, n_phi+1, n_k))
    mask_comb[:, 0:6, :]   = original_mask[:, 5::-1, :, posneg, box].T
    mask_comb[:, 6:12, :]  = original_mask[:, 11:5:-1, :, posneg, box].T
    mask_comb[:, 12:18, :] = original_mask[:, 5::-1, :, posneg, box].T
    mask_comb[:, 18:, :]   = original_mask[:, 11:4:-1, :, posneg, box].T

    return specm_comb_lim, mask_comb

def plot_polarfig(mask, kspectra, array_color, n_phi, title,kmin,kmax, 
                  ax=None, fig=None, vmin=None, vmax=None, disp_partitions=False):
    """
    Plot one 2D spectrum against wave number vector
    Usage:
        plot_polarfig(mask, k_spectra, array_color,
                      n_phi, title, disp_mask)
    With:
        :param mask         :detected partitions            :NA :NA         :E
        :param kspectra     :wave number vector             :L2 :k_spectra  :E
        :param array_color  :2D spectrum                    :NA :NA         :E
        :param n_phi        :number of azimuthal  angle     :L2 :n_phi      :E
        :param title        :number of azimuths             :NA :NA         :E
        :param disp_mask    :display or not masks           :NA :NA         :E
        :type mask         :numpy.ma
        :type kspectra     :float
        :type array_color  :float
        :type n_phi        :int
        :type title        :string
        :type disp_mask    :boolean
        :returns    None
        :rtype      None    :NA
    """
    
    if ax is None:
        fig, ax = plt.subplots(facecolor='white', figsize=(11, 9))
    else:
        fig = plt.gcf()
        
    # size of wave number vector
    n_k = len(kspectra[:])
    # polar coordinates
    r = np.tile(kspectra[:], n_phi+1).reshape(n_phi+1, n_k)
    # grid , 25 = n_pĥi +1
    theta, _ = np.mgrid[0.0:2.0*pi:25j, 0:n_k:1]

    x = theta
    y = r 
    # Spectrum map
    pc = ax.pcolormesh(x, y, np.ma.filled(array_color[:,:-1], fill_value=0.0),
                   cmap='viridis', vmin=vmin, vmax=vmax)

    # radius labels locations
    labels = [30, 50, 100, 200, 400]
    ax.set_rticks([2*pi/i for i in labels])
    labels = [str(i) + ' m' for i in labels]
    ax.yaxis.set_ticklabels(labels)
    ax.set_rlabel_position(-95)
    # radius labels in white
    ax.tick_params(labelcolor='w', axis='y')
    ax.set_rlim(kmin,kmax)
    #ax.set_rmax(0.21)

    # theta ticks
    ax.set_xticks([i*pi/6 for i in range(12)])
    ax.set_xticklabels(['90','60','30','0','330','300','270','240','210','180','150','120'])
    # colorbar
    fig.colorbar(pc, ax=ax)

    # add Partitions edges
    if disp_partitions:
        ax.contour(x, y, mask[0, :, :],
                    levels=[0.999],
                    colors='#FF0E00',
                    origin='upper',
                    linewidths=2.0,
                    linestyles='dashed',
                    label='partition 1',)
        ax.contour(x, y, mask[1, :, :],
                    levels=[0.999],
                    colors='#FFFFFF',
                    origin='upper',
                    linewidths=1.0,
                    linestyles='solid',
                    label='partition 2',)
        ax.contour(x, y, mask[2, :, :],
                    levels=[0.999],
                    colors='#D4FF9C',
                    origin='upper',
                    linewidths=2.0,
                    linestyles='dashdot',
                    label='partition 3',)
        # patch for legend
        p1 = Patch(edgecolor='#FF0E00',linestyle='dashed',
                   linewidth=1.0, fill=False)
        p2 = Patch(edgecolor='#FFFFFF', linestyle='solid',
                   linewidth=1.0, fill=False)
        p3 = Patch(edgecolor='#D4FF9C',linestyle='dashdot',
                   linewidth=1.0, fill=False)
        
        legend = ax.legend([p1, p2, p3],
                   ['Partition 1', 'Partition 2', 'Partition 3'],
                   loc='best')
        frame = legend.get_frame()
        frame.set_facecolor('black')
        for text in legend.get_texts():
            plt.setp(text, color = 'w')  
    ax.set_title(title)
    ax.grid(True)

    
def plotSpectrum(box,posneg,beam,NC_File_L2_path,NC_File_L2_name,min_wavelength,max_wavelength,vmin,vmax, partitions):
    #initializations
    NC_File_L2=NC_File_L2_path+NC_File_L2_name
    cdf = netcdf.Dataset(NC_File_L2)
    time = cdf.variables['time_spec_l2'][:]
    lon = cdf.variables['lon_spec_l2'][:]
    lat = cdf.variables['lat_spec_l2'][:]
    K_SPECTRA = cdf.variables['k_spectra'][:]
    PARTI_COMB  = cdf.variables['mask_combined'][:]
    PARTI_BEAM = cdf.variables['mask'][:]
    P_COMBINED = cdf.variables['p_combined'][:]                
    PP_MEAN = cdf.variables['pp_mean'][:] 
    N_KI    = int(32)
    N_PHI  = int(12)
    N_PARTITIONS = int(3)
    k_max= round(2*pi/min_wavelength,3)
    k_min= round(2*pi/max_wavelength,3)
    

    # Get nadir box time
    t0 = datetime.datetime(2009,1,1)
    date = (t0 + datetime.timedelta(seconds=float(time[posneg,box,0]))).strftime('%Y-%m-%dT%H:%M:%S')
    
    # Compute ascendant (= 1) /descendant (= -1) flag        
    flg_asc_dsc = np.sign(lat[0, 1:] - lat[0, :-1]).astype(np.int8)
    
    # display spectrum
    if beam == 0:
        beam_title='combined'
    else:
        PARTI_COMB = PARTI_BEAM[:,:,:,:,:,int((beam-6)/2)]
        P_COMBINED = PP_MEAN[:,:,:,:,int((beam-6)/2)]
        beam_title = 'beam '+ str(beam)+'$\degree$'

    specm_comb, mask_comb = process_pcomb(P_COMBINED,
                                K_SPECTRA, PARTI_COMB, posneg=posneg, 
                                box=box, n_k=N_KI, n_phi=N_PHI*2, 
                                n_partitions=N_PARTITIONS)
    title = '2D mean slope spectrum, ' + str(beam_title) + \
        ' for box: ' + str(box) + ', posneg: '+ str(posneg)
    f = NC_File_L2_name
    suptitle = '\nFile: {0}\nCoordinates: ({1}°, {2}°)\nDate: {3}'.format(f, lon[posneg,box], lat[posneg,box], date)
    title += suptitle
    plt.figure(figsize=(11, 9))
    ax = plt.subplot(111, projection='polar')

    plot_polarfig(mask_comb,K_SPECTRA, specm_comb, N_PHI*2, title,k_min,k_max, 
                  ax=ax, fig=None, vmin=vmin, vmax=vmax,disp_partitions=partitions)


