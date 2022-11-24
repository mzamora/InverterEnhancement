#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Effects of cloud enhancement on inverter input
@author: MZamoraZ @UChile - mzamora.github.io
Checking with Ladd's article on Solar PRO
"""

import numpy as np
import datetime as dt
import pandas as pd
import pvlib
from pvlib import pvsystem
from pvlib.location import Location
from pvlib.modelchain import ModelChain
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS
#import seaborn as sns
import scipy.io
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from os import listdir

#sns.set(style="ticks",font_scale=1.1)
temperature_model_parameters=TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']

''' read GHI from RDU TMY '''
meta,rdu=pvlib.iotools.read_psm3('../SAMresults/RDU_TMY.csv')
lat=meta['Latitude']
lon=meta['Latitude']
alt=meta['Elevation']
albedo=rdu['Surface Albedo']
meanalbedo=albedo.mean()

#load modules and inverters
cec_modules = pvlib.pvsystem.retrieve_sam('CECMod')
#pv_cols=[col for col in sandia_modules.columns if 'Yingli' in col]
cec_inverters = pvlib.pvsystem.retrieve_sam('cecinverter')
#sma_cols = [col for col in cec_inverters.columns if 'SMA' in col]
inverter = cec_inverters['SMA_America__STP30000TL_US_10__480V_'] 
module = cec_modules['Yingli_Energy__China__YL330P_35b']
eta_m=module.V_mp_ref*module.I_mp_ref/module.A_c/1000
system = pvsystem.PVSystem(surface_tilt=25, 
                           surface_azimuth=180, 
                           albedo=albedo.values,
                           #albedo=meanalbedo, 
                           module_parameters=module, 
                           temperature_model_parameters={#'b': -0.0594, 'deltaT': 3,'a': -3.47,  
                                                         'noct':module.T_NOCT ,
                                                         'eta_m_ref':eta_m},
                           inverter_parameters=inverter, 
                           modules_per_string=19, 
                           strings_per_inverter=5)

location=Location(lat, lon,altitude=alt)
#let's try model chain
mc = ModelChain(system, location, clearsky_model='ineichen', 
                transposition_model='perez', 
                solar_position_method='nrel_numpy', 
                airmass_model='kastenyoung1989',
                dc_model='cec', #'cec', #from module specifications
                ac_model='sandia', #from inverter specifications
                aoi_model='no_loss', 
                spectral_model='no_loss', #spectral model 'no loss' otherwise error
                temperature_model='noct_sam', 
                losses_model='no_loss')

#times=pd.DatetimeIndex(rdu.index.strftime('%Y-%m-%d %H:%M:%S'))
weather=pd.DataFrame({'ghi':rdu['GHI'].values, 'dni':rdu['DNI'].values, 'dhi':rdu['DHI'].values,
                      'temp_air':rdu['Temperature'].values,'wind_speed':rdu['Wind Speed'].values}
                     ,index=rdu.index-dt.timedelta(hours=8))

mc.run_model(weather)
plt.plot(weather['ghi'].values[0:4*24])
plt.plot(mc.results.effective_irradiance.values[0:4*24])
plt.plot(mc.results.solar_position['elevation'].values[0:4*24])


# try manually
# calc params cec
solpos=location.get_solarposition(rdu.index-dt.timedelta(hours=2), 
                                  pressure=rdu['Pressure'],
                                  temperature=rdu['Temperature'])
relairmass=pvlib.atmosphere.get_relative_airmass(solpos['zenith'])
dniextra=pvlib.irradiance.get_extra_radiation(rdu.index-dt.timedelta(hours=2))
total_irradiance=pvlib.irradiance.get_total_irradiance(surface_tilt=25, 
                                                       surface_azimuth=180, 
                                                       solar_zenith=solpos['zenith'], 
                                                       solar_azimuth=solpos['azimuth'], 
                                                       dni=rdu['DNI'].values, 
                                                       ghi=rdu['GHI'].values, 
                                                       dhi=rdu['DHI'].values,
                                                       dni_extra=dniextra.values,
                                                       airmass=relairmass.values,
                                                       albedo=albedo.values,
                                                       model='perez')
IL, I0, Rs, Rsh, nNsVth=pvlib.pvsystem.calcparams_cec(mc.results.effective_irradiance, 
                                        mc.results.cell_temperature, 
                                        module.alpha_sc, module.a_ref, 
                                        module.I_L_ref, module.I_o_ref, 
                                        module.R_sh_ref, module.R_s, module.Adjust, 
                                        EgRef=1.121, dEgdT=- 0.0002677, 
                                        irrad_ref=1000, temp_ref=25)
IL, I0, Rs, Rsh, nNsVth=pvlib.pvsystem.calcparams_desoto(mc.results.effective_irradiance,
                                        mc.results.cell_temperature, 
                                        module.alpha_sc, module.a_ref, 
                                        module.I_L_ref, module.I_o_ref, 
                                        module.R_sh_ref, module.R_s, 
                                        EgRef=1.121, dEgdT=- 0.0002677, 
                                        irrad_ref=1000, temp_ref=25)


I_L_ref, I_o_ref, R_s, R_sh_ref, a_ref, Adjust=pvlib.ivtools.sdm.fit_cec_sam(celltype='multiSi', 
                                                      v_mp=module.V_mp_ref, 
                                                      i_mp=module.I_mp_ref, 
                                                      v_oc=module.V_oc_ref, 
                                                      i_sc=module.I_sc_ref, 
                                                      alpha_sc=module.alpha_sc, 
                                                      beta_voc=module.beta_oc, 
                                                      gamma_pmp=module.gamma_r, 
                                                      cells_in_series=72)
IL, I0, Rs, Rsh, nNsVth=pvlib.pvsystem.calcparams_cec(mc.results.effective_irradiance, 
                                        mc.results.cell_temperature, 
                                        module.alpha_sc, module.a_ref, 
                                        I_L_ref, I_o_ref, R_sh_ref, R_s, Adjust, 
                                        EgRef=1.121, dEgdT=- 0.0002677, 
                                        irrad_ref=1000, temp_ref=25)

# call singlediode
curve_info = pvsystem.singlediode(
    photocurrent=IL,
    saturation_current=I0,
    resistance_series=Rs,
    resistance_shunt=Rsh,
    nNsVth=nNsVth,
    ivcurve_pnts=100,
    method='lambertw')

# per inverter
modules_per_string=19
strings_per_inverter=5
voc=curve_info['v_oc']*modules_per_string
isc=curve_info['i_sc']*strings_per_inverter
pdc=curve_info['p_mp']*strings_per_inverter*modules_per_string

# search for current results
Vocm0=mc.results.dc.v_oc.values
Iscm0=mc.results.dc.i_sc.values
Ee0=mc.results.effective_irradiance.values
Tamb0=rdu['Temperature'].values
Tc0=mc.results.cell_temperature.values

fig,ax=plt.subplots(1,2,figsize=[8,4])
ax[0].plot(voc,Vocm0,'.',[400,700],[400,700],'--')
ax[0].set_xlabel('$V_{oc}$ (V) CEC DC model pvlib')
ax[0].set_ylabel('$V_{oc}$ (V) No DC model pvlib')
ax[1].plot(isc,Iscm0,'.',[0,55],[0,55],'--')
ax[1].set_xlabel('$I_{sc}$ (A) CEC DC model pvlib')
ax[1].set_ylabel('$I_{sc}$ (A) No DC model pvlib')
plt.tight_layout()

# load SAM results
fname='../SAMresults/results2.csv'
sam=pd.read_csv(fname)
Vocm0_sam=sam['Subarray 1 Open circuit DC voltage | (V)']
Iscm0_sam=sam['Subarray 1 String short circuit DC current | (A)']
Ee0_sam=sam['Subarray 1 POA front total irradiance nominal | (W/m2)']
Tc0_sam=sam['Subarray 1 Cell temperature | (C)']

fig,ax=plt.subplots(2,2,figsize=[8,6])
ax[0][0].plot(Vocm0,Vocm0_sam,'.',alpha=0.2)
ax[0][0].plot([540,900],[540,900],'--')
ax[0][0].set_xlim([540,900]); ax[0][0].set_ylim([540,900])
ax[0][0].set_xlabel('$V_{oc}$ (V) CEC DC model pvlib')
ax[0][0].set_ylabel('$V_{oc}$ (V) SAM')
ax[0][1].plot(Iscm0/5,Iscm0_sam,'.',alpha=0.2)
ax[0][1].plot([0,10],[0,10],'--')
ax[0][1].set_xlabel('String $I_{sc}$ (A) CEC DC model pvlib')
ax[0][1].set_ylabel('String $I_{sc}$ (A) SAM')
ax[1][0].plot(Ee0,Ee0_sam,'.',alpha=0.2)
ax[1][0].plot([0,1000],[0,1000],'--')
ax[1][0].set_xlabel('POA Irradiance (W/m$^2$) pvlib')
ax[1][0].set_ylabel('POA Irradiance (W/m$^2$) SAM')
ax[1][1].plot(Tc0,Tc0_sam,'.',alpha=0.2)
ax[1][1].plot([0,70],[0,70],'--')
ax[1][1].set_xlabel('Cell temperature (ºC) pvlib')
ax[1][1].set_ylabel('Cell temperature (ºC) SAM')
plt.tight_layout()

# temp coefficients (from SAM)
a=0.034 #0.05 # Isc
b=0.289 #0.32 # Voc
Isc=9.29

Voc0_sam=Vocm0_sam*(1+b/100*(Tamb0-Tc0_sam))
Isc0_sam=Isc*Ee0_sam/1000*(1+a/1000*(Tc0_sam-25))
Voc0=Vocm0*(1+b/100*(Tamb0-Tc0))
Isc0=Isc*Ee0/1000*(1+a/1000*(Tc0-25))

fig,ax=plt.subplots(1,2,figsize=[9,4])
ax[0].plot(Isc0_sam,Iscm0_sam,'.',alpha=0.2)
ax[0].set_xlabel('Eq. 2 w/ SAM')
ax[0].set_ylabel('$I_{sc}$ from SAM')
ax[1].plot(Isc0,Iscm0/5,'.',alpha=0.2)
ax[1].set_xlabel('Eq. 2 w/ pvlib')
ax[1].set_ylabel('$I_{sc}$ from pvlib')
for axi in ax: axi.plot([0,10],[0,10],'--')
plt.tight_layout()

print('== NEC 3h Rule ==')
print('1. pvlib')
#pvlib stats NEC
Isc0_3hmean=np.array([(Isc0[i]+Isc0[i+1]+Isc0[i+2])/3 for i in range(0,len(Isc0)-2)])
print('Max 1 h open circuit voltage: '+str(np.nanmax(Voc0))+' V')
print('Max 3 h mean short circuit current: '+str(np.nanmax(Isc0_3hmean))+' A')
print('Max 1 h mean short circuit current: '+str(np.nanmax(Isc0))+' A')
#SAM stats NEC
print('2. SAM')
Isc0_sam3hmean=np.array([(Isc0_sam[i]+Isc0_sam[i+1]+Isc0_sam[i+2])/3 for i in range(0,len(Isc0_sam)-2)])
print('Max 1 h open circuit voltage: '+str(np.nanmax(Voc0_sam))+' V')
print('Max 3 h mean short circuit current: '+str(np.nanmax(Isc0_sam3hmean))+' A')
print('Max 1 h mean short circuit current: '+str(np.nanmax(Isc0_sam))+' A')
# Isc from model 
print('== Isc from model ==')
print('1. pvlib')
#pvlib stats NEC
Iscm0_3hmean=np.array([(Iscm0[i]+Iscm0[i+1]+Iscm0[i+2])/3 for i in range(0,len(Iscm0)-2)])
print('Max 1 h open circuit voltage: '+str(np.nanmax(Vocm0))+' V')
print('Max 3 h mean short circuit current: '+str(np.nanmax(Iscm0_3hmean/5))+' A')
print('Max 1 h mean short circuit current: '+str(np.nanmax(Iscm0/5))+' A')
#SAM stats NEC
print('2. SAM')
Iscm0_sam3hmean=np.array([(Iscm0_sam[i]+Iscm0_sam[i+1]+Iscm0_sam[i+2])/3 for i in range(0,len(Iscm0_sam)-2)])
print('Max 1 h open circuit voltage: '+str(np.nanmax(Vocm0_sam))+' V')
print('Max 3 h mean short circuit current: '+str(np.nanmax(Iscm0_sam3hmean))+' A')
print('Max 1 h mean short circuit current: '+str(np.nanmax(Iscm0_sam))+' A')


''' T NOCT SENSITIVITY '''

tnocts=np.linspace(40,50,11)
isc01h_max_tnoct=np.zeros(len(tnocts))
isc03h_max_tnoct=np.zeros(len(tnocts))
iscm01h_max_tnoct=np.zeros(len(tnocts))
iscm03h_max_tnoct=np.zeros(len(tnocts))
for tnoct in tnocts:
    module.T_NOCT=tnoct
    # re do simulation
    system = pvsystem.PVSystem(surface_tilt=25, 
                           surface_azimuth=180, 
                           albedo=albedo,
                           #albedo=meanalbedo, 
                           module_parameters=module, 
                           temperature_model_parameters={#'b': -0.0594, 'deltaT': 3,'a': -3.47,  
                                                         'noct':module.T_NOCT ,
                                                         'eta_m_ref':eta_m},
                           inverter_parameters=inverter, 
                           modules_per_string=19, 
                           strings_per_inverter=5)
    mc = ModelChain(system, location, clearsky_model='ineichen', 
                transposition_model='perez', 
                solar_position_method='nrel_numpy', 
                airmass_model='kastenyoung1989',
                dc_model='cec', #'cec', #from module specifications
                ac_model='sandia', #from inverter specifications
                aoi_model='no_loss', 
                spectral_model='no_loss', #spectral model 'no loss' otherwise error
                temperature_model='noct_sam', 
                losses_model='no_loss')
    weather=pd.DataFrame({'ghi':rdu['GHI'].values, 'dni':rdu['DNI'].values, 'dhi':rdu['DHI'].values,
                      'temp_air':rdu['Temperature'].values,'wind_speed':rdu['Wind Speed'].values}
                     ,index=rdu.index-dt.timedelta(hours=8))
    mc.run_model(weather)
    Vocm0=mc.results.dc.v_oc.values
    Iscm0=mc.results.dc.i_sc.values
    Ee0=mc.results.effective_irradiance.values
    Tamb0=rdu['Temperature'].values
    Tc0=mc.results.cell_temperature.values
    a=0.034 #0.05 # Isc
    b=0.289 #0.32 # Voc
    Isc=9.29
    Isc0=Isc*Ee0/1000*(1+a/1000*(Tc0-25))
    # save max Isc 
    idx=np.where(tnocts==tnoct)[0][0]
    Isc03h=np.array([(Isc0[i]+Isc0[i+1]+Isc0[i+2])/3 for i in range(0,len(Isc0)-2)])
    Iscm03h=np.array([(Iscm0[i]+Iscm0[i+1]+Iscm0[i+2])/3 for i in range(0,len(Iscm0)-2)])
    isc01h_max_tnoct[idx]=np.nanmax(Isc0)
    isc03h_max_tnoct[idx]=np.nanmax(Isc03h)
    iscm01h_max_tnoct[idx]=np.nanmax(Iscm0)
    iscm03h_max_tnoct[idx]=np.nanmax(Iscm03h)
    
fig,ax=plt.subplots(1,1)
ax.plot(tnocts,isc01h_max_tnoct,'.-')
ax.plot(tnocts,iscm01h_max_tnoct/5,'.-')
ax.plot(tnocts,isc03h_max_tnoct,'.-')
ax.plot(tnocts,iscm03h_max_tnoct/5,'.-')
ax.set_xlabel('$T_{NOCT}$ (ºC)')
ax.set_ylabel('Max. $I_{sc}$ (A)')
ax.legend(['1 h, Eq.2 ','1 h, pvlib','3 h, Eq.2 ','3 h, pvlib'],loc='lower right')
