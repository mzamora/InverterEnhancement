#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 10:01:33 2022
Effects of cloud enhancement on inverter input
@author: MZamoraZ @UChile - mzamora.github.io
Main file for bifacial runs, yearly mean albedo case
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
import warnings

# supressing shapely warnings that occur on import of pvfactors
warnings.filterwarnings(action='ignore', module='pvfactors')

''' read GHI from SURFRAD '''
dir0='../../SURFRAD/'
sites=['Bondville_IL','Boulder_CO','Desert_Rock_NV',
       'Fort_Peck_MT','Goodwin_Creek_MS','Penn_State_PA',
       'Sioux_Falls_SD']
site=sites[0]
if site=='Bondville_IL':
    lat=40.05192; lon=-88.37309; ele=230; albedo=0.247
elif site=='Boulder_CO':
    lat=40.12498; lon=-105.2368; ele=1689; albedo=0.199
elif site=='Desert_Rock_NV':
    lat=36.62373; lon=-116.01947; ele=1007; albedo=0.211
elif site=='Fort_Peck_MT':
    lat=48.30783; lon=-105.10170; ele=634; albedo=0.247
elif site=='Goodwin_Creek_MS':
    lat= 34.2547; lon=-89.8729; ele=98.0; albedo=0.2 
elif site=='Penn_State_PA':
    lat=40.72012; lon=-77.93085; ele=376; albedo=0.252
elif site=='Sioux_Falls_SD':
    lat=43.73403; lon=-96.62328; ele=473; albedo=0.238    
    # albedos: https://www.nrel.gov/docs/fy21osti/75924.pdf
year=2020
yeardir=dir0+site+'/'+str(year)+'/'
lss=listdir(yeardir)
lss.sort()
ndays=len(lss)

# pv system
cec_modules = pvlib.pvsystem.retrieve_sam('CECMod')
cec_inverters = pvlib.pvsystem.retrieve_sam('cecinverter')
inverter = cec_inverters['Chint_Power_Systems_America__CPS_SCA36KTL_DO_US__480V_'] 
module = cec_modules['Silfab_Solar_Inc_SLA_X_285_Clear']
bifaciality=0.85 
eta_m=eta_m=module.V_mp_ref*module.I_mp_ref/module.A_c/1000
module.eta_m=eta_m

# number of modules per string, following SAM's algorithm
ilr=1.2 #
ptotal=33000 #200 kW between 6 inverters
vmid=(inverter.Mppt_low+inverter.Mppt_high)/2
ns=int(vmid/module.V_mp_ref) 
while(ns*module.V_oc_ref>inverter.Vdcmax):
    ns=ns-1
npar=int(ptotal/(ns*module.STC))
ninv=np.ceil(ns*npar*module.STC/(ilr*inverter.Pdco))
print('estimated ilr='+str(ns*npar*module.STC/inverter.Pdco))
print('estimated ilr with bifacial gain='+str(ns*npar*321.4/inverter.Pdco))



# bifacial systems in https://ieeexplore.ieee.org/ielaam/5503869/8694165/8644027-aam.pdf
# sandia lab: h=0.5m, grc=0.28, albedo=0.25, single axis tracking
# eastern oregon: 
#     a) 100kW silfab 285 modules 2 up landscape, 
#     b) 200 silfab 285, 2 up landscape, to 6 Chint 36 kW
# module collector width of 2m, row spacing of 5.65 m, measured gcr=0.35, tracker hub of 1.5m gives H=0.75
 

# RUN fixed tilt cases
for site in sites:
    if site=='Bondville_IL':
        lat=40.05192; lon=-88.37309; ele=230; albedo=0.247; tiltopt=33
    elif site=='Boulder_CO':
        lat=40.12498; lon=-105.2368; ele=1689; albedo=0.199; tiltopt=37
    elif site=='Desert_Rock_NV':
        lat=36.62373; lon=-116.01947; ele=1007; albedo=0.211; tiltopt=35
    elif site=='Fort_Peck_MT':
        lat=48.30783; lon=-105.10170; ele=634; albedo=0.247; tiltopt=40
    elif site=='Goodwin_Creek_MS':
        lat= 34.2547; lon=-89.8729; ele=98.0; albedo=0.2 ; tiltopt=30
    elif site=='Penn_State_PA':
        lat=40.72012; lon=-77.93085; ele=376; albedo=0.252; tiltopt=32
    elif site=='Sioux_Falls_SD':
        lat=43.73403; lon=-96.62328; ele=473; albedo=0.238; tiltopt=38

    tilts=[10,25,tiltopt]
    for tiltang in tilts:
        #tiltang=10
        f = open('../'+site+'_tilt'+str(tiltang)+'_output_bifacial_meanalbedo.txt', 'w')
        f.write(site+', tilt='+str(tiltang)+'\n')
        
        # now just run the whole thing and get what we are after
        met=pd.DataFrame()
        poa=pd.DataFrame()
        for year in range(2011,2021):
            yeardir=dir0+site+'/'+str(year)+'/'
            lss=listdir(yeardir)
            lss.sort()
            metyear=pd.DataFrame()
            for filei in lss:
                weather=pvlib.iotools.read_surfrad(yeardir+filei)[0] # load surfrad data
                metyear=metyear.append(weather)
            
            solpos=pvlib.solarposition.ephemeris(metyear.index, lat, lon)
            # delete days with missing data
            ii=0
            while True:
                try:
                    i0=ii+np.where(solpos.elevation[ii:-1]>0)[0][0]
                except IndexError:
                    break
                try:
                    i1=i0+np.where(solpos.elevation[i0:-1]<0)[0][0]
                except IndexError:
                    break
                ghinans=np.sum(np.isnan(metyear.ghi[i0:i1]))
                dhinans=np.sum(np.isnan(metyear.dhi[i0:i1]))
                dninans=np.sum(np.isnan(metyear.dni[i0:i1]))
                if ghinans+dhinans+dninans>0: #there's at least 1 nan
                    #f.write('Deleting a day: '+metyear.index[i0].strftime('%Y%m%d')+'\n')
                    #delete the dc results for the whole day
                    metyear.loc[metyear.index[i0:i1],'ghi']=np.nan
                    metyear.loc[metyear.index[i0:i1],'dhi']=np.nan
                    metyear.loc[metyear.index[i0:i1],'dni']=np.nan
                ii=i1 #update counter
              
            # fixed tilt case (not very realistic)
            #albs=metyear.uw_solar/metyear.ghi #time series of albedo for bifacial because annual mean can be misleading
            irrad=pvlib.bifacial.pvfactors.pvfactors_timeseries(solar_azimuth=solpos.azimuth, solar_zenith=solpos.zenith, 
                                                          surface_azimuth=180, surface_tilt=tiltang, timestamps=solpos.index, 
                                                          dni=metyear.dni, dhi=metyear.dhi, gcr=0.35, pvrow_height=1.5, 
                                                          pvrow_width=2, albedo=albedo, #albs.values, 
                                                          axis_azimuth=0)
            irrad=pd.concat(irrad,axis=1)
            #eff_irr_bifi=irrad['total_abs_front']+bifaciality*irrad['total_abs_back']
            irrad['effective_irradiance'] = (irrad['total_abs_front'] + (irrad['total_abs_back'] * bifaciality))
            
            # save data per site per year
            metyear.to_csv('../met'+site+str(year)+'_tilt'+str(tiltang)+'_bifacial_meanalbedo.csv')
            pd.DataFrame(irrad.effective_irradiance).to_csv('../poa'+site+str(year)+'_tilt'+str(tiltang)+'_bifacial_meanalbedo.csv')
            
# RUN single axis tracking

for site in sites:
    if site=='Bondville_IL':
        lat=40.05192; lon=-88.37309; ele=230; albedo=0.247; tiltopt=33
    elif site=='Boulder_CO':
        lat=40.12498; lon=-105.2368; ele=1689; albedo=0.199; tiltopt=37
    elif site=='Desert_Rock_NV':
        lat=36.62373; lon=-116.01947; ele=1007; albedo=0.211; tiltopt=35
    elif site=='Fort_Peck_MT':
        lat=48.30783; lon=-105.10170; ele=634; albedo=0.247; tiltopt=40
    elif site=='Goodwin_Creek_MS':
        lat= 34.2547; lon=-89.8729; ele=98.0; albedo=0.2 ; tiltopt=30
    elif site=='Penn_State_PA':
        lat=40.72012; lon=-77.93085; ele=376; albedo=0.252; tiltopt=32
    elif site=='Sioux_Falls_SD':
        lat=43.73403; lon=-96.62328; ele=473; albedo=0.238; tiltopt=38

    f = open('../'+site+'_tracking_output_bifacial_meanalbedo.txt', 'w')
    f.write(site+', Single axis tracking \n')
        
    # now just run the whole thing and get what we are after
    
    met=pd.DataFrame()
    poa=pd.DataFrame()
    pmp=pd.DataFrame()
    dc=pd.DataFrame()
    ac=pd.DataFrame()
    tcell=pd.DataFrame()
    elevation=pd.DataFrame()
    for year in range(2011,2021):
        yeardir=dir0+site+'/'+str(year)+'/'
        lss=listdir(yeardir)
        lss.sort()
        metyear=pd.DataFrame()
        for filei in lss:
            weather=pvlib.iotools.read_surfrad(yeardir+filei)[0] # load surfrad data
            metyear=metyear.append(weather)
        
        solpos=pvlib.solarposition.ephemeris(metyear.index, lat, lon)
        # delete days with missing data
        ii=0
        while True:
            try:
                i0=ii+np.where(solpos.elevation[ii:-1]>0)[0][0]
            except IndexError:
                break
            try:
                i1=i0+np.where(solpos.elevation[i0:-1]<0)[0][0]
            except IndexError:
                break
            ghinans=np.sum(np.isnan(metyear.ghi[i0:i1]))
            dhinans=np.sum(np.isnan(metyear.dhi[i0:i1]))
            dninans=np.sum(np.isnan(metyear.dni[i0:i1]))
            if ghinans+dhinans+dninans>0: #there's at least 1 nan
                f.write('Deleting a day: '+metyear.index[i0].strftime('%Y%m%d')+'\n')
                #delete the dc results for the whole day
                metyear.loc[metyear.index[i0:i1],'ghi']=np.nan
                metyear.loc[metyear.index[i0:i1],'dhi']=np.nan
                metyear.loc[metyear.index[i0:i1],'dni']=np.nan
            ii=i1 #update counter
        
        #single axis tracker position
        tracker_data=pvlib.tracking.singleaxis(solpos['apparent_zenith'], solpos['azimuth'])
        tracker_tilt=tracker_data['surface_tilt'].fillna(0)
        tracker_azimuth=tracker_data['surface_azimuth'].fillna(0)
        # 
        #albs=metyear.uw_solar/metyear.ghi #time series of albedo for bifacial because annual mean can be misleading
        irrad=pvlib.bifacial.pvfactors.pvfactors_timeseries(solar_azimuth=solpos.azimuth, solar_zenith=solpos.zenith, 
                                                      surface_azimuth=tracker_azimuth, surface_tilt=tracker_tilt, timestamps=solpos.index, 
                                                      dni=metyear.dni, dhi=metyear.dhi, gcr=0.35, pvrow_height=1.5, 
                                                      pvrow_width=2, albedo=albedo, #albs.values, 
                                                      axis_azimuth=0)
        irrad=pd.concat(irrad,axis=1)
        irrad['effective_irradiance'] = (irrad['total_abs_front'] + (irrad['total_abs_back'] * bifaciality))
        #eff_irr_bifi=irrad['total_abs_front']+bifaciality*irrad['total_abs_back']
        
        # system = pvsystem.PVSystem(surface_tilt=tracker_tilt, 
        #                        surface_azimuth=tracker_azimuth, 
        #                        albedo=albedo,
        #                        module_parameters=module, 
        #                        temperature_model_parameters={#'b': -0.0594, 'deltaT': 3,'a': -3.47,  
        #                                                      'noct':module.T_NOCT ,
        #                                                      'module_efficiency':eta_m},
        #                        inverter_parameters=inverter, 
        #                        modules_per_string=20, 
        #                        strings_per_inverter=5)
        #location=Location(lat, lon,altitude=ele)
        #let's try model chain
        # mc = ModelChain(system, location, clearsky_model='ineichen', 
        #                         transposition_model='perez', 
        #                         solar_position_method='nrel_numpy', 
        #                         airmass_model='kastenyoung1989',
        #                         dc_model='cec', #from module specifications
        #                         ac_model='sandia', #from inverter specifications
        #                         aoi_model='no_loss', 
        #                         spectral_model='no_loss', #spectral model 'no loss' otherwise error
        #                         temperature_model='noct_sam', 
        #                         losses_model='no_loss')
        # mc.run_model_from_effective_irradiance(irrad) #bifacial irradiance
        #mc.run_model(metyear)
                # save data per site per year
        metyear.to_csv('../met'+site+str(year)+'_tracking_bifacial_meanalbedo.csv')
        pd.DataFrame(irrad.effective_irradiance).to_csv('../poa'+site+str(year)+'_tracking_bifacial_meanalbedo.csv')
        #pd.DataFrame(mc.results.dc.p_mp).to_csv('../pmp'+site+str(year)+'_tracking_bifacial_meanalbedo.csv')
        #pd.DataFrame(mc.results.dc).to_csv('../dc'+site+str(year)+'_tracking_bifacial_meanalbedo.csv')
        #pd.DataFrame(mc.results.ac).to_csv('../ac'+site+str(year)+'_tracking_bifacial_meanalbedo.csv')
        #pd.DataFrame(mc.results.cell_temperature).to_csv('../tcell'+site+str(year)+'_tracking_bifacial_meanalbedo.csv')
        #pd.DataFrame(solpos.elevation).to_csv('../elevation'+site+str(year)+'_tracking_bifacial_meanalbedo.csv')
            
            
            
            
# functions for re reading afterwards            
            
def downsampledstats_tilt(poa,met,timeres,module,inverter,lat,lon,ele,albedo,tilt_ang):
    poawbg_=poa.resample(timeres).mean()
    weather_=met.resample(timeres).mean()
    times_=poawbg_.index
    celltemp=pvlib.temperature.noct_sam(poa_global=poawbg_['effective_irradiance'], temp_air=weather_.temp_air, 
                                                wind_speed=weather_.wind_speed, noct=module.T_NOCT, module_efficiency=module.eta_m)
    poawbg_['cell_temperature']=celltemp.values
    system_ = pvsystem.PVSystem(surface_tilt=tilt_ang, surface_azimuth=180, 
                            albedo=albedo, module_parameters=module, 
                            temperature_model_parameters={'noct':module.T_NOCT ,'module_efficiency':module.eta_m},
                            inverter_parameters=inverter, modules_per_string=20, strings_per_inverter=5)
    location_=Location(lat, lon,altitude=ele)
    mc_ = ModelChain(system_, location_, clearsky_model='ineichen', transposition_model='perez', 
                            solar_position_method='nrel_numpy', airmass_model='kastenyoung1989',
                            dc_model='cec', #from module specifications
                            ac_model='sandia', #from inverter specifications
                            aoi_model='no_loss', 
                            spectral_model='no_loss', #spectral model 'no loss' otherwise error
                            temperature_model='noct_sam', 
                            losses_model='no_loss')
    mc_.run_model_from_effective_irradiance(poawbg_)
    # get current results
    Vocm0_=pd.DataFrame(mc_.results.dc.v_oc.values,index=times_)
    Iscm0_=pd.DataFrame(mc_.results.dc.i_sc.values,index=times_)
    Vmpp0_=pd.DataFrame(mc_.results.dc.v_mp.values,index=times_)
    Impp0_=pd.DataFrame(mc_.results.dc.i_mp.values,index=times_)
    return Vocm0_,Iscm0_,Vmpp0_,Impp0_

def output_tres_table(T1min,T5min,T15min,T30min,T1h):
    Tout=np.array([[np.nanmax(T1min),np.nanmax(T1min.rolling('5T').mean()),
                       np.nanmax(T1min.rolling('15T').mean()),np.nanmax(T1min.rolling('30T').mean()),
                       np.nanmax(T1min.rolling('1H').mean()),np.nanmax(T1min.rolling('3H').mean())],
                      [np.nan,np.nanmax(T5min),np.nanmax(T5min.rolling('15T').mean()),
                       np.nanmax(T5min.rolling('30T').mean()),np.nanmax(T5min.rolling('1H').mean()),
                       np.nanmax(T5min.rolling('3H').mean())],
                      [np.nan,np.nan,np.nanmax(T15min),np.nanmax(T15min.rolling('30T').mean()),
                       np.nanmax(T15min.rolling('1H').mean()),np.nanmax(T15min.rolling('3H').mean())],
                      [np.nan,np.nan,np.nan,np.nanmax(T30min),
                       np.nanmax(T30min.rolling('1H').mean()),np.nanmax(T30min.rolling('3H').mean())],
                      [np.nan,np.nan,np.nan,np.nan,np.nanmax(T1h),
                       np.nanmax(T1h.rolling('3H').mean())]])
    return Tout

def find_events_over125(II,thresh):
# finds all events over the threshold, returns durations and max values
    ii=0
    overI_dt=[]
    overI_max=[]
    while True:
        try:
            i0=ii+np.where(II[ii:-1]>thresh)[0][0]
        except IndexError:
            break
        i1=i0+np.where(II[i0:-1]<thresh)[0][0]
        overI_dt.append(i1-i0) # in minutes
        overI_max.append(np.nanmax(II[i0:i1]))
        ii=i1 #update counter
    return overI_dt,overI_max

def downsampledstats_tracking(poa,met,timeres,module,inverter,lat,lon,ele,albedo):
    poawbg_=poa.resample(timeres).mean()
    weather_=met.resample(timeres).mean()
    times_=poawbg_.index
    celltemp=pvlib.temperature.noct_sam(poa_global=poawbg_['effective_irradiance'], temp_air=weather_.temp_air, 
                                                wind_speed=weather_.wind_speed, noct=module.T_NOCT, module_efficiency=module.eta_m)
    poawbg_['cell_temperature']=celltemp.values
    solpos_=pvlib.solarposition.ephemeris(times_, lat, lon)
    tracker_data_=pvlib.tracking.singleaxis(solpos_['apparent_zenith'], solpos_['azimuth'])
    tracker_tilt_=tracker_data_['surface_tilt'].fillna(0)
    tracker_azimuth_=tracker_data_['surface_azimuth'].fillna(0)
    system_ = pvsystem.PVSystem(surface_tilt=tracker_tilt_, surface_azimuth=tracker_azimuth_, 
                            albedo=albedo, module_parameters=module, 
                            temperature_model_parameters={'noct':module.T_NOCT ,'module_efficiency':module.eta_m},
                            inverter_parameters=inverter, modules_per_string=20, strings_per_inverter=5)
    location_=Location(lat, lon,altitude=ele)
    mc_ = ModelChain(system_, location_, clearsky_model='ineichen', transposition_model='perez', 
                            solar_position_method='nrel_numpy', airmass_model='kastenyoung1989',
                            dc_model='cec', #from module specifications
                            ac_model='sandia', #from inverter specifications
                            aoi_model='no_loss', 
                            spectral_model='no_loss', #spectral model 'no loss' otherwise error
                            temperature_model='noct_sam', 
                            losses_model='no_loss')
    mc_.run_model_from_effective_irradiance(poawbg_)
    # get current results
    Vocm0_=pd.DataFrame(mc_.results.dc.v_oc.values,index=times_)
    Iscm0_=pd.DataFrame(mc_.results.dc.i_sc.values,index=times_)
    Vmpp0_=pd.DataFrame(mc_.results.dc.v_mp.values,index=times_)
    Impp0_=pd.DataFrame(mc_.results.dc.i_mp.values,index=times_)
    return Vocm0_,Iscm0_,Vmpp0_,Impp0_



# READ and SAVE fixed tilt results

outpd=pd.DataFrame()
# read the yearly data, then append
for site in sites:
    if site=='Bondville_IL':
        lat=40.05192; lon=-88.37309; ele=230; albedo=0.247; tiltopt=33
    elif site=='Boulder_CO':
        lat=40.12498; lon=-105.2368; ele=1689; albedo=0.199; tiltopt=37
    elif site=='Desert_Rock_NV':
        lat=36.62373; lon=-116.01947; ele=1007; albedo=0.211; tiltopt=35
    elif site=='Fort_Peck_MT':
        lat=48.30783; lon=-105.10170; ele=634; albedo=0.247; tiltopt=40
    elif site=='Goodwin_Creek_MS':
        lat= 34.2547; lon=-89.8729; ele=98.0; albedo=0.2 ; tiltopt=30
    elif site=='Penn_State_PA':
        lat=40.72012; lon=-77.93085; ele=376; albedo=0.252; tiltopt=32
    elif site=='Sioux_Falls_SD':
        lat=43.73403; lon=-96.62328; ele=473; albedo=0.238; tiltopt=38

    tilts=[10,25,tiltopt]
    for tiltang in tilts:
        met=pd.DataFrame()
        poa=pd.DataFrame()
        f = open('../out/overNEC_'+site+'_tilt'+str(tiltang)+'_output_bifacial_meanalbedo.txt', 'w')
        for year in range(2011,2021):
            metyear=pd.read_csv('../met'+site+str(year)+'_tilt'+str(tiltang)+'_bifacial_meanalbedo.csv',index_col=0,parse_dates=[0])
            poayear=pd.read_csv('../poa'+site+str(year)+'_tilt'+str(tiltang)+'_bifacial_meanalbedo.csv',index_col=0,parse_dates=[0])
            #append wanted dataframes
            met=met.append(metyear)
            poa=poa.append(poayear)
            
        # quickly check that air temp and poa are not crazy *for elevation>15... for some reason view factors lead to weird morning POA sometimes*
        solpos=pvlib.solarposition.ephemeris(met.index, lat, lon)
        poa=poa[solpos.elevation>15]
        met=met[solpos.elevation>15]
        
        # calculate results with correct cell temperature    
        times=poa.index
        celltemp=pvlib.temperature.noct_sam(poa_global=poa['effective_irradiance'], temp_air=met.temp_air, 
                                                wind_speed=met.wind_speed, noct=module.T_NOCT, module_efficiency=eta_m)
        poa['cell_temperature']=celltemp.values
        system = pvsystem.PVSystem(surface_tilt=tiltang, surface_azimuth=180, 
                            albedo=albedo, module_parameters=module, 
                            temperature_model_parameters={'noct':module.T_NOCT ,'module_efficiency':module.eta_m},
                            inverter_parameters=inverter, modules_per_string=18, strings_per_inverter=5)
        location=Location(lat, lon,altitude=ele)
        mc = ModelChain(system, location, clearsky_model='ineichen', transposition_model='perez', 
                            solar_position_method='nrel_numpy', airmass_model='kastenyoung1989',
                            dc_model='cec', #from module specifications
                            ac_model='sandia', #from inverter specifications
                            aoi_model='no_loss', 
                            spectral_model='no_loss', #spectral model 'no loss' otherwise error
                            temperature_model='noct_sam', 
                            losses_model='no_loss')
        mc.run_model_from_effective_irradiance(poa)
        # search for current results
        Vocm0_1min=pd.DataFrame(mc.results.dc.v_oc.values,index=times)
        Iscm0_1min=pd.DataFrame(mc.results.dc.i_sc.values,index=times)
        Vmpp0_1min=pd.DataFrame(mc.results.dc.v_mp.values,index=times)
        Impp0_1min=pd.DataFrame(mc.results.dc.i_mp.values,index=times)
        
        
        ## now re run for different dt after resampling the weather
        ''' 5, 15, 30, 60 min '''
        Vocm0_5min,Iscm0_5min,Vmpp0_5min,Impp0_5min=downsampledstats_tilt(poa,met,'5T',module,inverter,lat,lon,ele,albedo,tiltang)
        Vocm0_15min,Iscm0_15min,Vmpp0_15min,Impp0_15min=downsampledstats_tilt(poa,met,'15T',module,inverter,lat,lon,ele,albedo,tiltang)
        Vocm0_30min,Iscm0_30min,Vmpp0_30min,Impp0_30min=downsampledstats_tilt(poa,met,'30T',module,inverter,lat,lon,ele,albedo,tiltang)
        Vocm0_1h,Iscm0_1h,Vmpp0_1h,Impp0_1h=downsampledstats_tilt(poa,met,'1H',module,inverter,lat,lon,ele,albedo,tiltang)
        
        Isc_out=output_tres_table(Iscm0_1min/5,Iscm0_5min/5,Iscm0_15min/5,Iscm0_30min/5,Iscm0_1h/5)
        Voc_out=output_tres_table(Vocm0_1min/5,Vocm0_5min/5,Vocm0_15min/5,Vocm0_30min/5,Vocm0_1h/5)
        Imp_out=output_tres_table(Impp0_1min/5,Impp0_5min/5,Impp0_15min/5,Impp0_30min/5,Impp0_1h/5)
        Vmp_out=output_tres_table(Vmpp0_1min/5,Vmpp0_5min/5,Vmpp0_15min/5,Vmpp0_30min/5,Vmpp0_1h/5)
        np.savetxt('../out/table_Isc_'+site+'_tilt'+str(tiltang)+'_output_bifacial_meanalbedo.txt',Isc_out,delimiter=',')
        np.savetxt('../out/table_Voc_'+site+'_tilt'+str(tiltang)+'_output_bifacial_meanalbedo.txt',Voc_out,delimiter=',')
        np.savetxt('../out/table_Imp_'+site+'_tilt'+str(tiltang)+'_output_bifacial_meanalbedo.txt',Imp_out,delimiter=',')
        np.savetxt('../out/table_Vmp_'+site+'_tilt'+str(tiltang)+'_output_bifacial_meanalbedo.txt',Vmp_out,delimiter=',')

        # save outputs
        stats1min=pd.concat({'Vocm0': Vocm0_1min, 'Iscm0':Iscm0_1min, 'Vmpp0': Vmpp0_1min, 'Impp0':Impp0_1min},axis=1)
        stats1min.to_csv('../out/'+site+'_tilt'+str(tiltang)+'_stats1min_bifacial_meanalbedo.csv',index=True)
        stats5min=pd.concat({'Vocm0': Vocm0_5min, 'Iscm0':Iscm0_5min, 'Vmpp0': Vmpp0_5min, 'Impp0':Impp0_5min},axis=1)
        stats5min.to_csv('../out/'+site+'_tilt'+str(tiltang)+'_stats5min_bifacial_meanalbedo.csv',index=True)
        stats15min=pd.concat({'Vocm0':Vocm0_15min, 'Iscm0':Iscm0_15min, 'Vmpp0': Vmpp0_15min, 'Impp0':Impp0_15min},axis=1)
        stats15min.to_csv('../out/'+site+'_tilt'+str(tiltang)+'_stats15min_bifacial_meanalbedo.csv',index=True)
        stats30min=pd.concat({'Vocm0':Vocm0_30min, 'Iscm0':Iscm0_30min, 'Vmpp0': Vmpp0_30min, 'Impp0':Impp0_30min},axis=1)
        stats30min.to_csv('../out/'+site+'_tilt'+str(tiltang)+'_stats30min_bifacial_meanalbedo.csv',index=True)
        stats1h=pd.concat({'Vocm0':Vocm0_1h, 'Iscm0':Iscm0_1h, 'Vmpp0': Vmpp0_30min, 'Impp0':Impp0_30min},axis=1)
        stats1h.to_csv('../out/'+site+'_tilt'+str(tiltang)+'_stats1h_bifacial_meanalbedo.csv',index=True)
        
        # frequency and duration of events Isc> NEC 125 rule
        nec125=1.25*module.I_sc_ref
        
        # frequency sum of all times when Isc>nec
        over_totalfreq=np.sum(Iscm0_1min/5>nec125)[0]/len(Iscm0_1min)
        over_totalfreq_mpp=np.sum(Impp0_1min/5>nec125)[0]/len(Impp0_1min)
        f.write('Minutes of Isc0>NEC 125% rule: '+str(np.sum(Iscm0_1min/5>nec125)[0])+'\n')
        f.write('Minutes of Imp>NEC 125% rule: '+str(np.sum(Impp0_1min/5>nec125)[0])+'\n')
        over_totaldaytime=np.sum(Iscm0_1min/5>nec125)[0]/np.sum(solpos.elevation>0)
        over_totaldaytime_mpp=np.sum(Impp0_1min/5>nec125)[0]/np.sum(solpos.elevation>0)
        f.write('Frequency (over all times): '+str(over_totalfreq)+'\n')
        f.write('Frequency (over day times): '+str(over_totaldaytime)+'\n')
        f.write('Frequency (over all times) mpp: '+str(over_totalfreq_mpp)+'\n')
        f.write('Frequency (over day times) mpp: '+str(over_totaldaytime_mpp)+'\n')
        
        # find all events
        Isc_overI_dt,Isc_overI_max=find_events_over125(Iscm0_1min/5,nec125)
        Imp_overI_dt,Imp_overI_max=find_events_over125(Impp0_1min/5,nec125)
        f.write('Max duration of Isc over 125 rule:'+str(np.max(Isc_overI_dt))+' min \n')
        f.write('Max duration of Imp over 125 rule:'+str(np.max(Imp_overI_dt))+' min \n')
        f.write('Events Isc>125% in 10 years:'+str(len(Isc_overI_dt))+' \n')
        f.write('Events Imp>125% in 10 years:'+str(len(Imp_overI_dt))+' \n')
        #overIevents=pd.DataFrame({'duration':overI_dt, 'maxIsc0':Isc_overI_max})
        #overIevents.to_csv('../out/'+site+'_tilt'+str(tiltang)+'_stats_overNECevents1min_bifacial_meanalbedo.csv',index=True)
        f.close()
        
        topd= {'site': [site], 'tilt': [tiltang], 'type':['BF, mean'], 
                  'min_Isc_overnec':[np.sum(Iscm0_1min/5>nec125)[0]], 
                  'freq_Isc_overnec':[over_totalfreq],
                  'dayfreq_Isc_overnec':[over_totaldaytime],
                  'nevents_Isc_overnec':[np.max(Isc_overI_dt)],
                  'min_Imp_overnec':[np.sum(Impp0_1min/5>nec125)[0]], 
                  'freq_Imp_overnec':[over_totalfreq_mpp],
                  'dayfreq_Imp_overnec':[over_totaldaytime_mpp],
                  'nevents_Imp_overnec':[np.max(Imp_overI_dt)],
                  }
        outpd=pd.concat([outpd,pd.DataFrame.from_dict(topd)])
        
        

# READ and SAVE tracking results
for site in sites:
    if site=='Bondville_IL':
        lat=40.05192; lon=-88.37309; ele=230; albedo=0.247; tiltopt=33
    elif site=='Boulder_CO':
        lat=40.12498; lon=-105.2368; ele=1689; albedo=0.199; tiltopt=37
    elif site=='Desert_Rock_NV':
        lat=36.62373; lon=-116.01947; ele=1007; albedo=0.211; tiltopt=35
    elif site=='Fort_Peck_MT':
        lat=48.30783; lon=-105.10170; ele=634; albedo=0.247; tiltopt=40
    elif site=='Goodwin_Creek_MS':
        lat= 34.2547; lon=-89.8729; ele=98.0; albedo=0.2 ; tiltopt=30
    elif site=='Penn_State_PA':
        lat=40.72012; lon=-77.93085; ele=376; albedo=0.252; tiltopt=32
    elif site=='Sioux_Falls_SD':
        lat=43.73403; lon=-96.62328; ele=473; albedo=0.238; tiltopt=38

    met=pd.DataFrame()
    poa=pd.DataFrame()
    elevation=pd.DataFrame()
    f = open('../out/overNEC_'+site+'_tracking_output_bifacial_meanalbedo.txt', 'a')
    for year in range(2011,2021):
        metyear=pd.read_csv('../met'+site+str(year)+'_tracking_bifacial_meanalbedo.csv',index_col=0,parse_dates=[0])
        poayear=pd.read_csv('../poa'+site+str(year)+'_tracking_bifacial_meanalbedo.csv',index_col=0,parse_dates=[0])
        #append wanted dataframes
        met=met.append(metyear)
        poa=poa.append(poayear)
        
    # manually clean weird spots
    iweirdpoa=np.where(poa>4000)[0]
    for ii in iweirdpoa:
        poa.effective_irradiance[ii]=0.5*(poa.effective_irradiance[ii-1]+poa.effective_irradiance[ii+1])
    iweirdtemp=np.where(met.temp_air<-40)[0]
    for ii in iweirdtemp:
        met.temp_air[ii]=0.5*(met.temp_air[ii-1]+met.temp_air[ii+1])
    # calculate results with correct cell temperature    
    times=poa.index
    celltemp=pvlib.temperature.noct_sam(poa_global=poa['effective_irradiance'], temp_air=met.temp_air, 
                                                wind_speed=met.wind_speed, noct=module.T_NOCT, module_efficiency=eta_m)
    poa['cell_temperature']=celltemp.values
    solpos=pvlib.solarposition.ephemeris(met.index, lat, lon)
    tracker_data=pvlib.tracking.singleaxis(solpos['apparent_zenith'], solpos['azimuth'])
    tracker_tilt=tracker_data['surface_tilt'].fillna(0)
    tracker_azimuth=tracker_data['surface_azimuth'].fillna(0)
    system = pvsystem.PVSystem(surface_tilt=tracker_tilt, 
                            surface_azimuth=tracker_azimuth, 
                            albedo=albedo,
                            module_parameters=module, 
                            temperature_model_parameters={#'b': -0.0594, 'deltaT': 3,'a': -3.47,  
                                                          'noct':module.T_NOCT ,
                                                          'module_efficiency':eta_m},
                            inverter_parameters=inverter, 
                            modules_per_string=20, 
                            strings_per_inverter=5)
    location=Location(lat, lon,altitude=ele)
    mc = ModelChain(system, location, clearsky_model='ineichen', 
                            transposition_model='perez', 
                            solar_position_method='nrel_numpy', 
                            airmass_model='kastenyoung1989',
                            dc_model='cec', #from module specifications
                            ac_model='sandia', #from inverter specifications
                            aoi_model='no_loss', 
                            spectral_model='no_loss', #spectral model 'no loss' otherwise error
                            temperature_model='noct_sam', 
                            losses_model='no_loss')
    mc.run_model_from_effective_irradiance(poa)
    # get current results
    Vocm0_1min=pd.DataFrame(mc.results.dc.v_oc.values,index=times)
    Iscm0_1min=pd.DataFrame(mc.results.dc.i_sc.values,index=times)
    Vmpp0_1min=pd.DataFrame(mc.results.dc.v_mp.values,index=times)
    Impp0_1min=pd.DataFrame(mc.results.dc.i_mp.values,index=times)
     
    '''5, 15, 30, 60 min'''
    Vocm0_5min,Iscm0_5min,Vmpp0_5min,Impp0_5min=downsampledstats_tracking(poa,met,'5T',module,inverter,lat,lon,ele,albedo)
    Vocm0_15min,Iscm0_15min,Vmpp0_15min,Impp0_15min=downsampledstats_tracking(poa,met,'15T',module,inverter,lat,lon,ele,albedo)
    Vocm0_30min,Iscm0_30min,Vmpp0_30min,Impp0_30min=downsampledstats_tracking(poa,met,'30T',module,inverter,lat,lon,ele,albedo)
    Vocm0_1h,Iscm0_1h,Vmpp0_1h,Impp0_1h=downsampledstats_tracking(poa,met,'1H',module,inverter,lat,lon,ele,albedo)

    Isc_out=output_tres_table(Iscm0_1min/5,Iscm0_5min/5,Iscm0_15min/5,Iscm0_30min/5,Iscm0_1h/5)
    Voc_out=output_tres_table(Vocm0_1min/5,Vocm0_5min/5,Vocm0_15min/5,Vocm0_30min/5,Vocm0_1h/5)
    Imp_out=output_tres_table(Impp0_1min/5,Impp0_5min/5,Impp0_15min/5,Impp0_30min/5,Impp0_1h/5)
    Vmp_out=output_tres_table(Vmpp0_1min/5,Vmpp0_5min/5,Vmpp0_15min/5,Vmpp0_30min/5,Vmpp0_1h/5)
    np.savetxt('../out/table_Isc_'+site+'_tracking_output_bifacial_meanalbedo.txt',Isc_out,delimiter=',')
    np.savetxt('../out/table_Voc_'+site+'_tracking_output_bifacial_meanalbedo.txt',Voc_out,delimiter=',')
    np.savetxt('../out/table_Imp_'+site+'_tracking_output_bifacial_meanalbedo.txt',Imp_out,delimiter=',')
    np.savetxt('../out/table_Vmp_'+site+'_tracking_output_bifacial_meanalbedo.txt',Vmp_out,delimiter=',')
    
    # # save outputs
    stats1min=pd.concat({'Vocm0': Vocm0_1min,'Iscm0':Iscm0_1min,'Vmpp0': Vmpp0_1min,'Impp0':Impp0_1min},axis=1)
    stats1min.to_csv('../out/'+site+'_tracking_stats1min_bifacial_meanalbedo.csv',index=True)
    stats5min=pd.concat({'Vocm0': Vocm0_5min, 'Iscm0':Iscm0_5min,'Vmpp0': Vmpp0_5min,'Impp0':Impp0_5min},axis=1)
    stats5min.to_csv('../out/'+site+'_tracking_stats5min_bifacial_meanalbedo.csv',index=True)
    stats15min=pd.concat({'Vocm0':Vocm0_15min, 'Iscm0':Iscm0_15min,'Vmpp0': Vmpp0_15min,'Impp0':Impp0_15min},axis=1)
    stats15min.to_csv('../out/'+site+'_tracking_stats15min_bifacial_meanalbedo.csv',index=True)
    stats30min=pd.concat({'Vocm0':Vocm0_30min, 'Iscm0':Iscm0_30min,'Vmpp0': Vmpp0_30min,'Impp0':Impp0_30min},axis=1)
    stats30min.to_csv('../out/'+site+'_tracking_stats30min_bifacial_meanalbedo.csv',index=True)
    stats1h=pd.concat({'Vocm0':Vocm0_1h, 'Iscm0':Iscm0_1h,'Vmpp0': Vmpp0_1h,'Impp0':Impp0_1h},axis=1)
    stats1h.to_csv('../out/'+site+'_tracking_stats1h_bifacial_meanalbedo.csv',index=True)
        
    # frequency and duration of events Isc> NEC 125 rule
    nec125=1.25*module.I_sc_ref
    
    # frequency sum of all times when Isc>nec   
    over_totalfreq=np.sum(Iscm0_1min/5>nec125)[0]/len(Iscm0_1min)
    over_totalfreq_mpp=np.sum(Impp0_1min/5>nec125)[0]/len(Impp0_1min)
    f.write('Minutes of Isc0>NEC 125% rule: '+str(np.sum(Iscm0_1min/5>nec125)[0])+'\n')
    f.write('Minutes of Imp>NEC 125% rule: '+str(np.sum(Impp0_1min/5>nec125)[0])+'\n')
    over_totaldaytime=np.sum(Iscm0_1min/5>nec125)[0]/np.sum(solpos.elevation>0)
    over_totaldaytime_mpp=np.sum(Impp0_1min/5>nec125)[0]/np.sum(solpos.elevation>0)
    f.write('Frequency (over all times): '+str(over_totalfreq)+'\n')
    f.write('Frequency (over day times): '+str(over_totaldaytime)+'\n')
    f.write('Frequency (over all times) mpp: '+str(over_totalfreq_mpp)+'\n')
    f.write('Frequency (over day times) mpp: '+str(over_totaldaytime_mpp)+'\n')
    
    # find all events
    Isc_overI_dt,Isc_overI_max=find_events_over125(Iscm0_1min/5,nec125)
    Imp_overI_dt,Imp_overI_max=find_events_over125(Impp0_1min/5,nec125)
    f.write('Max duration of Isc over 125 rule:'+str(np.max(Isc_overI_dt))+' min \n')
    f.write('Max duration of Imp over 125 rule:'+str(np.max(Imp_overI_dt))+' min \n')
    f.write('Events Isc>125% in 10 years:'+str(len(Isc_overI_dt))+' \n')
    f.write('Events Imp>125% in 10 years:'+str(len(Imp_overI_dt))+' \n')
    #overIevents=pd.DataFrame({'duration':overI_dt, 'maxIsc0':Isc_overI_max})
    #overIevents.to_csv('../out/'+site+'_tilt'+str(tiltang)+'_stats_overNECevents1min_bifacial_meanalbedo.csv',index=True)
    f.close()
    
    topd= {'site': [site], 'tilt': ['HSAT'], 'type':['BF, mean'], 
              'min_Isc_overnec':[np.sum(Iscm0_1min/5>nec125)[0]], 
              'freq_Isc_overnec':[over_totalfreq],
              'dayfreq_Isc_overnec':[over_totaldaytime],
              'nevents_Isc_overnec':[np.max(Isc_overI_dt)],
              'min_Imp_overnec':[np.sum(Impp0_1min/5>nec125)[0]], 
              'freq_Imp_overnec':[over_totalfreq_mpp],
              'dayfreq_Imp_overnec':[over_totaldaytime_mpp],
              'nevents_Imp_overnec':[np.max(Imp_overI_dt)],
              }
    outpd=pd.concat([outpd,pd.DataFrame.from_dict(topd)])


# outpd contains the summary statistics
outpd.to_csv('../out/summary_BFmean.csv',sep=',')

## searching the worst day (max Isc for Boulder, CO)

#i_max=np.argmax(Iscm0_1min)
#date_max=Iscm0_1min.index[i_max].date()
# load and plot resource and POA for different conditions
