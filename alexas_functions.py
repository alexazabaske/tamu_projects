#!/usr/bin/env python
# coding: utf-8

import numpy as np
import xarray as xr
import pickle
import pandas as pd

## used by get_PC_components
from sklearn.decomposition import PCA
## used by set new_time_variable
import datetime
from dateutil.relativedelta import relativedelta
## used by Fourier_Analysis
import math 

#############################################################
ep = [-10, 10, 160, 280] 
mc = [-10, 15, 110, 160]

nn30 = [-5, 5, 210, 270]
nn34 = [-5, 5, 190, 240]
#############################################################

#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################
def list_my_functions():
    """ PURPOSE: list the available functions in the alexas_functions library
        inputs: nothing
        returns: nothing """
    
    print('the available functions are: ')
    print('list_my_functions')
    print('get_filename')
    print('get_CMIP_name_list')
    print('set_new_time_variable')
    
    print('uniform_coords')
    print('zonal_avg')
    print('Fourier_Analysis')
    
    print('get_landsea_mask')
    print('extract_region')
    print('mask_out_regions') #old name 'make_mask'
    print('reshape')
    print('get_PC_components')
    print('cc_ev')
    
    print('dump_into_pickle')
    print('open_pickle_data')
   
    print('search_box')
    print('calc_hits_num')
    
    print(':end of list.')
    
#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################
def get_filename(gen, var, exp='historical', name=None, r=1, i=1, p=1, f=1, years='*', realm='Amon', omtype='mod',
                root= 'C:\\Users\\alexa\\Documents\\RESEARCH\\DATA' ):
    
    """ PURPOSE: generate the complete path + filename of the climate model output data or obs data stored in local computer
        
        gen (str): options for this input is: can_ensm, mpi_ensm, had_ensm, CMIP5, or CMIP6. 
        gen is essentially the group or type of model (or obs) ensmbles, which follow a similar naming pattern as well
        var (str): standard variable CF name for CMIP products, such pr, tas, ts, etc.
        
        exp (str): offical CMIP experiment name, such as historical, rcp85, ssp585, etc. 'historical' is the default
        name (str): name of the specific model, such as MPI-ESM, CanESM2, CCSM4, EC-EARTH, etc
        ensemble components (int): r=1, i=1, p=1, f=1 are the defaults 
        if looping through ensemble members, send looping variable to the altering ensemble component (such as r=looping variable)
        years (str): defaut is '*' , so that all years of data can be accessed. 
        or, send a string that specifies the dates in the file name exactly as it appears in the file name
        realm (str): default is 'Amon' . For oceanic variables such as tos, send 'Omon' 
  
        omtype (str): the options for this input are 'mod' or 'obs'. 
        the default is 'mod', but if accessing obs data, you must send the omtype='obs' along with the other args and kwargs
        
        root (str): root='C:\\Users\\alexa\\Documents\\RESEARCH\\DATA' is the default which is specific to my computer. 
        if acessing on a different computer, input the directory path to where all ensemble data folders are. 
        I have the ensmeble folders organized by the variable 'gen' (see above)
        
        returns the entire path + file name to the file (str) (or files, if multiple files for various chunks of time)
 
        ## note: 'C:\\' are for windows. linux uses '/'   """
    
    #####################################
    ##### IF ACCESSING DATA FROM A MODEL
    if omtype=='mod':
        
        ## specs for can_ensm files
        ## loop through can_ensm by sending r in range(1,51)
        if gen=='can_ensm': 
            name='CanESM2'
            #exp='historical'
            
            r-=1 ##to make the math below work
            
            r1 = r%10 ## r only ranges 1-10 for canesm

            ## the tenth's place number will be saved here ((1 if 1-10, 2 if 11-20 (which
            ## will then be reset to be 1 to 10 in the next step if an 11-20 was given to the function)))
            ext_r = int( (r-r1)/10) +1 

            ## override previously submitted r to be the 'new' r that only can range 1-10
            ## the tenth's place inforation was saved in the previous line of code
            r = r1 +1 

            years='195001-202012' ## years for all the can_ensm files
            exp= exp + f'-r{ext_r}' ## the experiment name is where the tenth's place information goes 
            ensm =  f'r{r}i{i}p{p}' ## construct the ensemble name
            ## example: exp_ensm ==  historical-r3_r3i1p1 corresponds to ensemble member 33
            
            ## path is the root directory + navigating through any additional directories to get to the file
            ## can_ensm and mpi_ensm are organized by 'gen' and 'var'
            path = f'{root}\\{gen}\\{var}'
        
        ## specs for mpi_ensm files
        ## loop through mpi_ensm by sending r in range(0,100)
        elif gen=='mpi_ensm': 
            name='MPI-ESM'
            p=3 ## this is constant forall historical and rcp85 ensemble members

            ## specs for historical
            if exp== 'historical':
                years='185001-200512'
                i=1850

            ## specs for rcp85
            elif exp== 'rcp85':
                i=2005
                years='200601-209912'

            ensm = 'r'+str(r).zfill(3) + f'i{i}p{p}'
            path = f'{root}\\{gen}\\{var}'
        
        elif gen=='CMIP5':
            name=name ## passed into this function
            ensm = f'r{r}i{i}p{p}'
            ## path is the root directory + navigating any additional directories to get to the file
            ## CMIP5 and CMIP6 are organized by 'gen', 'ensm', 'exp', and 'var'
            path = f'{root}\\{gen}\\{ensm}\\{exp}\\{var}'
            years='*' ## this is already a defined because it's a function keyword, but putting it here, too, as a reminder

        elif gen=='CMIP6':
            name=name 
            ensm = f'r{r}i{i}p{p}f{f}'
            path = f'{root}\\{gen}\\{ensm}\\{exp}\\{var}'
            years='*' ## this is already a defined because it's a function keyword, but putting it here, too, as a reminder
            
        else: ## other 
            ## r=1, i=1, p=1 as default. can be specified when calling function. 
            ensm = f'r{r}i{i}p{p}'
            path = f'{root}\\{gen}\\{var}'

        ## construct the file name
        ## example: ts_Amon_MPI-ESM_rcp85_r088i2005p3_200601-209912.nc
        filename = f'{var}_{realm}_{name}_{exp}_{ensm}_{years}.nc'
        
        #using 'path' to get to the file, define the full filename includng the directories
        fullfilename = f'{path}\\{filename}'  
        
    ###########################################
    ##### IF ACCESSING DATA FROM OBSERVATIONS
    elif omtype=='obs':
        if gen=='had_ensm':
            ensm=r
            
            if var=='tas' or  var=='temperature_anomaly' or var=='sat': ## use any of these inputs
                var='temperature_anomaly' ## this is the name of the SAT variable in the file, and in the file directory 
                obs_file_set = 'HadCRUT.4.6.0.0.anomalies.'
                
            elif var=='ts' or var=='tos' or var=='sst': 
                var='sst' ## this is the name of the SST variable in the file, and in the file directory 
                obs_file_set  = 'HadSST.3.1.1.0.anomalies.'

            ## directory structure is root + gen + version + var
            path = f'{root}\\{gen}\\Version_4.6\\{var}'
        
            ## file name structure
            filename = f'{obs_file_set}{ensm}.nc'
            
            ## full file name to be returned
            fullfilename = f'{path}\\{filename}' 

    
    ## note: return statement is outside the omtype if statements 
    return fullfilename


#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################
def get_CMIP_name_list(gen, var_list, exp_list, root='D:\\CMIP_DATA'):
   
    """ PURPOSE: Get a list of CMIP5 or CMIP6 model names that there is data available for. 
        This function returns a list of model names for which data is available for all the 
        inputted variables and experiements. 
        ** Then this list of CMIP names can be looped over,
        by sending each model name to the alexas_function.get_filename() to construct the path and filename to 
        each CMIP file **
        
        gen (str): 'CMIP5' or 'CMIP6'
        var_list (list of str): list of variables ('tas', 'ts', 'psl', 'pr', etc.)
        exp_list (list of str): list of experiments ('historical', 'rcp85', 'ssp585')
        root (str): the directory path to the CMIP data inventory excel file
        
        return Name_List (list of str) """
    
    ## open excel file of inventory lists for CMIP5 or CMIP6 data
    df = pd.read_excel (f'{root}\\data_inventory_{gen}_forpython.xlsx')
    df = df.set_index('Model_Name')
    
    ## the following four lines create a list of length 1 if only one var and/or exp was given
    if type(exp_list)==str:
        exp_list = [exp_list]    
    if type(var_list)==str:
        var_list = [var_list]  
    
    ## initialize the Name_List (NL) 
    NL=np.empty(0)
    
    ## loop through all combinations of variables and experiments 
    for i in range(len(exp_list)):
        for j in range(len(var_list)):
            
            ## get inventory for that specific variable in that experiment
            col_name=f'{var_list[j]}_{exp_list[i]}'
            mname = df[col_name].index.values
            inventory = df[col_name].values

            ## keep ONLY the model names that are marked with an 'x', meaning the data is available
            NL_ij = np.array(mname[inventory=='x'])

            ## add the model names to the list
            NL = np.append(NL, NL_ij)
    
    ## get all the model names, and the number of times each model has the data available
    NL_un, NL_counts = np.unique(NL, return_counts=True)
    
    ## This line keeps only the model names that had the maximum number of occurances
    ## in other words, it keeps the model names that are available across all inputted vars and exp
    Name_List = NL_un[NL_counts==np.max(NL_counts)]
    
    return Name_List

#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################
def set_new_time_variable(da_, gen, exp='historical'):
    """ PURPOSE: create a uniform datetime index for gridded monthly CMIP model output. 
        This code converts them all to numpy.datetime64 type of datetimes. It also utilizes
        builtin datetime function and pandas datatime. This is to deal with the various types of 
        datetime objects that appear in the CMIP output files. 
        This code lines the datetime indices to the correct years for their respective experiement,
        and cuts off projections at the year 2100.
    
        da_ (xr.Dataset): the xarray dataset loaded from the cmip model data
        gen (str): 'CMIP5' or 'CMIP6'
        exp (str): the cmip experiment name
    
        returns the same dataset that was inputted but with the redefined time index """

    ### DETERMINE THE START AND END YEAR AND MONTHS BASED ON THE INPUT DATA 
    
    try: ## this first try works for most datetime objects to extract the year and month
        start_year = da_.time.values[0].year
        start_month = da_.time.values[0].month
    except: ##try to convert the unknown object to a pd datetime object to extract the year and month
        start_year = pd.to_datetime(da_.time.values[0]).year
        start_month = pd.to_datetime(da_.time.values[0]).month
    
    
    ### HISTORICAL 
    ## if the experiment is a CMIP5 or CMIP6 era historical simulation
    if exp=='historical':
        try: ## similar try and except to the above scenario
            end_year = da_.time.values[-1].year
            end_month = da_.time.values[-1].month
        except:
            end_year = pd.to_datetime(da_.time.values[-1]).year
            end_month = pd.to_datetime(da_.time.values[-1]).month
            
        ## some CMIP5 and CMIP6 historical simulations have been extended.
        ## This code snips off the extention to line up with the future projection 
        if end_year > 2005 and gen=='CMIP5':
            end_year = 2005
            end_month =12
            
        if end_year > 2014 and gen=='CMIP6':
            end_year = 2014
            end_month =12
    
    ### PROJECTION
    ## if the experiment is a future emission scenario simulation, such as rcp for v5 and ssp for v6 ,
    ## cut off datetimes to December 2099. The extended simulations tend to behave strangely... 
    elif exp[:3]=='rcp' or exp[:3]=='ssp': ## the prefix for available future simulation labels 
        end_year = 2099
        end_month = 12    
    
    
    ## INITIALIZE the timestamps for the new datetime object
    ts1 = datetime.datetime(start_year, start_month, 15, 12, 0, 0)
    tsE = datetime.datetime(end_year, end_month, 15, 12, 0, 0)
    
    print(exp, ts1, tsE)
    
    ## create an array of incremental monthly datetime objects, from start year through the length of the 
    ## original monthly time series 
    all_times = [ts1 + relativedelta(months=a_month) for a_month in range(len(da_.time))]
    
    ## set the time variable in the xarray dataset to the new time series
    da_['time']=all_times
    
    ## slice the entire xarray dataset to the standard start and end years,months 
    da_ = da_.sel(time=slice(ts1, tsE))

    ## convert the datetime objects into numpy.datetime64 objects if they aren't already
    if type(da_['time'].values[0]) != type(np.datetime64(all_times[0])):
        da_['time'] =  [np.datetime64(da_['time'].values[i]) for i in range(len(da_.time))] 
        ## this list comprehension converts each datetime.datetime into a np.datetime64
    
    ## return the entire xarray dataset, with the updated and trimmed time series index
    return da_

#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################
def zonal_avg(idata, slice_begin, slice_end):  
    ##idata should be an xarray data arry, such as da1.sst
    """ PURPOSE: To compute the average value of a variable across latitude bands , taking into account the different areas in 
        each latitude band (apply cosine weights)
        idata (xr.dataarray): an xarray data array to compute the zonal average of. shape should be (time, lat, lon)
        slice_begin (int or float): the bottom latitude of the slice to average over
        slice_end (int or float): the top latitude of the slice to average over
        
        returns an xarray data array of the zonal average of the variable. since the spatial components (lat,lon)
        have been averaged over, the shape of the returned xarray is 1-D (time)
        
        ## note: use (slice_begin, slice_end = -90, 90) for a global average """
    
    ## the if,else is to do the order of slicing correct no matter the sign of the input latitude values
    if idata['lat'][0] < 0:                      
        zslice = slice(slice_begin, slice_end) 
    else:
        zslice = slice(slice_end, slice_begin) 

    ## extract the subset (latitude slice) from the xr.dataarray 
    idata = idata.loc[dict(lat=zslice)]
    
    ## compute the cosine weights for th inputted latitude slice
    cosweight=np.cos(np.deg2rad(idata['lat'])) 
    
    ## average the data longitudinally
    idata2=idata.mean(dim='lon')
     
    ## average the data latitudinally, with the cosine weights applied  
    Zavg=np.sum(idata2*cosweight, axis=1)/np.sum(cosweight)
    
    return Zavg

#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################
def Fourier_Analysis(diff):
    
    """ PURPOSE: Calculate the annual Fourier coeficients (n=1), as well as phase and amplitude calculated from these coefficents
        diff (np.array): a numpy array of 12 values, each representing a monthly mean value. 
        
        returns: fourlist (np.array): a list of seven values 
        fourlist = [amplitude, phase (radians), phase (degrees -180to180), phase (degrees 0to360), a0, a1, b1] """
        
    ## input of 12 points (annual cycle of monthly mean temperature differences)
    ## each month represents 1/12th of the 2pi period
    p = math.pi/12
    months = [p*x for x in range(1,24,2)]   

    ## get the sin and cos of each pi month value
    cosmonths = np.array([math.cos(x) for x in months])   
    sinmonths = np.array([math.sin(x) for x in months])   
    
    ## get data point for that month times the month in its periodic form for sin and cos
    datacosmon = [x*y for x,y in zip(diff,cosmonths)]  
    datasinmon = [x*y for x,y in zip(diff,sinmonths)]  

    ## get ao, a1, b1
    ao=sum(diff)/12
    a1=sum(datacosmon)/6
    b1=sum(datasinmon)/6

    ## get the fourier fit of of the data, by using the fourier coefficients 
    #fit1point = ao + (a1*cosmonths[3]) + (b1*sinmonths[3])
    #fitlist = [ao + a1*x + b1*y for x,y in zip(cosmonths,sinmonths)]   

    ## get the amplitude
    amp = math.sqrt((a1**2)+(b1**2))   

    ## get the phase -180 to 180
    pshr = math.acos(a1/amp)
    if b1 < 0:
        pshr=-pshr
    pshd = (pshr*360)/(2*math.pi)
    
    ## get the phase 0 to 360
    pshdo=pshd
    if b1 < 0:
        pshdo = pshd + 360
        
    ## [amplitude, phase in radians, phase in deg 0-+/-180, ps 0-360, fourier constants]
    fourlist = [amp, pshr, pshd, pshdo, ao, a1, b1]
    
    return np.array(fourlist)
#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################

## < SKIP further documentation for now, come back to add after code is retested > 

def get_landsea_mask(all_percents, perc_val=100, mtype='land', nn='yes'):
    
    """ all_percents: the 2-D numpy array of percent ocean at each grid point,
        calculated by the "calculate_percent_ocean" function
        per_val: the percentage and above of ocean or land you want to keep (the rest is masked out)
        mtype='land', if mtype='sea' it will mask out according the percentage sea you want to keep rather than land
       
        nn='yes', if you want 1's and nans. if nn='no', you get 1's and 0's
       
        model='specifiy', meaning, send the all_percents data array yourself. 
        ## you can also select model = 'mpi' or 'can', and the all_percents file of the respective model will be chosen for you. ##
       
        returns a 2-D (lat, lon) array of 1's and nans (or 0's) that you can apply to the original data field """
    
    
    ## 'land' makes a land mask (keep land, mask out sea regions)
    if mtype=='land':
        all_percents = (100 - all_percents)
    ## 'sea' makes a sea mask (keep sea, mask out land regions)
    elif mtype=='sea':
        all_percents = all_percents
    
    ## this line sets everything below the desired percent of land or percent of sea desired to 0.
    ## for example, if you want to keep 80% land and up, everything below 70% land region will be set to 0. 
    perc_mtype = (all_percents>=perc_val)*all_percents
    
    ## this line sets the remaining percents (ex, above 80%) to a value of 1. 
    perc_mtype[perc_mtype>=perc_val]=1.
    
    ## this line sets the 0's to nans. If nn='no', it will leave them as 0's. 
    if nn=='yes':
        perc_mtype[perc_mtype==0.] = np.nan
    #else: dont set 0 values to nan, leave them as 0
    
    ## return mask of 1's and nans (or 1's and 0's)
    ## the mask is 2-D, same shape as lat/lon grid numpy array
    return perc_mtype

#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################
def extract_region(data, blat, tlat, llon, rlon, mean=False, skipna1=True):
    """ PURPOSE: get a subset of an xarray in a smaller lat lon region
        
        data: an xarray dataset or data array 
        blat, tlat, llon, rlon (ints or floats): the lat and lon bounds for the region of interest 
        (example, alexas_functions.ep or alexas_functions.mc contain the latlon bounds)
        
        mean (bool): default is False. mean=True will compute the mean over the region, 
        (this reduces the dimensions from lat, lon, time to just time)
        (the xarray dataset must have the diminsions 'lat' and 'lon'. 
        use alexas_functions.uniform_coords() if needed before sending the xarray dataset to this function)
     
        return a subsetted xarray dataset, possibly spatially averaged (if mean=True), for the region specified """
    
    region = data.loc[dict( lat=slice(blat, tlat), lon=slice(llon, rlon) )]
    
    
    ## This take the spatial mean over the region with the assumption of equal areas per latlon box.
    ## if averaging over large span of latitude, cosine weights should be applied based on latitude 
    ## (see alexas_functions.zonal_avg)
    if mean==True:
        region_mean_value = region.mean(dim=['lat','lon'], skipna=skipna1)
        return region_mean_value
    
    ## return the dataset trimmed to the desire region. no averaging or other changes made
    else:
        return region
    
#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################
def mask_out_regions(data, regionlist, mask_inward=True):
    """ PURPOSE: to create a 2-D mask (numpy array) that can be applied to an entire spatial field of a variable. 
        The function will convert the regions of interest, given by regionlist, and convert them to nans (or keep as ones).
        The rest of the gridpoints will be ones (or convertd to nans).
        This is so that the mask can be broadcasted to the entire datafield of the same shape as the input dataset. 
        
        data (xr.dataarry): spatial latlon field of a variable, with certain regions to be masked out
        regionlist (list):  should be (number of regions) by 4, 4 being the 4 vertices of the region (such as the EP, for example)
        mask_inward=True is the default option. This option will mask out the regions you are providing,
        and keep everything else. if set to mask_inward=False, the data inside the regions will remain,
        and everything outside of those regions masked out (set to zero). 
        
        returns a 2-D numpy array of ones and nans. It is the same size as the latlon field of the given variable 
        (for example, the CanESM2 latlon grid for precip is (64 by 128), so the return would be that size) """
    
    
    ## get all the latlon values from full dataset 
    alllatlons = np.asarray(np.meshgrid(data['lon'], data['lat'])) 
    alllatlons_list = alllatlons.reshape(2, alllatlons.shape[1]*alllatlons.shape[2])
    ##  note: alllatlons_list.shape = (2, latsize*lonsize)
    
    ## initialize an empty array , to get a list of the latlon points in each region all together
    regslatlons_list = np.empty((2,1))
    
    ## loop through list of regions
    ## the goal of this loop is to get a list of latlon points that is formatted simarly to (alllatlons_list)
    for r in regionlist:
        ## reduce dataset to region, get those lat lon values, put into list
        rr = alexas_functions.extract_region(data, r[0], r[1], r[2], r[3]) ## might need to delete the 'alexas_functions.' in front...
        rrlatlons = np.asarray(np.meshgrid(rr['lon'], rr['lat'])) 
        rrlatlons_list = rrlatlons.reshape((2,rrlatlons.shape[1]*rrlatlons.shape[2]))
        
        ## append list of latlons from specific region to full list of regions' latlons 
        regslatlons_list = np.append(regslatlons_list, rrlatlons_list, axis=1)
      
    
    ## create an array the same size as full dataset latlons
    mask = np.copy(alllatlons_list) 

    ## Options for masking or keeping each latlon pair
    twoones = np.array([1., 1.])
    twonans = np.array([np.nan, np.nan])
    #twozeros = np.array([0, 0])
    
    ## the default, meaning the INPUT regions ARE MASKED, other regions remain
    if mask_inward == True:
        plugin = twonans
        mask[:] = 1. 
       
    ## with this option the INPUT regions REMAIN, other regions are masked
    if mask_inward == False:
        plugin = twoones
        mask[:] = np.nan 

    ## loop through each lat lon point in alllatlons, check if it is in the regions.
    for ll in range(0, alllatlons_list.shape[1]):
        for nn in range(0, regslatlons_list.shape[1]):
            
            ## if the point is in the given region(s)      
            if alllatlons_list[0,ll] == regslatlons_list[0,nn] and alllatlons_list[1,ll] == regslatlons_list[1,nn]:
                mask[:,ll] = plugin

            # outside the region(s) of interest 
            else:
                mask[:,ll] = mask[:,ll]        
                       
                
    ## reshape the mask to the original shape of the inputted latlonfield
    mask = mask.reshape(2,alllatlons.shape[1],alllatlons.shape[2])
    
    ## returning it like this returns it as a 2-D lat-lon grid of ones and nans
    return mask[0,:,:] 
          
#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################
def reshape(data, var='pr'):
    """ PURPOSE: reshape a 3-D dataset (time, lat, lon) to 2-D dataset (time, space)
    data (xr.dataset): data to be reshaped, 
    var (str): default var='pr'
    the return is a 2-D numpy array """

    stime = data.sizes['time']
    slat  = data.sizes['lat']
    slon  = data.sizes['lon']
    
    reshaped_data = data[var].values.reshape((stime,slat*slon))
    
    return reshaped_data

#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################
def get_PC_components(data, number_of_PCs, opt='grid'):
    """ PURPOSE: Calculate principle components of a 3-D variable (time, lat,lon)
        data (xr.dataarray): a dataset variable with dimensions (time, lat, lon) 
        number_of_PCs (int): the number of components you want returned. generally 10 is a good number, components
        beyond that usually have very low amount of variance explained.
        opt (str): default is 'grid'. other options are 'ts' or 'all' (meaning both 'grid' and 'ts')
            'grid' will return the spatial principle components (same shape as input dimensions (lat,lon)). 
            'ts' will return the time series (same shape as input dimension (time)). 
            'all' will return both of the above options 
        
        return options:
        1) list_of_PC_grid: list (with size=number_of_PCs) of the spatial princible components (shape: lat,lon)
        2) list_of_PC_ts: list (with size=number_of_PCs) of the princible components time series (shape: time)
        3) explained variance: list of explained variance % for each PC
        
            if opt='grid' -> 1 & 3 are returned
            if opt='ts' -> 2 & 3 are returned
            if opt='all' -> 1, 2, & 3 are returned
        
        ## note: two other functions nested in this function: remove_nans() and insert_nans(), these are for dealing
        ## with nans in data set, as the PC function will not accept nan values. """

    ########################################################################################################
    ########################################################################################################   
    def remove_nans(redata):
        """ PURPOSE: remove nans from a 2-D dataset, while conserving indormation about their indice values
        redata: reshaped (2-dimensional, (time, space)) data, after reshape function """

        ## create an index, numbering each point
        inds_all = np.arange(0, len(redata.flatten()), 1)

        ## find inds that contain nan pr data (masked)
        inds_nan = np.where( np.isnan(redata.flatten())==True )[0] 

        ## delete the nans from the data, and the corresponding indices numbers
        redata_nn = np.delete(redata, inds_nan)
        inds_all_nn = np.delete(inds_all, inds_nan)

        #print(len(inds_nan), len(redata_nn))

        ## reshape back to 2-D
        redata_nn = redata_nn.reshape( redata.shape[0] ,  int( len(redata_nn)/(redata.shape[0]) ))
        inds_all_nn = inds_all_nn.reshape( redata.shape[0] ,  int( len(redata_nn.flatten())/(redata.shape[0]) ))
        inds_all = inds_all.reshape( redata.shape )

        ## redata_nn is the reshaped data without nans. 
        ## inds_all and inds_all_nn are the indices of redata before and after the nans were removed. 
        ## these inds arrays are used for inserting nans back into the grid after the pca 
        return redata_nn, inds_all, inds_all_nn
    
    ########################################################################################################
    ########################################################################################################     
    def insert_nans(PC0, inds_1d, inds_nn_1d):
        """ PURPOSE: inserts nans into the spatial field of a PC component where they were previously
        back into indices where they were removed """

        ## C0 is a pca._components[i], straight from the PCA function (so it's without nans)
        ## need 1 dimension of inds_all, and inds_all_nn.
        ## 1 dimension means 1 time stamp, since the location of the nans do not vary in time.

        ## these plots allow for comparison of where inds_nn_1d is not linear, because of removed nan indices
        #plt.plot(inds_nn_1d)
        #plt.plot(inds_1d)
        #plt.show()

        ## im not sure if this step is necessary but it makes the code easier for me to understand
        a = inds_1d.copy()
        b = inds_nn_1d.copy()
        PC00 = PC0.copy()

        #print(len(a), len(b))

        ## this figures out where to insert a -999, to be turned into nans in next steps
        for i in range(len(a)):
            if ( b[i] - a[i] ) !=0:

                b = np.insert(b, i, -999)
                PC00 = np.insert(PC00, i, -999)

        ## these plots show how the indicices line up, and nans are labeled as -999
        #plt.plot(a)
        #plt.plot(b)

        ## now the PC component can be reshaped to the original size of the input data latlon grid 
        PC000 = PC00.reshape(latsize,lonsize)

        ## convert -999's to nans
        PC000[PC000==-999] = np.nan

        return PC000
        ########################################################################################################
        ########################################################################################################  
        ## End of nested functions (used to managing nan values)
    ## Beginning of get_PC_components() function
    
    ## this step will fail is 'lat' is listed as 'latitude' in the dataarray
    ## use alexas_functions.uniform_coords() to fix
    latsize=len(data['lat'].values)
    lonsize=len(data['lon'].values)
    timesize_mon = len(data['time'].values)
    
    ## reshape pr data from 3-D (time, lat, lon) to 2-d (time, space)
    re_data = reshape(data)

    ## remove nans that were put in place from masking scenarios
    ## re_data_nn is the data without nans, and the inds of the data with and without nans
    re_data_nn, inds_all, inds_all_nn = remove_nans(re_data)

    ## generate PC function
    pca = PCA(n_components = 20) 
    ## ncomponents=20 because most of the information is in the first few principle components
   
    ## fit to the data
    pca.fit(re_data_nn)
    
    ## option for if only PC spatial fields to be returned, or if both spatial and time series to be returned
    if opt=='grid' or opt=='all':
        ## initialize list for 2-D numpy arrays to be added. The numpy arrays are the PC spatial fields
        list_of_PC_grid = []
        
        for gi in range(number_of_PCs):
            ## insert nans (masked out regions, missing data, etc) to their correct locations
            PC_gi = insert_nans(pca.components_[gi], inds_all[0], inds_all_nn[0])
            list_of_PC_grid.append(PC_gi)
            
        ## if PC time series are NOT desired to be returned from the function, return the data here       
        if opt=='grid': 
            ## RETURN OPTION 1 & 3 (see doc string)
            return list_of_PC_grid, (pca.explained_variance_ratio_ *100)[0:number_of_PCs]
    
    ## option for if only PC time series to be returned, or if both spatial and time series to be returned
    if opt=='ts' or opt=='all':
        ## initialize a list for 2-D numpy arrays to be added to. The numpy arrays are the PC time series 
        list_of_PC_ts = []
        
        for ti in range(number_of_PCs):
            ## compute the time series from the pc's . note, the nans to not need to be reinserted
            PC_ti = np.dot(re_data_nn,pca.components_[ti])
            list_of_PC_ts.append(PC_ti)
         
        ## if PC spatial fields are NOT desired to be returned from the function, return the data here   
        if opt=='ts':
            ## RETURN OPTION 2 & 3 (see doc string)
            return list_of_PC_ts, (pca.explained_variance_ratio_ *100)[0:number_of_PCs]
        
        ## if PC spatial field AND PC time series are desired to be returned from the function, return all of it here
        elif opt=='all': 
            ## RETURN OPTION 1, 2 & 3 (see doc string)
            return list_of_PC_grid, list_of_PC_ts, (pca.explained_variance_ratio_ *100)[0:number_of_PCs]

#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################
def cc_ev(data1, data2):
    """ PURPOSE: Calculate the the correlation coeficient and the explained variance of two 1-D arrays
        data1 (numpy array) & data2 (numpy array) need to be a 1-D array. Such as a single variable over time.
        len(data1) == len(data2) = True in order to work
        
        returns a two element list, the correlation coeficient and the explained variance (r^2) """
    
    ## calculate the correlation coeficient. only need the first element in the corrcoef matrix
    cc = np.corrcoef(data1, data2)[1,0]
    ev = cc **2    ## calculate r^2 (explained variance)
    
    return [cc, ev]   

#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################
## this function saves the variable into a file that you can open later
def dump_into_pickle(fname, datatodump):
    """
    fname is the name of the file where to save the variable. it should be a string. add '.pkl' as the extension
    datatodump is anything. a list, numpy array, a float, an xarray dataset. 
    """
    
    ## this dumps the variable into the file
    with open(fname, 'wb') as handle:
        pickle.dump(datatodump, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    ## no need to return anything with this function, your file should be in your directory
    
## this is how to open the data into the pickle file
def open_pickle_data(fname):
    """fname is the name (string) of the pickle file where the variable is stored"""
    datatoopen = pickle.load( open(fname,  'rb') )
    
    ## return the data 
    return datatoopen

########## EXAMPLE USING PICKLE FILES ##########
# ## a can be anything. lists, arrays pandas data frames, an xarray dataset, a string, anything 
# a = [0., 'one']

# ## example of dumping the variable *a* into a file to open later
# dump_into_pickle('example.pkl', a)

# ## loading the data from *a* into a new variable from the pickle file
# b = open_pickle_data('example.pkl')

# print(a==b)
################################################
#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################
def uniform_coords(dataset, old_label_list, new_label_list):
    """ PURPOSE: rename coordinates in an xarray dataset 
        dataset (xr.Dataset): the dataset with coordinate names to be changed 
        old_label_list: list of the coordinate names in the dataset (example: ['latitude', 'longitude']) 
        new_label_list: list of the new names for the coordinates (example: ['lat', 'lon']
        
        returns the dataset, but with the new names for the coordinates """
    
    if len(old_label_list) == len(new_label_list):
        
        for i in range(len(old_label_list)):
            #if old_label_list[i] in data_array.indexes.keys(): ##commenting this out.. I don't I need this line
            
            dataset = dataset.rename({old_label_list[i]: new_label_list[i]})
    else:
        print(' len(old_label_list) == len(new_label_list) needs to be true, currently it is not. ')
    
    return dataset

#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################
def search_box(data, index_seas, static_box, PCTS, ev_index, latslice=slice(-20,20), lonslice=slice(100,300), 
               latincr_rng = range(2,12,2), lonincr_rng = range(10,44,4), minsize=180):
    """PURPOSE:
    data: an xarray DataArray that has already been resampled to SEASONAL means
    use data.resample(time="Q-NOV").mean() for DJF, MAM, etc. , before sending to function 
    staticbox (list([])): list of static box or boxes. A single static box = list(tlat, blat, rlon, llon)
    latslice (slice): latitude
    lonslice(slice):
    latincr_rng (range)
    lonincr_rng (range) """

    seas_names = ['summer', 'fall', 'winter', 'spring']
    season_nums = np.array([8,11,2,5]) #JJA, SON, DJF, MMA

    ## initialize a lat list and a lon list of 4 empty numpy arrays,
    ## to store the central lat and lon points of potential boxes (greater corr than nn34)
    cenlons = [np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0)]
    cenlats = [np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0)]
    
    ## given the bounds of latitude and longitudes, get array of lat and lon values from the input data 
    lator_rng = data.sel(lat=latslice).lat.values
    lonor_rng= data.sel(lon=lonslice).lon.values

    ## the first and second for loops will loop through the SIZE (number of increments) of the box in both dimensions
    ## the third and fourth for loop will loop through the ORIGIN (central lat and lon) of the box in each box size

    #### ---- 1) BEGIN LOOP THROUGH LAT SIZE 
    for latincr in latincr_rng:

        print(' ')
        print('lat increment: ', latincr)
        print('lon increments: ', end=' ')
        
        #### ---- 2) BEGIN LOOP THROUGH LON SIZE
        for lonincr in lonincr_rng:
            print(lonincr, end=',') 

            #### ---- 3) BEGIN LOOP THROUGH LAT ORIGIN
            for lator in range(len(lator_rng)-latincr):
                
                ## extract the values of the top and bottom latitude points 
                tlat, blat  = (lator_rng[lator+latincr]), (lator_rng[lator])
                ## calculate the latsize (vertical height of the box)
                latsize= tlat-blat

                #### ---- 4) LON ORIGIN 
                for lonor in range(len(lonor_rng)-lonincr):
                    
                    ## extract the values of the top and bottom longitude points
                    rlon, llon = (lonor_rng[lonor+lonincr]), (lonor_rng[lonor])
                    ## calculate the lonsize (horizontal width of the box)
                    lonsize= rlon-llon
                    
                    ## before moving on to further calculations, determine if the box size meets size requirement
                    if lonsize*latsize >=minsize:
                        
                        ## calculate the spatially averaged seasonal SST within the lat-lon box in the loop
                        box_seas = extract_region(data, blat,tlat,llon,rlon,mean='yes', skipna1=True)

                        #### LOOP THROUGH 4 SEASONS 
                        for s in range(4):
                            
                            ## season nums correspond to the numeric month of each season's first month
                            ## 2,5,8,11
                            seas=season_nums[s]
                            
                            ## <add code>
                            box_1seas = box_seas.sel( time = box_seas['time.month'] == seas )['ts']
                            
                            ## nn34_seas was computed in the previous cell
                            index_1seas = index_seas.sel( time = index_seas['time.month'] == seas )['ts']

                            ## calculate the ndi value, using the nn34 and box within the loop
                            ndi_1seas = index_1seas - box_1seas

                            ## this is where I'd loop through multiple PCs, but not necessary right now.
                            #for pci in range(num_of_pcs):
                            pci=0 ## first principle component

                            ## this stores the correlation and R2 value for one box possibility in all seasons
                            cc_box, ev_box = cc_ev(ndi_1seas, PCTS[s][pci])

                            ## this is where the central lat and lon of a box candidate get added
                            ## <add code>
                            if ev_box > ev_index[s, 0, pci]: ## 2D is 0 because num_ensms == 1
                                
                                ## <ADD MORE CODE from here all the way through!>
                                cenlon = lonor_rng[lonor+int(lonincr/2)]
                                cenlat = lator_rng[lator+int(latincr/2)]

                                cenlons[s] = np.append(cenlons[s], cenlon)
                                cenlats[s] = np.append(cenlats[s], cenlat)
                                #corrs = np.append(corrs, cc_box)

                                
    print('...done.')                           
    return cenlats, cenlons #,boxsizes 

#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################
def calc_hits_num(hits_lats, hits_lons):
    """"""
    lats_un = np.unique(hits_lats)
    lons_un = np.unique(hits_lons)


    latlongrid = np.zeros((lats_un.size, lons_un.size))
    
    for a in range(len(lats_un)):
        latlongrid[a,:]=lats_un[a]

    for o in range(len(lons_un)):
        latlongrid[:,o]=lons_un[o]

    num_hits = np.zeros_like(latlongrid)

    for cla in range(len(lats_un)):
        curr_lat = lats_un[cla]
        
        ## get indices of all hits at this current lat
        inds= np.where(hits_lats==curr_lat)[0]
        
        ## get the corressponding longitudes for each lat hit
        curr_lons_al  = hits_lons[inds]
        
        ## sum up hits 
        for clo in range(len(curr_lons_al)):
            ind = np.where([latlongrid[cla]==curr_lons_al[clo]])[1][0]
            num_hits[cla,ind]+=1
 
    return num_hits, lats_un, lons_un

#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################