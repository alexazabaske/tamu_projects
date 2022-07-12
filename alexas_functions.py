#!/usr/bin/env python
# coding: utf-8

# In[ ]:

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

#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
def list_my_functions():
    print('the available functions are: ')	
    print('list_my_functions')
    print('test_function')
    print('get_filename')
    print('get_landsea_mask')
    print('extract_region')
    print('cc_ev')
    print('mask_out_regions') #old name 'make_mask'
    print('reshape')
    print('get_PC_components')
    print('dump_into_pickle')
    print('open_pickle_data')
    print('zonal_avg')
    print('Fourier_Analysis')
    print('get_CMIP_name_list')
    print('set_new_time_variable')
    print(':end of list.')

    ### TEST

#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
def test_function():
    print('hello!')
    return 8
    
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
def get_filename(gen, var, exp='historical', name=None, r=1, i=1, p=1, f=1, years='*', realm='Amon', omtype='mod',
                root= 'C:\\Users\\alexa\\Documents\\RESEARCH\\DATA' ):
    
    """ all inputs are strings, and the filename return is a string
        the full filename is returned, including the full path to the file
        
        gen: can_ensm, mpi_ensm, CMIP5, etc
        exp: experiment 'historical', 'rcp85' etc 
        var: standard variable name 'pr', 'tas', etc
        name: model name example 'MPI-ESM'
        ensemble components: r=1, i=1, p=1, f=1
        years='*': or specify full years range in the file name
        realm='Amon'
        
        omtype='obs', 'mod'
        
        root='C:\\Users\\alexa\\Documents\\RESEARCH\\DATA' which is specific to my computer. 
        input the directory path to where all ensemble data folders are 
        
        ## 'C:\\' are for windows. linux uses '/'   
        
        """
    
    
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

        ## construct the full file name
        ## example: ts_Amon_MPI-ESM_rcp85_r088i2005p3_200601-209912.nc
        filename = f'{var}_{realm}_{name}_{exp}_{ensm}_{years}.nc'
        
        #using 'path' to get to the file, define the full filename includng the directories
        fullfilename = f'{path}\\{filename}'  
        
    #####################################################################################################
    elif omtype=='obs':
        if gen=='had_ensm':
            ensm=r
            
            if var=='tas' or  var=='temperature_anomaly' or var=='sat': ## use any of these inputs
                var='temperature_anomaly' ## this is the name of the SAT variable in the file, and in the file directory 
                obs_file_set = 'HadCRUT.4.6.0.0.anomalies.'
                
            elif var=='ts' or var=='tos' or var=='sst': 
                var='sst' ## this is the name of the SST variable in the file, and in the file directory 
                obs_file_set  = 'HadSST.3.1.1.0.anomalies.'

            
            path = f'{root}\\{gen}\\Version_4.6\\{var}'

            filename = f'{obs_file_set}{ensm}.nc'
            
            fullfilename = f'{path}\\{filename}' 

    
    return fullfilename

#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
def get_landsea_mask(all_percents=None, perc_val=100, mtype='land', nn='yes'):
    
    """all_percents: the 2-D numpy array of percent ocean at each grid point,
        calculated by the "calculate_percent_ocean" function
        
        per_val: the percentage and above of ocean or land you want to keep (the rest is masked out)
        
        mtype='land', if mtype='sea' it will mask out according the percentage sea you want to keep rather than land
       
       nn='yes', if you want 1's and nans. if nn='no', you get 1's and 0's
       
        model='specifiy', meaning, send the all_percents data array yourself. 
        ## you can also select model = 'mpi' or 'can', and the all_percents file of the respective model will be chosen for you. ##
       
        returns a 2-D (lat, lon) array of 1's and nans (or 0's) that you can apply to the original data field"""
    

    
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

#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
def extract_region(data, blat, tlat, llon, rlon, mean='no', skipna1=True):
    """
    -data should be an xarray dataset 
    -four vertices of the region should be provided (the ep or mc regions that are defined, for example)
    -blat, tlat, llon, rlon
    -an optional keyword of whether you want the mean over the region (therefore reducing the
     dimensions by having only one point at each time step)
     
     the return is an xarray dataset for the region specified
    """
    
    region = data.loc[dict( lat=slice(blat, tlat), lon=slice(llon, rlon) )]

    if mean=='yes':
        region_mean_value = region.mean(dim=['lat','lon'], skipna=skipna1)
        return region_mean_value
    
    else:
        return region
    
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
def cc_ev(data1, data2):

    cc = np.corrcoef(data1, data2)[1,0]
    ev = cc **2    

    return [cc, ev]
   
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
def mask_out_regions(dataset, regionlist, mask_inward=True):
    """
    -input an xarray dataset 
    -regionlist should be (number of regions) by 4, 4 being the 4 vertices of the region (such as the EP, for example)
    -mask_inward is the default option. it means that you want to mask the regions you are providing,
    and keep everything else. if set to False, you will keep the data in the regions, and mask out everything else. 
    -this function returns a mask, it is a 2-D numpy array of ones and nans. It is the same size as the lat/lon field 
    of the given variable (for example, the CanESM lat/lon grid for precip is (64 by 128))
    
    This function makes a list of all the lat/lon combinations of the given dataset. Then, it makes a 
    list of the lat/lon combinations within the regions specified. 
    
    It intitializes a mask that is the size of the lat/lon grid 
        of ones if mask_inward =True
        or of nans if mask_inward=False 
    
    Then, at each lat/lon point in the mask's grid (which is the same 
    size as the variable's), it checks whether or not the point is in the region or not. 
    
        if mask_inward = True, it turns the one at the point into a nan. 
        if mask_inward = False, it turns the nan at the point into a one. 
    
    Once each point in the mask has been checked, it is reshaped and returned (numpy array).
    This mask can now by multiplied with the precip values, to mask the appropriate values.
    """
    
    
    ##get lat lon values from full dataset, put into list
    alllatlons = np.asarray(np.meshgrid(dataset.lon, dataset.lat)) 
    alllatlons_list = alllatlons.reshape(2, alllatlons.shape[1]*alllatlons.shape[2])
    
    ## initialize the list of points within all input regions
    regslatlons_list = np.empty((2,1))
    
    ## loop through list of regions
    for r in regionlist:
        ##reduce dataset to region, get those lat lon values, put into list
        rr = alexas_functions.extract_region(dataset, r[0], r[1], r[2], r[3]) 
        rrlatlons = np.asarray(np.meshgrid(rr.lon, rr.lat)) 
        rrlatlons_list = rrlatlons.reshape((2,rrlatlons.shape[1]*rrlatlons.shape[2]))
        
        ## append list of latlons from specific region to full list of regions' latlons 
        regslatlons_list = np.append(regslatlons_list, rrlatlons_list, axis=1)
      
    
    ## create a mask that is the same size as full dataset latlons
    mask = np.copy(alllatlons_list) 

    ## plug in value options for masking
    #twozeros = np.array([0, 0])
    twoones = np.array([1., 1.])
    twonans = np.array([np.nan, np.nan])

    
    ## the default, meaning the INPUT regions ARE MASKED, other regions remain
    if mask_inward == True:
        plugin = twonans
        mask[:] = 1. 
       
    ## with this option the INPUT regions ARE SAVED, other regions are masked
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
                       
                
    ##reshape the mask
    mask = mask.reshape(2,alllatlons.shape[1],alllatlons.shape[2])
    
    ## returning it like this returns it as a 2-D lat-lon grid of ones and nans
    return mask[0,:,:] 
          
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
def reshape(data):
    """reshape 3-D dimensional data to 2-D data
    input is an xarray, the return is a numpy array
    (the original version of this function written by NG for ATMO 632 class)"""

    stime = data.sizes['time']
    slat  = data.sizes['lat']
    slon  = data.sizes['lon']
    
    reshaped_data = data['pr'].values.reshape((stime,slat*slon))
    return reshaped_data

#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
def get_PC_components(data, number_of_PCs, opt='grid'):
    """ send in xarray of data for PCA, and the number of components to output 
        opt: 'grid', 'ts', 'all' """

    #################################################################################################
    #################################################################################################   
    def remove_nans(redata):
        """send reshaped (2-dimensional) data, after reshape function"""

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
    
    #################################################################################################
    #################################################################################################     
    def insert_nans(PC0, inds_1d, inds_nn_1d):
        """inserts nans into PC component back into indices where they were removed"""

        ## PC0 is a pca._components[i], stragith from the PCA function (so it's without nans)
        ## need 1 dimension of inds_all, and inds_all_nn.
        ## 1 dimension means 1 time stamp, since the location of the nans do not vary in time.

        ## these plots allow for comparison of where inds_nn_1d is not linear, because of removed nan indices
        #plt.plot(inds_nn_1d)
        #plt.plot(inds_1d)
        #plt.show()

        ## im not sure if this step is necessary but it makes the code cleaner anyhow
        a = inds_1d.copy()
        b = inds_nn_1d.copy()
        PC00 = PC0.copy()

        #print(len(a), len(b))

        ## this figures out where to insert 'nans' (-999, to be turned into nans)
        for i in range(len(a)):
            if ( b[i] - a[i] ) !=0:

                ## i don't actually need this line, but i'm keeping it in here for now
                diff = b[i] - a[i]

                b = np.insert(b, i, -999)
                PC00 = np.insert(PC00, i, -999)

        ## these plots show how the indicices line up, and nans are labeled as -999
        #plt.plot(a)
        #plt.plot(b)

        ## now the PC component can be reshaped to the original size of the data lat/lon grid 
        PC000 = PC00.reshape(latsize,lonsize)

        ## convert -999's to nans
        PC000[PC000==-999] = np.nan

        return PC000
        #################################################################################################
        #################################################################################################  
    
    
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
   
    ## fit to the data
    pca.fit(re_data_nn)
    
    #########################################################
    if opt=='grid' or opt=='all':
        list_of_PC_grid = []
        
        for gi in range(number_of_PCs):

            PC_gi = insert_nans(pca.components_[gi], inds_all[0], inds_all_nn[0])
            list_of_PC_grid.append(PC_gi)
                
        if opt=='grid': 
            return list_of_PC_grid, (pca.explained_variance_ratio_ *100)[0:number_of_PCs]
        #elif opt=='all': 
        #   continue  
    #########################################################
    #########################################################
    if opt=='ts' or opt=='all':
        list_of_PC_ts = []
        
        for ti in range(number_of_PCs):

            PC_ti = np.dot(re_data_nn,pca.components_[ti])
            list_of_PC_ts.append(PC_ti)
            
        if opt=='ts':
            return list_of_PC_ts, (pca.explained_variance_ratio_ *100)[0:number_of_PCs]
        
        elif opt=='all': 
            return list_of_PC_grid, list_of_PC_ts, (pca.explained_variance_ratio_ *100)[0:number_of_PCs]
    #########################################################
          
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
## this function saves the variable into a file that you can open later
def dump_into_pickle(fname, datatodump):
    """
    fname is the name of the file where to save the variable. it should be a string. make sure to add '.pkl' as the extension
    datatodump is anything. a list, nummpy array, a float, an axarray dataset. 
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
###############################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
def uniform_coords(data_array, old_label_list, new_label_list):
    
    if len(old_label_list) == len(new_label_list):
        for i in range(len(old_label_list)):
            if old_label_list[i] in data_array.indexes.keys(): ##if not in
                data_array = data_array.rename({old_label_list[i]: new_label_list[i]})

    return data_array

#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
def zonal_avg(idata, slice_begin, slice_end):  
    ##idata should be an xarray data arry, such as da1.sst
  
    if idata['lat'][0] < 0:                      
        zslice = slice(slice_begin, slice_end)
    else:
        zslice = slice(slice_end, slice_begin)

    idata = idata.loc[dict(lat=zslice)]
    cosweight=np.cos(np.deg2rad(idata['lat'])) 
    
    idata2=idata.mean(dim='lon')
    
        
    Zavg=np.sum(idata2*cosweight, axis=1)/np.sum(cosweight)
    
    
    return Zavg

#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
def Fourier_Analysis(diff):
    
    ## input of 12 points (annual cycle of monthly mean temperature differences)

    ## each month represents 1/12th of the 2pi period
    p = math.pi/12
    months = [p*x for x in range(1,24,2)]   

    ## get the sin and cos of each month
    cosmonths = np.array([math.cos(x) for x in months])   
    sinmonths = np.array([math.sin(x) for x in months])   
    
    ## get data point for that month x the month in its periodic form for sin and cos
    datacosmon = [x*y for x,y in zip(diff,cosmonths)]  
    datasinmon = [x*y for x,y in zip(diff,sinmonths)]  

    ## get ao, a1, b1
    ao=sum(diff)/12
    a1=sum(datacosmon)/6
    b1=sum(datasinmon)/6

    ## get the fit
    #fit1point = ao + (a1*cosmonths[3]) + (b1*sinmonths[3])
    fitlist = [ao + a1*x + b1*y for x,y in zip(cosmonths,sinmonths)]   

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

##########################################################################################################################
##########################################################################################################################
##########################################################################################################################
def get_CMIP_name_list(gen, var_list, exp_list, root='D:\\CMIP_DATA'):
   
    """ PURPOSE: Get a list of CMIP5 or CMIP6 model names that there is data available for. 
        This function returns a list of model names for which data is available for all the 
        inputted variables and experiements. 
        
        gen (str): 'CMIP5' or 'CMIP6'
        var_list (list of str): list of variables ('tas', 'ts', 'psl', 'pr', etc.)
        exp_list (list of str): list of experiments ('historical', 'rcp85', 'ssp585')
        root (str): the directory path to the CMIP data inventory excel file
        
        return Name_List (list of str) 
        """
    
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

##########################################################################################################################
##########################################################################################################################
##########################################################################################################################
def set_new_time_variable(da_, gen, exp='historical'):
    """
    PURPOSE: create a uniform datetime index for gridded monthly CMIP model output. 
    This code converts them all to numpy.datetime64 type of datetimes. It also utilizes
    builtin datetime function and pandas datatime. This is to deal with the various types of 
    datetime objects that appear in the CMIP output files. 
    This code lines the datetime indices to the correct years for their respective experiement,
    and cuts off projections at the year 2100.
    
    da_ (xr.data_set): the xarray dataset loaded from the cmip model data
    gen (str): 'CMIP5' or 'CMIP6'
    exp (str): the cmip experiment name
    """

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

##########################################################################################################################
##########################################################################################################################
##########################################################################################################################
