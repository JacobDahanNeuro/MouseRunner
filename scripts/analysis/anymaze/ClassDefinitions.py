# -*- coding: utf-8 -*-

import os
import pdb
import sys
import glob
import time
import pickle
import inspect
import datetime
import functools
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
from itertools import cycle
import matplotlib.pyplot as plt
from operator import attrgetter
from operator import itemgetter
import HelperFunctions as Helper
from collections import namedtuple
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
scripts_dir = os.path.dirname(os.path.dirname(os.getcwd()))
sys.path.append(os.path.join(scripts_dir,'behavior'))
from MouseRunner import Mouse
from statannotations.Annotator import Annotator as annot

class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return super().find_class(__name__, name)
        except AttributeError:
            return super().find_class(module, name)

class MappedMethod:
    """
    This class maps a single method to all elements of an iterable, with both
    *args and **kwargs compatibility.
    :param elements: Elements onto which the method should be applied.
    :type elements: iterable
    :param ret: Determines whether output of method called is returned.
    :type ret: bool
    :param *args: Any args applicabale to the method (will be applied uniformly to all elements).
    :type *args: any (method-dependent)
    :param **kwargs: Any kwargs applicable to the method (will be applied unformly to all elements).
    :type **kwargs: any (method-dependent)
    """
    def __init__(self, elements, ret=False):
        self.elements = elements
        self.ret      = ret
    def __getattr__(self, attr):
        def apply_to_all(*args, **kwargs):
            if self.ret:
                ret = []
                for obj in self.elements:
                    ret.append(getattr(obj, attr)(*args, **kwargs))
                return ret
            else:
                for obj in self.elements:
                    getattr(obj, attr)(*args, **kwargs)
        return apply_to_all


class Container:
    def __init__(self,ContainerType,**kwargs):
        self.ContainerType=ContainerType
        self.Set(**kwargs) 
    def Set(self,**kwargs):
        self.__dict__.update(kwargs) 
    def SetAttr(self,label,val):
        self.__dict__[label] = val
    def display(self,level=0):
        for attribute in self.__dict__.keys():
            if not attribute.startswith('__'):
                value = getattr(self, attribute)
                if not callable(value):
                    if not isinstance(value,Container):
                        if not isinstance(value,pd.DataFrame) and not isinstance(value,pd.Series):
                            print('\t'*level + attribute, ':', value)
                        else:
                            print('\t'*level + attribute, ':', 'Pandas DataFrame or Series')
                    else:
                        new_level = level + 1
                        print('\t'*level + attribute, ':')
                        value.display(new_level)
        return ""
    def __repr__(self):
        return "{} Container".format(self.ContainerType)
    def __str__(self):
        return self.display()
    def __iter__(self):
        for attribute in self.__dict__.keys():
            if not attribute.startswith('__'):
                value = getattr(self,attribute)
                if isinstance(value,Container):
                    yield value.name

class Analysis(Container):
    def __init__(self,input_data,dtypes):
        self.Files  = Container('Files')
        self.Data   = Container('Data')
        self.Input  = input_data
        self.dtypes = dtypes
        self.Mouse  = self.Input.mouse_id
        self.Day    = self.Input.session_day
        self.organize_files()
        self.analyze_anymaze()
        
    def organize_files(self):
        print('Reading files for Mouse {}...'.format(self.Mouse))
        for dtype in self.dtypes:
            setattr(self.Files,dtype,self.Input[dtype])
            self.read_file(dtype)
            
    def read_file(self,dtype):
        print('\tReading file of type: {}...'.format(dtype))
        if dtype in {'genotype', 'virus', 'special'}:
            data = ''.join(open(self.Input[dtype],'r').readlines())
            setattr(self.Data,dtype,data)
        elif dtype in {'parameters'}:
            SessionParameters = Container('SessionParameters')
            for (param_id,param) in pd.read_pickle(self.Input[dtype]).items():
                setattr(SessionParameters,param_id,param)
            setattr(self,'SessionParameters',SessionParameters)
        elif dtype in {'parameters_constant'}:
            MouseParameters = Container('MouseParameters')
            params = CustomUnpickler(open(self.Input[dtype],'rb')).load()
            for param_id in params.__dict__.keys():
                if not param_id.startswith('__'):
                    param = getattr(params, param_id)
                    setattr(MouseParameters,param_id,param)
            setattr(self,'MouseParameters',MouseParameters)
        elif dtype in {'session'}:
            data = pd.read_csv(self.Input[dtype],\
                               header=None).transpose().set_axis(['cue',\
                              'is_shock','time'],axis=1)
            setattr(self.Data,'timing',data)
        elif dtype in {'anymaze'}:
            data = pd.read_csv(self.Input[dtype]).set_axis(['time','freezing','not_freezing'],axis=1)
            setattr(self.Data,'anymaze',data)
      
    def analyze_anymaze(self):
        if {'anymaze','session'}.issubset(self.dtypes):
            self.align_time()
            self.analyze_freezing()
      
    def align_time(self):
        print('Aligning time series for Mouse {}...'.format(self.Mouse))
        self.pad_time()
        self.set_time()
        self.interpolate_time()
        self.label_trials()
                                 
    def pad_time(self):
        self.Data.timing.loc[-1]=[np.nan,np.nan,0.000]
        self.Data.timing.loc[len(self.Data.timing)]=[np.nan,np.nan,\
                                                     self.Data.timing.time.max()+self.SessionParameters.avg_isi]
        self.Data.timing.sort_index(inplace=True)
        self.Data.timing.reset_index(drop=True,inplace=True)
            
    def set_time(self):
        self.Data.timing.time  = self.Data.timing.time.apply(lambda x:time.strftime("%H:%M:%S.{}".format(\
                                                            ("%.3f" % x).split('.')[1]),time.gmtime(x)))
        self.Data.timing.time  = pd.to_datetime(self.Data.timing.time)
        self.Data.anymaze.loc[len(self.Data.anymaze)]=[self.Data.timing.time.max(),\
                                                       self.Data.anymaze.freezing.iloc[-1],\
                                                       self.Data.anymaze.not_freezing.iloc[-1]]
        self.Data.anymaze.time = pd.to_datetime(self.Data.anymaze.time)
        self.Data.anymaze      = self.Data.anymaze[self.Data.anymaze.time <= self.Data.timing.time.max()].reset_index(level=0,inplace=False)
        self.Data.timing.set_index(self.Data.timing.time,inplace=True)
        self.Data.anymaze.set_index(self.Data.anymaze.time,inplace=True)
        
    def interpolate_time(self):
        print('\tInterpolating time series...')
        status_ids                     = {'pre':-1,'cue':0,'post':1}
        nans_to_fill                   = int(self.SessionParameters.cs_duration / 0.001 - 1)
        epoch_size                     = int((self.SessionParameters.cs_duration / 2) / 0.001)
        self.Data.timing               = self.Data.timing.resample('L').asfreq().drop('time',1)
        self.Data.anymaze              = self.Data.anymaze.resample('L').pad().drop('time',1).drop('index',1)
        self.Data.timing.cue           = self.Data.timing.cue.interpolate(method='pad',limit=nans_to_fill,inplace=False)
        self.Data.timing.is_shock      = self.Data.timing.is_shock.interpolate(method='pad',limit=nans_to_fill,inplace=False)
        cue_ids                        = self.Data.timing.cue.copy().values
        is_shock                       = self.Data.timing.is_shock.copy().values
        cue_status                     = np.full_like(self.Data.timing.cue,np.nan,dtype=np.double)
        cue_idxs                       = np.array(list(itertools.chain(*np.where(self.Data.timing.cue.notnull()))))
        pre_cue_idxs                   = np.setdiff1d((cue_idxs-epoch_size),cue_idxs)
        post_cue_idxs                  = np.setdiff1d((cue_idxs+epoch_size),cue_idxs)
        status_idxs                    = (pre_cue_idxs,cue_idxs,post_cue_idxs)
        for ((status,status_id),status_idx) in zip(status_ids.items(),status_idxs):
            if status == 'pre':
                trialized_idxs         = Helper.groupby_consecutive(status_idx)
                trialized_start_idxs   = np.array([np.max(idxs) + 1 for idxs in trialized_idxs])
                trial_ids              = np.repeat(self.Data.timing.iloc[trialized_start_idxs].cue.values,epoch_size)
                trial_shocks           = np.repeat(self.Data.timing.iloc[trialized_start_idxs].is_shock.values,epoch_size)
            if status != 'cue':
                cue_ids[status_idx]    = trial_ids
                is_shock[status_idx]   = trial_shocks
            cue_status[status_idx]     = status_id
        self.Data.timing['cue']        = cue_ids        
        self.Data.timing['is_shock']   = is_shock        
        self.Data.timing['cue_status'] = cue_status
        self.Data.DataFrame            = pd.merge(self.Data.timing,self.Data.anymaze,on='time')
             
    def label_trials(self):
        trial_labels                 = np.full_like(self.Data.DataFrame.cue,np.nan,dtype=np.double)
        trial_idxs                   = np.array(list(itertools.chain(*np.where(self.Data.DataFrame.cue.notnull()))))
        trialized_idxs               = Helper.groupby_consecutive(trial_idxs)
        for (trial,idxs) in enumerate(trialized_idxs,1):
            trial_labels[idxs]       = np.repeat(trial,np.size(idxs)) 
        self.Data.DataFrame['trial'] = trial_labels
        self.Data.DataFrame          = self.Data.DataFrame.reindex(columns=['trial','cue','cue_status',\
                                                                            'is_shock','freezing','not_freezing'])
        
    def analyze_freezing(self):
        print('Analyzing freezing behavior for Mouse {}...'.format(self.Mouse))
        self.Data.Freezing = Container('Freezing')
        self.calculate_baseline()
        self.calculate_freezing()
    
    def calculate_baseline(self):
        print('\tCalculating baseline freezing...')
        baseline_end                  = pd.to_datetime(time.strftime("%H:%M:%S",\
                                                       time.gmtime(self.SessionParameters.baseline-(self.SessionParameters.cs_duration/2))))
        self.Data.Freezing.BaselineDF = self.Data.DataFrame[self.Data.DataFrame.index < baseline_end]
        self.Data.Freezing.Baseline   = self.Data.Freezing.BaselineDF.freezing.mean()
        
    def calculate_freezing(self):
        print('\tCalculating trialized freezing...')
        self.Data.Freezing.Trialized = self.Data.DataFrame.groupby(['cue','trial','cue_status'],\
                                                                   as_index=False).mean().sort_values(['trial',\
                                                                   'cue_status']).reset_index(drop=True,inplace=False)
        self.Data.Freezing.Mean      = pd.DataFrame(self.Data.Freezing.Trialized.groupby(['cue','cue_status']).freezing.mean()).reset_index([0,1])
        self.store_data()
        
    def store_data(self):
        series_data   = [self.Mouse,self.Day,self.MouseParameters.mouse_type,\
                         self.MouseParameters.cs1,self.MouseParameters.cs2,\
                         self.Data.Freezing.Baseline,self.Data.Freezing.Trialized,\
                         self.Data.Freezing.Mean]
        series_labels = ['mouse_id','session_day','mouse_type','cs1','cs2',\
                         'baseline_freezing','trialized_freezing','mean_freezing']
        if 'genotype' in self.dtypes:
            series_data.extend([self.Data.genotype])
            series_labels.extend(['genotype'])
        if 'virus' in self.dtypes:
            series_data.extend([self.Data.virus])
            series_labels.extend(['virus'])
        if 'special' in self.dtypes:
            series_data.extend([self.Data.special])
            series_labels.extend(['special'])
        self.Data.Freezing.All = pd.Series(data=series_data,index=series_labels)
    
    def __repr__(self):
        return "Analysis Struct (Mouse ID {} - {})".format(self.Mouse,self.Day)
    
class AnalysisContainer(Container):
    def __init__(self,MatchedFiles,data_ids,data_types,data_constants,\
                 similar_group_ids,similar_groups,group_order):
        self.ContainerType            = 'Analysis'
        self.AllData                  = Container('AllData')
        self.AllData.MatchedFiles     = MatchedFiles
        self.Parameters               = Container('Parameters')
        self.Parameters.Mice          = self.AllData.MatchedFiles.mouse_id.unique()
        self.Parameters.nMice         = len(self.Parameters.Mice)
        self.Parameters.Days          = self.AllData.MatchedFiles.session_day.unique()
        self.Parameters.nDays         = len(self.Parameters.Days)
        self.Parameters.DataIDs       = data_ids
        self.Parameters.nDataIDs      = len(self.Parameters.DataIDs)
        self.Parameters.dtypes        = sorted(set(data_types) | set(data_constants))
        self.Parameters.ndtypes       = len(self.Parameters.dtypes)
        self.Parameters.Files         = self.AllData.MatchedFiles[self.Parameters.dtypes]
        self.Parameters.SimilarIDs    = similar_group_ids
        self.Parameters.SimilarGroups = similar_groups if similar_groups else None
        self.Parameters.GroupBy       = True if group_order else False
        self.Parameters.GroupOrder    = group_order if group_order else None
        self.init_analysis()

    def init_analysis(self):
        self.AllData.Analysis = [Analysis(Data,self.Parameters.dtypes) for (idx,Data)\
                                in self.AllData.MatchedFiles.iterrows()]
        if {'anymaze','session'}.issubset(self.Parameters.dtypes):
            self.AllData.Freezing = pd.DataFrame(list(map(attrgetter('Data.Freezing.All'),self.AllData.Analysis)))
            if self.Parameters.GroupBy:
                setattr(self.AllData,'GroupedData',Container('GroupedData'))
                if self.Parameters.SimilarIDs:
                    self.custom_group()
                self.groupby()
        
    def custom_group(self):
        for (dtype,groups) in zip(self.Parameters.SimilarIDs,self.Parameters.SimilarGroups):
            replacement_dict = {group: ', '.join(groups) for group in groups}
            self.AllData.Freezing[dtype] = self.AllData.Freezing[dtype].replace(replacement_dict)
            
    def groupby(self):
        for (group_number,(name,group)) in enumerate(self.AllData.Freezing.groupby(self.Parameters.GroupOrder),1):
            group_name = 'group_{}'.format(group_number)
            setattr(self.AllData.GroupedData,group_name,Container('Group'))
            setattr(getattr(self.AllData.GroupedData,group_name),'name',group_name)
            setattr(getattr(self.AllData.GroupedData,group_name),'GroupParameters',Container('GroupParameters'))
            setattr(getattr(self.AllData.GroupedData,group_name),'DataFrame',group)
            self.analyze_group(group_name)
            for group_key,value in zip(self.Parameters.GroupOrder,name):
                setattr(getattr(getattr(self.AllData.GroupedData,group_name),'GroupParameters'),group_key,value)
            
    def analyze_group(self,group_name):
        DataFrame         = getattr(getattr(self.AllData.GroupedData,group_name),'DataFrame')
        MeanData          = DataFrame.mean_freezing.values
        Means             = pd.concat(MeanData).groupby(['cue','cue_status']).mean()
        Means['sem']      = pd.concat(MeanData).groupby(['cue','cue_status']).sem().values
        Means['single']   = pd.concat(MeanData).groupby(['cue','cue_status']).freezing.apply(list).values
        setattr(getattr(self.AllData.GroupedData,group_name),'Means',Means)
        
    def filter_mice(self,which='Freezing',by='mouse_id',criteria=[''],regroup=True):
        attr = getattr(self.AllData,which)
        if by in {'mouse_id'}:
            attr_filtered = attr[~attr.mouse_id.isin(criteria)]
            setattr(self.AllData,which,attr_filtered)
        elif by in {'baseline_freezing'}:
            attr_filtered = attr[attr.baseline_freezing <= float(criteria.pop())]
            setattr(self.AllData,which,attr_filtered)
        if regroup:
            if self.Parameters.GroupBy:
                setattr(self.AllData,'GroupedData',Container('GroupedData'))
                self.groupby()
            
    def filter_trials(self,which='Freezing',by='trialized_freezing',criteria=0.5,regroup=True):
        attr = getattr(self.AllData,which)
        if by in {'trialized_freezing'}:
            for df in attr.trialized_freezing.values:
                pre_freezing       = df.groupby(['cue_status']).get_group(-1)
                thresholded_trials = pre_freezing[pre_freezing.freezing < criteria].trial.values
                df.drop(df[~df.trial.isin(thresholded_trials)].index,inplace=True)
        setattr(self.AllData,which,attr)
        if regroup:
            if self.Parameters.GroupBy:
                setattr(self.AllData,'GroupedData',Container('GroupedData'))
                self.groupby()
            
    def filter_groups(self,filter_key,which='Freezing',regroup=True):
        for (group_number,(name,group)) in enumerate(getattr(self.AllData,which).groupby(self.Parameters.GroupOrder),1):
            group_name  = 'group_{}'.format(group_number)
            is_filtered = True
            for group_key,value in zip(self.Parameters.GroupOrder,name):
                if filter_key in value:
                    is_filtered = False
            if is_filtered:
                delattr(self.AllData.GroupedData,group_name)
        if regroup:
            if self.Parameters.GroupBy:
                setattr(self.AllData,'GroupedData',Container('GroupedData'))
                self.groupby()

    def normalize(self,which='Freezing',by='pre',scale=True,scale_by=0,scale_to=1,regroup=True):
        attr = getattr(self.AllData,which).copy()
        if by in {'pre'}:
            for df in attr.trialized_freezing.values:
                pre_freezing                               = df.groupby(['cue_status']).get_group(-1).freezing.values
                df['normalized_freezing']                  = np.full_like(df.freezing,np.nan,dtype=np.double)
                df.normalized_freezing[df.cue_status == 0] = df.groupby(['cue_status']).get_group(0).freezing.values - pre_freezing
                df.normalized_freezing[df.cue_status == 1] = df.groupby(['cue_status']).get_group(1).freezing.values - pre_freezing
                df.normalized_freezing[df.cue_status <  0] = pre_freezing - pre_freezing
                if scale:
                    df = self.scale(df,scale_by,scale_to)
        elif by in {'baseline'}:
            for (df,baseline_freezing) in zip(attr.trialized_freezing.values,attr.baseline_freezing.values):
                df['normalized_freezing']  = df.freezing.sub(baseline_freezing)
                if scale:
                    df = self.scale(df,scale_by,scale_to)
        elif by is None:
            for df in attr.trialized_freezing.values:
                df['normalized_freezing'] = df.freezing
                if scale:
                    df = self.scale(df,scale_by,scale_to)
        setattr(self.AllData,which,attr)
        if regroup:
            if self.Parameters.GroupBy:
                setattr(self.AllData,'GroupedData',Container('GroupedData'))
                self.groupby()

    def scale(self,df,scale_by=0,scale_to=1):
        if scale_to == 1:
            scale = df.groupby(['cue','cue_status']).get_group((scale_by,0)).normalized_freezing.mean()
            df.normalized_freezing = df.normalized_freezing.div(scale)
        return df
        
    def plotify(self,which='trialized_freezing',epoch='cue',stats_display='star',\
                palette=['magenta','mediumturquoise','palegreen','cornflowerblue','coral'],\
                plots=['lineplot','barplot'],stats_tests=['anova','t-test_ind'],\
                comparitor_traits={'virus':'AAV5-CamKII-eGFP, AAV5-CamKII-eYFP','special':'None','session_day':'day3'},\
                ignore_traits={'session_day':'day4','mouse_type':'Demonstrator'},\
                swarm=True,normalized=True,alpha=0.05):
        epoch2epoch = {'pre':-1,'cue':0,'post':1}
        colors      = cycle(palette[1:])
        ConcatMeans = dict()
        for group in self.AllData.GroupedData:
            traits          = getattr(self.AllData.GroupedData,group).GroupParameters
            matched_traits  = [trait for (trait,value) in traits.__dict__.items() if (trait,value) in comparitor_traits.items()]
            mismatch_traits = [trait for (trait,value) in traits.__dict__.items() if (trait,value) in ignore_traits.items()]
            is_comparitor   = True if len(matched_traits) == len(comparitor_traits.keys()) else False
            if mismatch_traits:
                continue
            GroupedData     = getattr(getattr(self.AllData.GroupedData,group),'DataFrame')
            GroupMice       = GroupedData.mouse_id.values
            GroupedData     = getattr(GroupedData,which)
            ConcatData      = pd.DataFrame(columns=['cue','normalized_freezing'])
            if normalized:
                for (mouse,df) in zip(GroupMice,GroupedData):
                    df                 = df.groupby(['cue_status']).get_group(epoch2epoch[epoch])
                    Data               = df.groupby(['cue']).normalized_freezing.mean().reset_index(drop=False)
                    dtype              = 'normalized_freezing'
                    Data['mouse']      = np.repeat(mouse,len(Data))
                    Data['comparitor'] = np.repeat(is_comparitor,len(Data))
                    ConcatData         = pd.concat([ConcatData,Data])
            else:
                for (mouse,df) in zip(GroupMice,GroupedData):
                    df         = df.groupby(['cue_status']).get_group(epoch2epoch[epoch])
                    Data       = df.groupby(['cue']).freezing.mean().reset_index(drop=False)
                    dtype      = 'freezing'
                    Data['comparitor'] = np.repeat(is_comparitor,len(Data))
                    ConcatData = pd.concat([ConcatData,Data])
            fig,ax=plt.subplots()
            title=[]
            for attribute in getattr(self.AllData.GroupedData,group).GroupParameters.__dict__.keys():
                if not attribute.startswith('__'):
                    value = getattr(getattr(self.AllData.GroupedData,group).GroupParameters,attribute)
                    title.append((attribute,value))
            title=', '.join(["{} : {}".format(attr,value) for (attr,value) in title])
            if 'lineplot' in plots:
                stats_test = stats_tests[plots.index('lineplot')]
                sns.lineplot(x='cue',y='normalized_freezing',hue='mouse',marker='o',\
                             data=ConcatData,palette=[next(colors) for i in range(len(GroupMice))]).set(title=title)
                if 'anova' in stats_test:
                    anova,p_values,pairs = Helper.stat_compare(ConcatData,dtype=dtype,by=epoch,alpha=alpha)
                    if anova.pvalue < alpha:
                        Helper.annotate(ax,df=ConcatData,p_values=p_values,pairs=pairs,dtype=dtype,fmt=stats_display)
                plt.savefig(r'J:\Jacob Dahan\MouseRunner\{}.svg'.format(group),bbox_inches='tight')
            ConcatMean = ConcatData.groupby(['cue'],as_index=False).agg({'normalized_freezing':'mean','comparitor':'first'})
            ConcatMeans[group] = ConcatMean.assign(vals=ConcatData.groupby(['cue'],as_index=False).agg(list).normalized_freezing)
        if 'barplot' in plots:
            comparitor_means = pd.concat([means.assign(group=np.repeat(group,len(means)))\
                                          for (group,means) in ConcatMeans.items()\
                                          if means.comparitor.unique()])
            for (group,means) in ConcatMeans.items():
                if means.comparitor.unique():
                    continue
                else:
                    if len(means) == len(comparitor_means):
                        BarData = pd.concat([comparitor_means,means.assign(group=np.repeat(group,len(means)))])
                        SwarmData = BarData.set_index(['cue','normalized_freezing','group']).vals.apply(pd.Series).stack().rename('vals').reset_index()
                        pairs = Helper.generate_pairs(BarData,x='cue',hue='group')
                        fig,ax = plt.subplots()
                        c1,c2 = palette[0], next(colors)
                        sns.barplot(x='cue',y='normalized_freezing',hue='group',data=BarData,ci=68,palette=[c1,c2],ax=ax)
                        sns.swarmplot(x='cue',y='vals',hue='group',data=SwarmData,dodge=True,palette=[c1,c2],edgecolor='black',linewidth=1,ax=ax)
                        annotator = annot(ax,pairs,plot='barplot',x='cue',y='vals',hue='group',data=SwarmData)
                        annotator.configure(test='t-test_ind',correction_format="replace").apply_test().annotate()
                        plt.savefig(r'J:\Jacob Dahan\MouseRunner\bar-{}.svg'.format('-'.join(BarData.group.unique())),bbox_inches='tight')

    def __repr__(self):
        return "Analysis Container"
            
