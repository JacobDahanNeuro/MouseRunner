# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 15:51:09 2021

@author: jacobdahan
"""

### TODO
import pdb
import os
import re
import time
import glob
import operator
import itertools
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns 
from scipy import stats
import PySimpleGUI as sg
from pathlib import Path
import matplotlib.pyplot as plt
from ordered_set import OrderedSet
from collections import namedtuple
from statannotations.Annotator import Annotator
from statsmodels.stats.multicomp import pairwise_tukeyhsd


class ComparisonBar(namedtuple('ComparisonBar', 'start stop height')):
    __slots__ = ()
    @property
    def range(self):
        return abs(self.stop - self.start)

class Statistics(namedtuple('Statistics','mousetype anova_f anova_p tukey')):
    __slots__ = ()
    
class File(namedtuple('File','filepath mouse_id')):
    __slots__ = ()

class InterpolatedAnalysis(namedtuple('InterpolatedAnalysis','mousetype data')):
    __slots__ = ()
    @property
    def freezing(self):
        return (self.data.loc['Pre'],\
                self.data.loc['Cue'],\
                self.data.loc['Post'])

def gen_stats(df,mousetype):
    statistics    = dict()
    for cue_status in set(df.index):
        anova        = stats.f_oneway(*[df.loc[cue_status,cue_type].values for cue_type in df.columns])
        df_melt      = pd.melt(df.loc[cue_status].reset_index(),id_vars=['CueStatus'],value_vars=df.columns)
        tukeyhsd     = pairwise_tukeyhsd(endog=df_melt.value,groups=df_melt.variable,alpha=0.05)
        tukeyhsd_df  = pd.DataFrame(data=tukeyhsd._results_table.data[1:],\
                                    columns=tukeyhsd._results_table.data[0]) 
        statistics[cue_status] = Statistics(mousetype,anova.statistic,anova.pvalue,tukeyhsd_df) 
    return statistics
       
def p2text(p):
    if p <= 0.001:
        text = '***'
    elif p < 0.01:
        text = '**'
    elif p < 0.05:
        text = '*'
    else:
        text = 'ns (p={:0.2f})'.format(p)
    return text
    
def annotate(ax,df,statistics,offset=5):
    #bar_locs = np.arange(len(df.columns) * len(set(df.reset_index().CueStatus)))
    #group    = 0
    # for cue_status in OrderedSet(df.reset_index().CueStatus):
    #     locations    = list(itertools.combinations(bar_locs[(0 + group * 3):(3 + group * 3)],2))
    #     bar_pairs    = list(itertools.combinations(df.columns, 2))
    #     compbars     = list()
    bar_pairs = np.repeat(list(itertools.combinations(OrderedSet(df.CueStatus.values),2)),repeats=len(df.columns))
    pvalues=list()
    bar_pairs = [[('Pre','CS-'),('Pre','CS+1')],\
                 [('Cue','CS-'),('Cue','CS+1')],\
                 [('Post','CS-'),('Post','CS+1')],\
                 [('Pre','CS-'),('Pre','CS+2')],\
                 [('Cue','CS-'),('Cue','CS+2')],\
                 [('Post','CS-'),('Post','CS+2')],\
                 [('Pre','CS+1'),('Pre','CS+2')],\
                 [('Cue','CS+1'),('Cue','CS+2')],\
                 [('Post','CS+1'),('Post','CS+2')]]
    for pair in bar_pairs:
        key = pair[0][0]
        start_cue = pair[0][1]
        stop_cue   = pair[1][1]
        tukey = statistics[key].tukey
        p = tukey[(((tukey.group1 == start_cue) & (tukey.group2 == stop_cue)) | \
            ((tukey.group2 == start_cue) & (tukey.group1 == stop_cue)))]['p-adj'].values[0]
        pvalues.append(p)
    annotator = Annotator(ax,bar_pairs,data=df,x='CueStatus',y='Value',hue='CueType',\
                          hue_order=['CS-','CS+1','CS+2'],order=['Pre','Cue','Post'])
    annotator.configure(test=None, text_format='simple', loc='inside')
    annotator.set_pvalues_and_annotate(pvalues)

def open_window(title,layout,w,h):
    sg.theme('BlueMono')  
    window = sg.Window(title,layout,finalize=True,size=(w,h), 
                       element_justification='center')
    return window

def match_csvs(cue_csvs, anymaze_csvs):
    cue_timing_files = list(map(File,cue_csvs,list(map(lambda x: os.path.basename(os.path.dirname(x)),cue_csvs))))
    anymaze_files    = list(map(File,anymaze_csvs,list(map(lambda x: re.split('_',Path(x).stem)[-1],anymaze_csvs))))
    ca, c_fis, a_fis = np.intersect1d(list(map(operator.attrgetter('mouse_id'),cue_timing_files)),\
                                      list(map(operator.attrgetter('mouse_id'),anymaze_files)),\
                                      return_indices=True)
    return list(map(operator.attrgetter('filepath'),list(operator.itemgetter(*c_fis)(cue_timing_files)))),\
           list(map(operator.attrgetter('filepath'),list(operator.itemgetter(*a_fis)(anymaze_files))))

def grab_files():
    w, h      = sg.Window.get_screen_size()
    load_title= 'Loading'
    load_text = [[sg.Text('Loading file selector...')]]
    load      = open_window(load_title,load_text,int(w/3),int(h/3))
    w, h      = int(w/2.5), int(h/8)
    title     = 'Select Cue Timing and AnyMaze Analysis Files'
    layout    = [[sg.Text('Cue Timing Directory:',size=(18,1),justification='right'),
                  sg.In(default_text='{}'.format(os.getcwd()),\
                        enable_events=True,key='CUEFILES'),
                  sg.FolderBrowse('Select Directory',target='CUEFILES')],
                 [sg.Text('AnyMaze Analysis Directory:',size=(18,1),justification='right'),
                  sg.In(default_text='{}'.format(os.getcwd()),\
                        enable_events=True,key='FREEZEFILES'),
                  sg.FolderBrowse('Select Directory',target='FREEZEFILES')],
                 [sg.OK()]]
    load.close()
    main = open_window(title,layout,w,h)
    main.BringToFront()
    while True:
        event, values = main.read()
        if event in (sg.WINDOW_CLOSED, 'Exit'):
            main.close()
            return None, None
        if event == 'OK':
            cue_csvs               = glob.glob(values['CUEFILES']+"/**/day3-timing*{}".format('.csv'),recursive=True)
            cue_csvs               = sorted(cue_csvs, key = lambda x: re.split('_',os.path.basename(os.path.dirname(x)))[-1])
            anymaze_csvs           = glob.glob(values['FREEZEFILES']+"/**/*{}".format('.csv'),recursive=True)
            anymaze_csvs           = sorted(anymaze_csvs, key = lambda x: re.split('_',Path(x).stem)[-1])
            cue_csvs, anymaze_csvs = match_csvs(cue_csvs, anymaze_csvs)
            return cue_csvs, anymaze_csvs

def slice_df_by_cue(df,cue_id,cue_duration=30):
    epoch_size    = int((cue_duration / 2) / 0.001)
    cue_idxs      = np.where(df.Cue==cue_id)[0]
    pre_cue_idxs  = np.setdiff1d((cue_idxs - epoch_size), cue_idxs)
    post_cue_idxs = np.setdiff1d((cue_idxs + epoch_size), cue_idxs)
    slice_idx     = np.concatenate((pre_cue_idxs,cue_idxs,post_cue_idxs))
    cue_df        = df.iloc[slice_idx].sort_index(inplace=False)
    return cue_df

def freezing_by_cue_status(df):
    pre_cue_freezing  = df[df.CueStatus==-1].Freezing.mean()
    cue_freezing      = df[df.CueStatus==0].Freezing.mean()
    post_cue_freezing = df[df.CueStatus==1].Freezing.mean()
    return {'Pre':pre_cue_freezing,
            'Cue':cue_freezing,
            'Post':post_cue_freezing}

def interpolate_data(cue_csv,anymaze_csv):
    cue_df                    = pd.read_csv(cue_csv,header=None).transpose()
    cue_df.columns            = ['Cue','Time']
    max_time                  = cue_df.Time.max() + 60
    cue_df                    = pd.concat([pd.DataFrame({'Cue':np.nan,'Time':0.000},index=[0]), 
                                           cue_df,
                                           pd.DataFrame({'Cue':np.nan,'Time':max_time},
                                                        index=[-1])]).reset_index(drop = True)
    cue_df.Time               = cue_df.Time.apply(lambda x:time.strftime('%H:%M:%S.{}'.format(("%.3f" % x).split('.')[1]), time.gmtime(x)))
    cue_df.Time               = pd.to_datetime(cue_df.Time)
    cue_df                    = cue_df.set_index(cue_df.Time, inplace=False)
    interp_cue_df             = cue_df.resample('L').asfreq().Cue.rename_axis('Time').reset_index(level=0, inplace=False)
    cue_duration              = 30
    nans_to_fill              = int(cue_duration / 0.001 - 1)
    epoch_size                = int((cue_duration / 2) / 0.001)
    interp_cue_df.Cue         = interp_cue_df.Cue.interpolate(method='pad',limit=nans_to_fill)
    cue_status                = np.full_like(interp_cue_df.Cue, np.nan, dtype=np.double)
    cue_idxs                  = np.where(interp_cue_df.Cue.notnull())[0]
    pre_cue_idxs              = np.setdiff1d((cue_idxs - epoch_size), cue_idxs)
    post_cue_idxs             = np.setdiff1d((cue_idxs + epoch_size), cue_idxs)
    cue_status[cue_idxs]      = 0
    cue_status[pre_cue_idxs]  = -1
    cue_status[post_cue_idxs] = 1
    interp_cue_df['CueStatus']= cue_status
    anymaze_df                = pd.read_csv(anymaze_csv)
    anymaze_df                = pd.concat([anymaze_df,
                                           pd.DataFrame({'Time':cue_df.Time.max(),
                                                         'Freezing':anymaze_df.Freezing.iloc[-1],
                                                         'Not freezing':anymaze_df['Not freezing'].iloc[-1]},
                                                          index=[-1])]).reset_index(drop = True)
    anymaze_df.Time           = pd.to_datetime(anymaze_df.Time)
    anymaze_df                = anymaze_df[(anymaze_df.Time <= cue_df.Time.max())].reset_index(level=0, inplace=False)
    anymaze_df                = anymaze_df.set_index(anymaze_df.Time, inplace=False)
    interp_anymaze_df         = anymaze_df.resample('L').pad().Freezing.rename_axis('Time').reset_index(level=0, inplace=False)
    interp_concat_df          = pd.merge(interp_cue_df, interp_anymaze_df, on='Time')
    interp_concat_df          = interp_concat_df.drop(interp_concat_df.tail(1).index,inplace=False)
    cs_minus_df               = slice_df_by_cue(interp_concat_df,0)
    cs_one_df                 = slice_df_by_cue(interp_concat_df,1)
    cs_two_df                 = slice_df_by_cue(interp_concat_df,2)
    cs_minus_freezing         = freezing_by_cue_status(cs_minus_df)
    cs_one_freezing           = freezing_by_cue_status(cs_one_df)
    cs_two_freezing           = freezing_by_cue_status(cs_two_df)
    mousetype                 = 'Experienced' if 'Experienced' in cue_csv else 'Naive'
    return InterpolatedAnalysis(mousetype,
                                pd.DataFrame({'CS-':cs_minus_freezing,'CS+1':cs_one_freezing,'CS+2':cs_two_freezing}))

def plot_and_store(interpolated_data):
    palette = {'CS-'  : 'black',
               'CS+1' : 'fuchsia',
               'CS+2' : 'gold'}
    try:
        experienced_data            = list(map(operator.attrgetter('data'),\
                                           filter(lambda x: x.mousetype == 'Experienced',\
                                           interpolated_data)))
        experienced_data            = pd.concat(experienced_data)
        experienced_data.index.name = 'CueStatus'
        experienced_data            = experienced_data.multiply(100)
        experienced_stats           = gen_stats(experienced_data,'Experienced')
        experienced_melt            = pd.melt(experienced_data.reset_index(),id_vars=['CueStatus'],\
                                              value_vars=experienced_data.columns)
        experienced_melt.columns    = ['CueStatus','CueType','Value']
        fig, ax = plt.subplots()
        sns.barplot(x='CueStatus',y='Value',hue='CueType',data=experienced_melt,\
                    hue_order=['CS-','CS+1','CS+2'],order=['Pre','Cue','Post'],\
                    palette=palette,ax=ax,ci=68)
        sns.swarmplot(x='CueStatus',y='Value',hue='CueType',data=experienced_melt,\
                    hue_order=['CS-','CS+1','CS+2'],order=['Pre','Cue','Post'],\
                    color='black',edgecolor='black',dodge=True)
        annotate(ax,experienced_melt,experienced_stats)
        ax.set(xlabel='Cue Status',ylabel='% Freezing')
        ax.set(ylim=(0,140))
        ax.set(yticks=[0,20,40,60,80,100])
        plt.savefig('./experienced.svg')
    except:
        experienced_data = None
    try:
        naive_data            = list(map(operator.attrgetter('data'),\
                                     filter(lambda x: x.mousetype == 'Naive',\
                                            interpolated_data)))
        naive_data            = pd.concat(naive_data)
        naive_data.index.name = 'CueStatus'
        naive_data            = naive_data.multiply(100)
        naive_stats           = gen_stats(naive_data,'Experienced')
        naive_melt            = pd.melt(naive_data.reset_index(),id_vars=['CueStatus'],\
                                              value_vars=naive_data.columns)
        naive_melt.columns    = ['CueStatus','CueType','Value']
        fig, ax = plt.subplots()
        sns.barplot(x='CueStatus',y='Value',hue='CueType',data=naive_melt,\
                    hue_order=['CS-','CS+1','CS+2'],order=['Pre','Cue','Post'],\
                    palette=palette,ax=ax,ci=68)
        sns.swarmplot(x='CueStatus',y='Value',hue='CueType',data=naive_melt,\
                    hue_order=['CS-','CS+1','CS+2'],order=['Pre','Cue','Post'],\
                    color='black',edgecolor='black',dodge=True)
        ax.set(xlabel='Cue Status',ylabel='% Freezing')
        annotate(ax,naive_melt,naive_stats)
        ax.set(ylim=(0,140))
        ax.set(yticks=[0,20,40,60,80,100])
        plt.savefig('./naive.svg')
    except:
        naive_data = None
    

def analyze_anymaze():
    cue_csvs, anymaze_csvs = grab_files()
    interpolated_data    = list()
    for cue_csv,anymaze_csv in zip(cue_csvs,anymaze_csvs):
        interpolated_data.append(interpolate_data(cue_csv,anymaze_csv))
    plot_and_store(interpolated_data)
    