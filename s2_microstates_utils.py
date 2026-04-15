import json
import numpy as np
import mne

import matplotlib.pyplot as plt
import matplotlib

import pycrostates
import os
import glob
import matplotlib
from matplotlib.widgets import Slider
import sys
import pickle
from lempel_ziv_complexity import lempel_ziv_complexity

#%% Metamaps
def LoadMetamaps(filename=None, plot=False, n_ms=4, reorder_str=['CBAD','DCBAE','DBCFEA','DGAEFBC','FACEGHBD']):
    
    if filename is None:
        filename = os.path.join(os.path.dirname(__file__), 'matlab/metamaps_export.json')
    elif not filename.endswith('.json'):
        raise ValueError('File must be a JSON file.')
    else:
        if not os.path.exists(filename):
            raise FileNotFoundError(f'File {filename} not found.')
    
    with open(filename) as f:
        metamaps_all = json.load(f)
        metamaps_all = list(map(lambda expmap: {**expmap, 'Maps': np.array(expmap['Maps'])}, metamaps_all))

    for s, (solution, reorder_str) in enumerate(zip(metamaps_all,reorder_str)):
        solution['Order'] = np.array(list(reorder_str))
        # Reorder maps based on the 'Order' for this solution
        order_indices = np.argsort(metamaps_all[s]['Order'])#[ord(char) - ord('A') for char in order_str]  # Convert letters to indices (A -> 0, B -> 1, etc.)
        solution['Maps'] = solution['Maps'][order_indices]
        solution['Order'] = solution['Order'][order_indices]
        print(solution['Order'])


    if plot:
        # plotting all solutions 
        n_solutions,max_cluster = len(metamaps_all), metamaps_all[-1]['Maps'].shape[0]
        fig,axs = plt.subplots(n_solutions,max_cluster,figsize=(2 * n_solutions, 2 * max_cluster),subplot_kw={'frame_on':0,'xticks':[],'yticks':[]})
        plt.ion()
        vlim_ms = (min([sol['Maps'].min() for sol in metamaps_all]),max([sol['Maps'].max() for sol in metamaps_all]))
        for s, solution in enumerate(metamaps_all):
            chanlocs = solution['chanlocs']
            xyz_locs = np.array([[-d['Y'], d['X'], d['Z']] for d in chanlocs])  # Electrode locations
            labels_locs = [d['labels'] for d in chanlocs]  # Electrode labels
            
            for m, msmap in enumerate(solution['Maps']):
                im,cb = mne.viz.plot_topomap(msmap,xyz_locs[:,:-1],axes=axs[s,m], 
                                    cmap='RdBu_r',vlim=(vlim_ms))
                if m==0: axs[s,m].set_ylabel(f"{len(solution['Maps'])}-MS\nsolution",fontsize=14)
                if s==len(metamaps_all)-1: axs[s,m].set_xlabel(f"{chr(ord('A')+m)}", fontsize=14)
                
        # add cbar      
        cbar_ax = fig.add_axes([.9,.35,.02,.6])
        clb = fig.colorbar(im, cax=cbar_ax)
        clb.ax.set_title(u'V (\u03bcV)',fontsize=12) 
        # add electrode locations legend
        chnames_ax = fig.add_axes([.6,.6,.3,.3])
        mne.viz.plot_topomap(np.zeros((19,)),xyz_locs[:,:-1],names=labels_locs,axes=chnames_ax)
        for tt in plt.findobj(chnames_ax, matplotlib.text.Text):
                tt.set_fontsize(14)
        # finalize plot
        plt.suptitle('Meta-microstates [Koenig et al., EEG-Meta-Microstates... (2024, Brain Topography)]',fontsize=16)
        plt.tight_layout()
        plt.show()
        
    min_len = min([len(solution['Labels']) for solution in metamaps_all])
    max_len = max([len(solution['Labels']) for solution in metamaps_all])
    
    #TODO extract metamicrostates from 1 to 15 in Matlab
    if n_ms > max_len: # n_ms requested is more than the maximum available
        raise ValueError(f"n_ms should be less than or equal to {max_len}")
    elif n_ms < min_len: # n_ms requested is less than the minimum available
        precomputed_maps = next((solution['Maps'] for solution in metamaps_all if len(solution['Labels']) == min_len), None)
        meta_gev = next((solution['ExpVar'] for solution in metamaps_all if len(solution['Labels']) == min_len), None)
        meta_lbl = next((solution['Order'] for solution in metamaps_all if len(solution['Labels']) == min_len), None)
        # cut to n_ms
        precomputed_maps = precomputed_maps[:n_ms]
        meta_lbl = meta_lbl[:n_ms]
    else: # n_ms requested is within the directly available range
        precomputed_maps = next((solution['Maps'] for solution in metamaps_all if len(solution['Labels']) == n_ms), None)
        meta_gev = next((solution['ExpVar'] for solution in metamaps_all if len(solution['Labels']) == n_ms), None)
        meta_lbl = next((solution['Order'] for solution in metamaps_all if len(solution['Labels']) == n_ms), None)
        
    return precomputed_maps, meta_gev, meta_lbl
        
#%% Backfitting tools
def microstates_extraction(rawobj, meta_modkmeans, optEX_cfg):
    raw_ = rawobj.copy().pick('eeg')
    
    # Intermediate computation
    C = backfit_intermediate(raw_, meta_modkmeans.cluster_centers_)
    
    # Backfitting
    sequence = backfit_pycrostates(raw_, meta_modkmeans, optEX_cfg)
    
    # Compute Global Explained Variance
    gev = compute_gev(raw_, sequence, C)
    
    return {'sequence': sequence, 
            'backfit_intermediate': C,
            'gev': gev,
            'gfp': np.nanstd(raw_.get_data('eeg').T, axis=1)   
            }
    
def apply_metrics(sequence,callable_dict):
    return {name: func(sequence) 
            for name,func in callable_dict.items()}# dict of callables to evaluated on the sequence}
    
def backfit_pycrostates(rawobj, modkmeans, optEX_cfg):
    '''
    Returns the sequence of microstates for the rawobj EEG data using the modkmeans microstate maps
    '''
    
    sequence = modkmeans.predict(rawobj.pick('eeg'),
                                 **optEX_cfg['pycrostates_params']
                                ).labels
    
    return sequence

def compute_gev(rawobj, sequence, C):
    '''
    rawobj: mne.io.Raw object with the EEG data, shape (n_channels, n_samples)
    sequence: np.array with the microstate labels, shape (n_samples,)
    C: correlation between the maps and the raw data, shape (n_samples, n_maps)
    
    Returns:
    gev: list with the Global Explained Variance for each microstate, shape (n_maps,)
    '''
    gfp = np.nanstd(rawobj.get_data('eeg').T, axis=1)
    gev = []
    for k in range(C.shape[1]):
        r = (sequence==k)
        gev.append( np.sum(gfp[r]**2 * C[r,k]**2)/np.sum(gfp**2) ) 
        
    return gev

def backfit_intermediate(rawobj, maps):
    '''
    rawobj: mne.io.Raw object with the EEG data, shape (n_channels, n_samples)
    maps: np.array with the microstate maps, shape (n_maps, n_channels)
    
    Returns:
    C: np.array with the correlation between the maps and the raw data, shape (n_samples, n_maps)
    '''
    rawarr = rawobj.get_data('eeg')
    # Normalize EEG wrt channels
    rawarr -= rawarr.mean(axis=0, keepdims=True)
    std = rawarr.std(axis=0, keepdims=True)
    std[std==0] = 1 # null map as in pycrostates.cluster._base._BaseCluster._segment
    rawarr /= std
    
    # Normalize MAPS wrt channels
    maps_norm = (maps - maps.mean(axis=1, keepdims=True))/maps.std(axis=1, keepdims=True)
    
    # Compute the correlation
    C = np.abs(np.dot(maps_norm, rawarr)/rawarr.shape[0]).T
    
    return C

def PycroModKMeans(maps,meta_gev,meta_label,plot=True):
    
    info_template_path = os.path.join(os.path.dirname(__file__),'info_template.pkl')
    if not os.path.exists(info_template_path):
        matches = glob.glob(os.path.join(os.path.dirname(__file__), '**', 'info_template.pkl'), recursive=True)
        if len(matches) == 0:
            raise FileNotFoundError("info_template.pkl not found in repository subdirectories.")
        info_template_path = matches[0]
    with open(info_template_path,
              'rb') as f:
        info_template = pickle.load(f)

    meta_mod_kmeans = pycrostates.cluster.ModKMeans(n_clusters=maps.shape[0],random_state=0)

    setattr(meta_mod_kmeans,'_GEV_',meta_gev)
    setattr(meta_mod_kmeans,'_cluster_centers_', maps)
    setattr(meta_mod_kmeans,'_fitted', True)
    setattr(meta_mod_kmeans,'_ignore_polarity', True)
    setattr(meta_mod_kmeans,'_info',info_template)

    setattr(meta_mod_kmeans,'_labels_', 'nodata')
    setattr(meta_mod_kmeans,'_fitted_data','nodata')

    meta_mod_kmeans.rename_clusters(new_names=list(meta_label))
    if plot: 
        meta_mod_kmeans.plot()
        plt.show()
    
    return meta_mod_kmeans

#%% Metrics

def coverage(sequence,n_ms):
    """
    Compute coverage of a sequence of microstates for each microstate
    i.e. symbol relative frequency
    """
    return [np.mean(sequence==i) for i in range(n_ms)]
    
def transition_matrix(sequence,n_ms):
    """
    Compute transition matrix of a sequence of microstates
    """
    transition_matrix = np.zeros((n_ms, n_ms))
    for i in range(n_ms):
        for j in range(n_ms):
            from_i_to_j = np.logical_and(sequence[:-1]==i, 
                                            sequence[1:]==j)
            transition_matrix[i,j] = np.mean(from_i_to_j)
    
    # check rows sum up to 1
    rowsum = transition_matrix.sum(axis=1)
    assert np.allclose(rowsum,1), \
            f"Rows of T sum up to {rowsum}, not 1"
    
    return transition_matrix
    
def duration(sequence,n_ms,fs):
    """ Compute the mean consecutive duration of each microstate in seconds """
    from itertools import groupby
    # (ms, duration) grouped sequence (run-length)
    sequence_RLE = [(k, len(list(g))) for k, g in groupby(sequence)]
    # mean continous duration of each microstate
    durations_samp = [[d for k,d in sequence_RLE if k==i] for i in range(n_ms)]
    # replace empty lists with 0
    durations_samp = [d if d else [0] for d in durations_samp]
    # mean and conversion to seconds
    duration_s = [np.mean(durations_samp[i])/fs
                        for i in range(n_ms)]
    return duration_s

''' TEST FOR duration function
# Function to convert string to array using ord
def string_to_array(s):
    return [ord(x) - ord('A') for x in s]

# Test cases as strings
test_cases = [
    'AAAAAAAAAAAAAAAAAAA',
    'AAAAAAAAAAAABBBBBBBB',
    'ABCBCABCABCCBACBABCAACBCBCCBBBBBACBAAB',
    '',  # Empty string
    'A',  # Single character
    'AB' * 50,  # Long repetitive pattern
    'Z' * 100,  # Characters at the end of the alphabet
    'ABCDEFGHIJKLMNOPQRSTUVWXYZ',  # All unique characters
    'A' * 1000,  # Very long single character
    'ABCD' * 250,  # Long repetitive pattern with more characters
    'A' * 500 + 'B' * 500,  # Long pattern with two characters
    'A' * 499 + 'B' * 501,  # Long pattern with two characters, slightly unbalanced
    'A' * 100 + 'B' * 100 + 'C' * 100 + 'D' * 100,  # Long pattern with four characters
]

for case in test_cases:
    array_representation = string_to_array(case)
    n_ms = 3#len(np.unique(array_representation))
    duration_representation = duration(array_representation,n_ms,1)
    from itertools import groupby
    RLE = [(k, len(list(g))) for k, g in groupby(array_representation)]
    print(f">>>String: {case}\n RLE:{RLE}\n Duration: {duration_representation}")
'''
def occurrence(sequence, n_ms, fs):
    """Compute the occurrence per second of each microstate"""
    # Create a binary matrix where each row corresponds to a microstate and each column to a sample
    binary_matrix = np.zeros((n_ms, len(sequence)))
    for ms in range(n_ms):
        binary_matrix[ms,:] = (sequence==ms)
    # Compute the occurrence per second for each microstate
    occurrence_per_sec = []
    for msbin in binary_matrix:
        # Compute the number of occurrences per second in each 1s window
        conv = np.convolve(msbin, np.ones(fs), mode='valid')
        # Compute the median number of occurrences per second
        occurrence_per_sec.append(np.median(conv))
    return occurrence_per_sec

def normalized_lempel_ziv_complexity(sequence,n_ms):
    """
    Compute the normalized Lempel-Ziv complexity of a sequence of microstates
    """
    
    seqstr = ''.join([str(i) for i in sequence])
    c = lempel_ziv_complexity(seqstr)
    L = len(seqstr)
    if L == 0 or L == 1:
        return 0
    c_bound = 1*L/(np.log(L)/np.log(n_ms))
    return c/c_bound

def nLZC(seq,n_ms,window=None):
    if window is not None:
        if window == 'auto':
            window = correlation_length(seq)
        L = len(seq)
        if L < window+window//2:
            return nLZC(seq,n_ms,window=None)
        else:
            # evaluate LZC for each subseq of length window with half overlap
            subseqs = [seq[i:i+window] for i in range(0,L-window,window//2)]
            all_lzc = np.array([nLZC(subseq,n_ms)[0] for subseq in subseqs]) 
            return np.mean(all_lzc), np.std(all_lzc)
        
    if len(seq)==0 or len(seq)==1:
        return 0,0
    
    seqstr = ''.join([str(i) for i in seq])
    lzc = lempel_ziv_complexity(seqstr)
    bound_bruijin = len(seqstr) / (np.log(len(seqstr))/np.log(n_ms))
    
    return lzc/bound_bruijin, 0


def correlation_length(seq):
    # compute autocorrelation length of a sequence
    # seq: input sequence
    # returns: autocorrelation length
    acf = np.correlate(seq, seq, mode='full').astype(float)
    acf = acf[acf.size//2:]
    acf /= acf[0]
    acf = np.abs(acf)
    return np.argmax(acf<1/np.e)

def normalized_entropy(sequence,n_ms):
    """
    Normalized Shannon entropy of the symbolic sequence x with ns symbols.
    Args:
        sequence: symbolic sequence, symbols = [0, 1, 2, ...]
        ns: number of symbols
    Returns:
        h: Shannon entropy of x
    """
    h = entropy(sequence,n_ms)
    return h/np.log2(n_ms)

def entropy(sequence,n_ms):
    """
    Shannon entropy of the symbolic sequence x with ns symbols.
    Args:
        sequence: symbolic sequence, symbols = [0, 1, 2, ...]
        ns: number of symbols
    Returns:
        h: Shannon entropy of x
    """
    h = -np.sum([p*np.log2(p) for p in coverage(sequence,n_ms) if p!=0])
    return h

# %%
