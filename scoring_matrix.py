
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
plt.style.use('ggplot')
import pandas as pd
import numpy as np
import  sklearn
from sklearn import metrics
from scipy import signal
from scipy.stats import spearmanr

print('The scikit-learn version is {}.'.format(sklearn.__version__))

#http://scipy.github.io/old-wiki/pages/Cookbook/SignalSmooth
#used to smooth before peak analysis
def smooth(x, window_len=8, window='hamming'):    
    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."

    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."
    if window_len<3:
        return x
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"
    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')
    y=np.convolve(w/w.sum(),s,mode='valid')
    return y


def cross_correlate(x, y, normalization='no'):
    x = x.astype(float)
    y = y.astype(float)
    
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError, "correlate only accepts 1 dimension arrays."
    if not normalization in ['no', 'max', 'average']:
        raise ValueError, "normalization is no, max, average"
    
    if normalization == 'no':
        pass
    if normalization == 'max':
        x = x/np.max(x)
        y = y/np.max(y)
    if normalization == 'average':
        x = x/np.mean(x)
        y = y/np.mean(y)

    a = np.correlate(x,x)[0]
    b = np.correlate(y,y)[0]
    c = np.correlate(x,y)[0]
    cc =c/max([b,a]) 
    return cc

#compute the peaks in common between two signals (common)
#divide the number of common peaks by the number 
#of total peaks in both signals (common/float(tot))
#so that the max value is 0.5
#compute coapex (1 if the max of both signal is in the same fraction)      
def peak_scores(x, y, peak_exclude = []):
    
    peaks_x = signal.find_peaks_cwt(x, widths=np.array([4,5]))
    peaks_y = signal.find_peaks_cwt(y, widths=np.array([4,5]))

    tot = 0
    common = 0
    for a in peaks_x:
        for b in peaks_y:
            print a,b
            #peak is the same if in the 
            #same fraction +/- 1            
            if abs(a-b) <= 1:
                if a not in peak_exclude:
                    if  b not in  peak_exclude:
                        common+=1
                        break 
    
    #compute total number of peaks for each signal
    for a in peaks_x:    
        if a not in peak_exclude:
            tot+=1
    for b in peaks_y:            
        if b not in peak_exclude:
            tot+=1
            
    max_peak_x = np.argmax(x)
    max_peak_y = np.argmax(y)
    if max_peak_x not in peak_exclude and max_peak_y not in peak_exclude:
        if max_peak_x == max_peak_x:
            coapex = 1
        else:
            coapex = 0 
    if tot == 0:
        tot = 1
    
    return common, common/float(tot), coapex


#scoring             
def score_profiles(profile_1, profile_2, peak_exclude=[]): 

    res = []
    start_label = ['EVS','MAE','MSE','MdAE','R2']
    labels = []
    labels += [n+'_abs' for  n in start_label]
    labels += [n+'_norm' for  n in start_label]
    labels += ['ADR','CS','FMS','HS','MIS','VMS','AS','CKS','F1','HL','PS','RS']
    labels += ['CC_abs','CC_max','CC_average']
    labels += ['PC','SC']    
    labels += ['CM','CM_norm','CA']
    #array input
    #transform to float for regression metrics
    y_true = profile_1.astype(float)
    y_pred = profile_2.astype(float)
    if y_true.sum() == 0.0 or y_pred.sum() == 0.0:
        res = pd.Series(data=np.zeros(len(labels)),index=labels)
        return res
    
    #labels += [n+'_abs' for  n in start_label]
    res.append( metrics.explained_variance_score(y_true, y_pred) )
    res.append( metrics.mean_absolute_error(y_true, y_pred) )
    res.append( metrics.mean_squared_error(y_true, y_pred) )
    res.append( metrics.median_absolute_error(y_true, y_pred) )
    res.append( metrics.r2_score(y_true, y_pred) )
    
    #labels += [n+'_norm' for  n in start_label]
    #use regression metrics after norm max
    res.append( metrics.explained_variance_score(y_true/y_true.max(), y_pred/y_pred.max()))
    res.append( metrics.mean_absolute_error(y_true/y_true.max(), y_pred/y_pred.max()))
    res.append( metrics.mean_squared_error(y_true/y_true.max(), y_pred/y_pred.max()))
    res.append( metrics.median_absolute_error(y_true/y_true.max(), y_pred/y_pred.max()))
    res.append( metrics.r2_score(y_true/y_true.max(), y_pred/y_pred.max()))     
    
    #use label metrics after norm max
    #round to first digit, multiply * 10 and transform to int
    y_true = np.array([int(round(n,1)*10) for n in  y_true/y_true.max()])
    y_pred = np.array([int(round(n,1)*10)  for n in  y_pred/y_pred.max()])
    
    #labels += ['ADR','CS','FMS','HS','MIS','VMS','AS','CKS','F1','HL','PS','RS']
    res.append( metrics.adjusted_rand_score(y_true,y_pred) )
    res.append( metrics.completeness_score(y_true, y_pred) )
    res.append( metrics.fowlkes_mallows_score(y_true, y_pred) )
    res.append( metrics.homogeneity_score(y_true, y_pred)  )  
    res.append( metrics.mutual_info_score(y_true, y_pred) )
    res.append( metrics.v_measure_score(y_true, y_pred) )    
    res.append( metrics.accuracy_score(y_true, y_pred) )
    res.append( metrics.cohen_kappa_score(y_true, y_pred) )
    res.append( metrics.f1_score(y_true, y_pred,average ='weighted') )
    res.append( metrics.hamming_loss(y_true, y_pred) )
    res.append( metrics.precision_score(y_true, y_pred, average ='weighted') )
    res.append( metrics.recall_score(y_true, y_pred, average ='weighted')  )
    
    #labels += ['CC_abs','CC_max','CC_average']
    res.append( cross_correlate(profile_1,profile_2,'no') )
    res.append( cross_correlate(profile_1,profile_2,'max') )
    res.append( cross_correlate(profile_1,profile_2,'average') )
    
    #labels += ['PC','SC']
    res.append( np.corrcoef(profile_1,profile_2)[0][1])
    res.append( spearmanr(profile_1,profile_2)[0] )
     
    #labels += ['CM','CM_norm','CA']
    common_peak, common_peak_norm, coapex = peak_scores(profile_1, profile_2, peak_exclude)
    res.append(common_peak)
    res.append(common_peak_norm)    
    res.append(coapex)    
    res = [round(n,3) for n in res]
    res = pd.Series(data=res,index=labels)
    print 100
    return  res
    

if __name__ == '__main__':
    
    #df = pd.DataFrame.from_csv('test/test_msms_count.txt',sep='\t')
    df = pd.DataFrame.from_csv('test/test_lfq.txt',sep='\t')    
    profile_1 = df.iloc[0,:]
    profile_2 = df.iloc[1,:]
    plt.plot(smooth(profile_1.values))
    plt.plot(smooth(profile_2.values))
    #plt.plot(profile_1.values)
    #plt.plot(profile_2.values)    
    profile_1 = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    profile_2 = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    print peak_scores(smooth(profile_1),smooth(profile_2),peak_exclude=np.arange(30,53,1))
    score_res = score_profiles(profile_1, profile_2, peak_exclude=np.arange(30,53,1))
    for a,b in zip(score_res.index.values,score_res.values):
        print b,'\t', a
     
                  
