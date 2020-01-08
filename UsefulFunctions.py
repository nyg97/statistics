'''
coding: utf-8
----------------------------------------------------
Useful functions for Applied Statistics Exam

Authors:
 - Ulrik Friis-Jensen (lgb543@alumni.ku.dk)
 
Co-authors:
 - Christian Noes Petersen (lbc622@alumni.ku.dk)
 - David Harding-Larsen (pfl888@alumni.ku.dk)
 - Lars Erik Skjegstad (zfj803@alumni.ku.dk)
 - Marcus Frahm Nygaard (nwb154@alumni.ku.dk)

Date:
 - 07-01-2020 (latest update)
-----------------------------------------------------
To Be added:
 - Output in LaTex Table format

Other To Do's:
 - Test all functions
-----------------------------------------------------
'''
# Imports
import sys   
import numpy as np
from scipy import stats
from iminuit import Minuit     
sys.path.append('../External_Functions')
from ExternalFunctions import Chi2Regression

# Functions for ChiSquare
def constant(x, const):
    return const

def linear(x, a, b):
    return a * x + b

def binomial(x, n, p, N = 1):
    return N * stats.binom.pmf(x,n,p)

def poisson(x, mu, N = 1) :
    return N * stats.poisson.pmf(x, mu)

def gaussian(x, N = 1.0, mu = 0.0, sigma = 1.0, binwidth = 1.0) :
    return binwidth * N * stats.norm.pdf(x, mu, sigma)

def gaussian_x2(x, N1, mu1, sigma1, N2, mu2, sigma2, binwidth1 = 1.0, binwidth2 = 1.0):
    return gaussian(x, N1, mu1, sigma1, binwidth=binwidth1) + gaussian(x, N2, mu2, sigma2, binwidth=binwidth2)

def gaussian_x3(x, N1, mu1, sigma1, N2, mu2, sigma2, N3, mu3, sigma3, binwidth1 = 1.0, binwidth2 = 1.0, binwidth3 = 1.0):
    return gaussian(x, N1, mu1, sigma1, binwidth=binwidth1) + gaussian(x, N2, mu2, sigma2, binwidth=binwidth2) + gaussian(x, N3, mu3, sigma3, binwidth=binwidth3)

def exponential_decay(x, C, k):
    return C * np.exp(-x/k)

def exponential_growth(x, C, k):
    return C * np.exp(x/k)

def sigmoid(x, L, x0):
    return L * ((x - x0) / np.sqrt(1 + (x - x0)**2))

# Simple functions
def mean_no_unc(data, get_values=False):
    '''
    Calculates the mean, RMS and uncertainty on mean for a data sample w/o uncertainties.
    '''
    mean = data.mean()
    unc_on_data = np.sqrt(np.sum((data-mean)**2)/(len(data)-1))
    unc_on_mean = unc_on_data / np.sqrt(len(data))
    print(f'''
    ____________________________________________________
    ----------------------------------------------------
    Mean of data set:  {mean:.4f} +/- {unc_on_mean:.4f} (RMS = {unc_on_data:.4f})
    ____________________________________________________''')
    if get_values:
        return mean, unc_on_data, unc_on_mean
    else:
        return None
    
def bin_data(data, Nbins, xmin, xmax):
    '''
    Converts a list or array to a histogram.
    Returns bin_centers, counts, error on counts and binwidth.
    '''
    counts, bin_edges = np.histogram(data, bins=Nbins, range=(xmin, xmax))
    bin_centers = (bin_edges[1:] + bin_edges[:-1])/2
    s_counts = np.sqrt(counts) 
    
    x = bin_centers[counts>0]
    y = counts[counts>0]
    sy = s_counts[counts>0]
    
    binwidth = (xmax-xmin) / Nbins
    return x, y, sy, binwidth

# Advanced functions

def chi2_test_uniform(bin_centers, counts, get_values=False):
    '''
    Tests if a histogram is uniformly distributed.
    '''
    data = counts
    expected = data.sum() / len(bin_centers)
    chi2 = np.sum( (data - expected)**2 / data )
    Ndof = len(bin_centers)
    p_chi2 = stats.chi2.sf(chi2, Ndof) 

    print(f'''
    _____________________________
    -----------------------------
    ChiSquare test (uniform dist)
    -----------------------------
    Chi2-value = {chi2:.3f}
    Ndof       = {Ndof}
    Chi2-prob  = {p_chi2:.2%}
    _____________________________''')
    if get_values:
        return chi2, Ndof, p_chi2
    else:
        return None
    
def pearsons_chi2(counts, expected_dist, get_values=False):
    '''
    Pearson's ChiSquare test for comparing a histogram to a distribution.
    Input arguments are the observed counts and the expected binomial/poisson.
    '''
    chi2 = 0
    events = 0
    for A, B in zip(counts, counts.sum()*expected_dist):
        if A != 0 and B != 0:
            chi2 += (A - B)**2 / (A + B)
            events += 1

    Ndof = events
    p_chi2 = stats.chi2.sf(chi2, Ndof) 

    print(f'''
    ___________________________
    ---------------------------
     Pearson's ChiSquare test
    ---------------------------
    Chi2-value = {chi2:.3f}
    Ndof       = {Ndof}
    Chi2-prob  = {p_chi2:.2%}
    ___________________________''')
    if get_values:
        return chi2, Ndof, p_chi2
    else:
        return None

def ks_comparison(data1, data2, alternative = 'two-sided', get_values=False):
    '''
    Kolmogorov-Smirnov test for comparing to datasets.
    Returns the test statistic, critical value and p-value either as a string or numbers.
    Alternative hypothesis can be:
        'two-sided'
        'less'
        'greater'
    '''
    D, p = stats.ks_2samp(data1, data2, alternative=alternative)
    d = D * np.sqrt(len(data1))
    print(f'''
    ____________________________________________________________
    ------------------------------------------------------------
    Result of Kolmogorov-Smirnov comparison between two datasets
    ------------------------------------------------------------
    KS statistic   :    {D:.4f}
    Critical value :    {d:.4f}
    p-value        :    {p:.2%}
    ____________________________________________________________
    ''')
    if get_values:
        return D, d, p
    else:
        return None
    
def ks_test(data1, cdf, alternative = 'two-sided', get_values=False):
    '''
    Kolmogorov-Smirnov test for comparing to datasets.
    Returns the test statistic, critical value and p-value either as a string or numbers.
    Alternative hypothesis can be:
        'two-sided'
        'less'
        'greater'
    '''
    D, p = stats.kstest(data1, cdf, alternative=alternative)
    d = D * np.sqrt(len(data1))
    print(f'''
    _____________________________________________
    ---------------------------------------------
          Result of Kolmogorov-Smirnov test
    ---------------------------------------------
    KS statistic   :    {D:.4f}
    Critical value :    {d:.4f}
    p-value        :    {p:.2%}
    _____________________________________________
    ''')
    if get_values:
        return D, d, p
    else:
        return None
    
def chi2_fit(func, x, y, yerr, get_values=False, pedantic = False, print_level = 0,latex_format=False, **kwdarg):
    '''
    ChiSquare fit of a given function to a given data set.
    
    Returns the fitted parameters for further plotting.
    
    **kwdarg allows the user to specify initial parameter 
    values and fix values using the syntax from Minuit
    '''
    chi2obj = Chi2Regression(func, x, y, yerr)
    minuit_obj = Minuit(chi2obj, pedantic=pedantic, print_level=print_level, **kwdarg)

    minuit_obj.migrad()   

    if (not minuit_obj.get_fmin().is_valid) :                                   # Check if the fit converged
        print("    WARNING: The ChiSquare fit DID NOT converge!!!")

    Chi2_value = minuit_obj.fval                                             # The Chi2 value
    NvarModel = len(minuit_obj.args)
    Ndof = len(x) - NvarModel
    ProbChi2 = stats.chi2.sf(Chi2_value, Ndof)
    if latex_format:
        print(r'''----------------------------------------------------------------------------------
NB! Units, caption, label and sometimes parameter names must be changed in LaTex.
----------------------------------------------------------------------------------
        
\begin{table}[b]
    \centering
    \begin{tabular}{lrr}
    \hline
    \hline
        Parameter & Value (Unit) & Unc. (Unit) \\
    \hline''')
        for name in minuit_obj.parameters:
            print(f'        ${name}$ & ${minuit_obj.values[name]:.5f}$ & ${minuit_obj.errors[name]:.5f}$ \\\ ')
        print(r'''    \hline
    \hline''')
        print(r'        $\chi^2$-value = {0:.3f} & Ndof = {1} & $\chi^2$-prob = {2:.3f} \\'.format(Chi2_value,Ndof,ProbChi2))
        print(r'''    \hline
    \hline
    \end{tabular}
    \caption{Results of $\chi^2$-fit.}
    \label{tab:chi2_fit}
\end{table}''')
    else:
        print(f'''
    _____________________________________________________
    -----------------------------------------------------
               ChiSquare Fit Results
    -----------------------------------------------------
    Chi2-value = {Chi2_value:.3f}
    Ndof       = {Ndof}
    Chi2-prob  = {ProbChi2:.2%}
    -----------------------------------------------------''')
        for name in minuit_obj.parameters:
            print(f'\n    Chi2 Fit result:    {name} = {minuit_obj.values[name]:.5f} +/- {minuit_obj.errors[name]:.5f}')
        print('    _____________________________________________________')
    if get_values:
        return minuit_obj.args, Chi2_value, Ndof, ProbChi2
    else:
        return minuit_obj.args
    
def MonteCarlo(func, N_points, xmin = 0, xmax = 1, ymin = 0, ymax = 1, print_result=True, **kwdarg):
    '''
    Generate random number according to a pdf using Monte Carlo.
    Inputs are:
        - the pdf
        - the number of points to be generated
        - Ranges of the x and y values (optional)
        - any additional arguments for the pdf (optional)    
    '''
    N_try = 0
    x_accepted = np.zeros(N_points)
    for i in range(N_points):

        while True:
            
            # Count the number of tries, to get efficiency/integral
            N_try += 1   

            # Range that f(x) is defined/wanted in:
            x_test = np.random.uniform(xmin, xmax)  

            # Upper bound for function values:
            y_test = np.random.uniform(ymin, ymax)

            if (y_test <= func(x_test, **kwdarg)):
                break

        x_accepted[i] = x_test
        
    # Efficiency
    eff = N_points / N_try                        

    # Error on efficiency (binomial)
    eff_error = np.sqrt(eff * (1-eff) / N_try) 

    # Integral
    integral =  eff * (xmax-xmin) * (ymax-ymin)

    # Error on integral
    eintegral = eff_error * (xmax-xmin) * (ymax-ymin)  
    if print_result:
        print(f'''
    _____________________________________________________________
    -------------------------------------------------------------
                             Monte Carlo 
    -------------------------------------------------------------
    Generation of random numbers according to the given pdf.
    -------------------------------------------------------------
    Intervals used to sample random numbers:
    x in [{xmin}, {xmax}]
    y in [{ymin}, {ymax}]
    
    Integral of the pdf is:  {integral:.4f} +/- {eintegral:.4f}
    
    Efficiency of the Accept/Reject method is:  {eff:.2%} +/- {eff_error:.2%}
    _____________________________________________________________''')
    return x_accepted

# Functions by Troels

# Calculate ROC curve from two histograms (hist1 is signal, hist2 is background):
def calc_ROC(hist1, hist2) :

    # First we extract the entries (y values) and the edges of the histograms
    y_sig, x_sig_edges, _ = hist1 
    y_bkg, x_bkg_edges, _ = hist2
    
    # Check that the two histograms have the same x edges:
    if np.array_equal(x_sig_edges, x_bkg_edges) :
        
        # Extract the center positions (x values) of the bins (both signal or background works - equal binning)
        x_centers = 0.5*(x_sig_edges[1:] + x_sig_edges[:-1])
        
        # Calculate the integral (sum) of the signal and background:
        integral_sig = y_sig.sum()
        integral_bkg = y_bkg.sum()
    
        # Initialize empty arrays for the True Positive Rate (TPR) and the False Positive Rate (FPR):
        TPR = np.zeros_like(y_sig) # True positive rate (sensitivity)
        FPR = np.zeros_like(y_sig) # False positive rate ()
        
        # Loop over all bins (x_centers) of the histograms and calculate TN, FP, FN, TP, FPR, and TPR for each bin:
        for i, x in enumerate(x_centers): 
            
            # The cut mask
            cut = (x_centers < x)
            
            # True positive
            TP = np.sum(y_sig[~cut]) / integral_sig    # True positives
            FN = np.sum(y_sig[cut]) / integral_sig     # False negatives
            TPR[i] = TP / (TP + FN)                    # True positive rate
            
            # True negative
            TN = np.sum(y_bkg[cut]) / integral_bkg      # True negatives (background)
            FP = np.sum(y_bkg[~cut]) / integral_bkg     # False positives
            FPR[i] = FP / (FP + TN)                     # False positive rate            
            
        return FPR, TPR
    
    else:
        AssertionError("Signal and Background histograms have different bins and ranges")
