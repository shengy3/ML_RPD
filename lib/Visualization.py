import pandas as pd
import numpy as np
from .Function import gaussian
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm



def plot_residual(residual, **plt_para):
    """
    residual: it take array with dimension (events, x, y) and draw the histogram of each evetns
    plt_para: take the keywords for the plot
    
    example: 
    'fit_function':fit_gaussian,
    "init_para": initial fit constant for the gaussian. (10, 1, 1)
    "n_bins": number of bin for the historgram in fit_range_def, 200
    "fit_range_def": the range for gaussian fitting. (-10, 10)
    "range_def": the whole range of the histogram, (-10, 10)
    "xlim": the range of the plot in x, [-3, 3],
    "density": normalize histogram to density, True
    "output_path": output path for figure, " "
    
    """
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20,10))
    
    ax0 = get_residual_subplot(ax[0], residual[:,0], **plt_para)
    ax0.set_title("X residual \n (prediction - truth)", size = 15)
    ax0.set_xlim(plt_para['xlim'])
    ax1 = get_residual_subplot(ax[1], residual[:,1], **plt_para)
    ax1.set_title("Y residual \n (prediction - truth)", size = 15)
    ax1.set_xlim(plt_para['xlim'])
    plt.subplots_adjust(hspace=0.25)
    plt.savefig(plt_para['output_path'])
    return fig, ax
                
def get_residual_subplot(ax, residual,  **fit_paras):
    ax, _, x_h= generate_hist(ax, residual, **fit_paras)
    fit_paras['n_bins'] = int(fit_paras['fit_range_def'][0]/ fit_paras['range_def'][0]  * fit_paras['n_bins'])
    fit_paras['range_def'] = fit_paras['fit_range_def']
    
    _, y, x= generate_hist(ax, residual, **fit_paras)
    
    
    if fit_paras['fit_function'] is not None:
        popt, cov= fit_paras['fit_function'](x,y,fit_paras['init_para'])
        ax = plot_subGaussian(ax, x_h, popt)
        annote_string = generate_annote_string(len(popt))
        std_mean = []
        for i in range(len(popt)):
            std_mean.append(popt[i][-2])
            std_mean.append(np.sqrt(cov[i][-2][-2]))
            std_mean.append(popt[i][-1])
            std_mean.append(np.sqrt(cov[i][-1][-1]))
        ax.annotate( annote_string.format(*std_mean), 
            xy=(0, 1), size=15,
            xycoords="axes fraction",
            xytext=(5, -5), textcoords="offset points",
            ha="left", va="top")

    return ax

def generate_hist(ax, residual, **fit_paras):
    y, b, _  = ax.hist(residual, bins = fit_paras['n_bins'], range = fit_paras['range_def'], density = fit_paras['density'] )
    ax.set_xlabel("Residual(mm)", size = 15)
    ax.set_ylabel("Number of events", size = 15)
    x = np.linspace(fit_paras['range_def'][0], fit_paras['range_def'][1], fit_paras['n_bins'])
    
    return ax, y, x

def plot_subGaussian(ax, x, popt):
    n_popt =len(popt)
    std = []
    mean = []
    #plot subgaussian
    if n_popt == 1:
        ax.plot(x,gaussian(x,*popt[0]), lw=1,label=f'Gaus1', color ='black')
        std = [abs(popt[0][-1])]
        mean = [popt[0][-2]]
    else:
        for i in range(n_popt):
            ax.plot(x,gaussian(x,*popt[i]), lw=1,label=f'Gaus{i+1}', color ='black')
    return ax

def generate_annote_string(n_popt):
    annote_string = ""
    for i in range(1, n_popt + 1):
        annote_string += f"m$_{i}$" + "={:.2f} $\pm$ {:.1e},\n" +f"\u03C3$_{i}$"+" = {:.2f}$\pm$ {:.1e} \n"
    return annote_string

