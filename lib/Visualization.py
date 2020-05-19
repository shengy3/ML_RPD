import pandas as pd
import numpy as np
from .Function import gaussian
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def plot_event(total_event, Xin, Yin, Xcm, Ycm, Scan, run):
    #plot total event
    plt.figure(figsize = (10,10))
    plt.imshow(total_event.reshape(4,4), norm=LogNorm())
    cb = plt.colorbar()
    cb.set_label('Number of photon', rotation=270 ,size = 15, labelpad = 25)
    cb.ax.tick_params(labelsize=20)
    Xin, Yin = get_beam_pos(Scan, run)
    plt.annotate(f"BeamPos: {Xin, Yin} \nCal_CoM:  {Xcm, Ycm}",
                 xy=(0, 1),
                 size=15,
                 xycoords="axes fraction",
                 color='white',
                 xytext=(5, -5),
                 textcoords="offset points",
                 ha="left",
                 va="top")
    plt.xlabel("X position (mm)", size = 20)
    plt.ylabel("y position (mm)", size = 20)
    plt.gca().set_xticklabels(['']*10)
    plt.gca().set_yticklabels(['']*10)

    plt.title(f" RPD Run {run}", size = 30)
    plt.savefig(f'./fig/{run}.png')
    plt.show()

def plot_hist_matrix(df, fig_name):
    dim = len(df.columns)
    fig, axes = plt.subplots(nrows = dim, ncols = dim, figsize = (20,20))
    for i, (labely, datay) in zip(range(dim), df.items()):
        for j, (labelx, datax) in zip(range(dim), df.items()):
            if i != j:
                axes[i,j].hist2d(datax,datay, bins = 50, range = [[-40, 40], [-40, 40]] ,norm = LogNorm())
            elif i == j:
                axes[i,j].hist(datax, bins=80, density = True, range =[-40,40])
            
            axes[i,j].set_xlabel(labelx, fontsize = 20, rotation = 0)
            axes[i,j].set_ylabel(labely, fontsize = 20, rotation = 90,  labelpad = 10)
            #axes[i,j].set_title(f"{i},{j}")

    for i, ax in enumerate(axes.flat):
        ax.label_outer()

    plt.subplots_adjust(wspace = 0.05, hspace = 0.07)
    plt.tight_layout()
    plt.savefig(f"./Output/fig/{fig_name}.pdf")
    

def plot_residual(residual, **plt_para):

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

