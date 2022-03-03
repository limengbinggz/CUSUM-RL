'''
Plot the empirical rejection probabilities of the combined p values.
'''
import platform, sys, os, re, subprocess
import numpy as np
import pandas as pd
from plotnine import *
os.chdir("/Users/mengbing/Documents/research/RL_nonstationary/cumsumrl/simulation_ihs")
sys.path.append("/Users/mengbing/Documents/research/RL_nonstationary/cumsumrl/")

#%%
gamma_label_list = ["0.9", "0.95"]
method_list = ["_normalized", "", "_int_emp"]
method_label_list = ["Normalised", "Unnormalised", "Integral"]
# gamma_quantile_list = [0.05, 0.1, 0.15, 0.2]
gamma_quantile_list = [0.1]
Ns = [100]
T = 50
kappa_list = np.arange(10, 41, step=5)
default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

for N in Ns:
    dir_label_list = []
    rej_table = []
    for gamma in gamma_label_list:
        for gamma_quantile in gamma_quantile_list:
            dir_nm = "output/combine_p_values/gamma" + re.sub("\\.", "", gamma) + \
                     "_gquantile" + re.sub("\\.", "", str(gamma_quantile))
            for j in range(len(method_list)):
                rej = pd.read_csv(dir_nm + "/rejections" + method_list[j] + "_table.csv", header=None)
                rej = rej.T
                rej['beta'] = gamma_quantile
                rej['method'] = method_label_list[j]
                rej['gamma'] = gamma
                rej_table.append(rej)
    rej_table = pd.concat(rej_table, ignore_index=True)
    print(rej_table)
    rej_table = rej_table.rename({0: 'kappa', 1: 'rej_rate'}, axis=1)
    rej_table.to_csv('output/rej_data.csv', index=False)
    # os.system("module load R")
    subprocess.call("Rscript --vanilla p01_plot_combine_p_values.R", shell=True)

    # # can also run the following code using plotnine
    # p = ggplot(rej_table,
    #            aes(x = 'kappa', y = 'rej_rate', color = 'method'))+ \
    #     facet_grid(facets=". ~ gamma", scales='free_y') + \
    #     geom_line(aes(group = 'method', linetype = 'method'), size = 1.5) +\
    #     geom_point(aes(shape = 'method'), size = 5, alpha=0.6) +\
    #     labs(x = r'$\kappa$', y = 'Rejection Rate', color = 'Method', shape = 'Method', linetype = 'Method') +\
    #     scale_x_continuous(breaks = tuple(kappa_list)) +\
    #     scale_y_continuous(breaks = (0, 0.05, 0.2, 0.4, 0.6, 0.8, 1.0)) +\
    #     scale_color_manual(values = default_colors[0:len(method_list)]) +\
    #     geom_hline(yintercept = 0.8, size = 1, linetype='dashed') +\
    #     geom_hline(yintercept = 0.05, size = 1, linetype='dashed') +\
    #     theme_bw() + \
    #     theme(legend_position="right",
    #       legend_direction="vertical",
    #       legend_box_spacing=0.4,
    #       axis_line=element_line(size=1, colour="black"),
    #       panel_grid_major=element_blank(),
    #       panel_grid_minor=element_blank(),
    #       panel_border = element_rect(colour = "black", size=0.5),
    #       panel_background=element_blank(),
    #       plot_title=element_text(size=18, face="bold"),
    #       text=element_text(size=12),
    #       axis_text_x=element_text(colour="black", size=9, rotation=90),
    #       axis_text_y=element_text(colour="black", size=9)
    #       ) + \
    #     guides(group = False)
    # # if len(method_list) > 1:
    # #     p += facet_grid(facets = ". ~ method", scales = 'free_y')
    # p.save("output/combine_p_values/real_rejection_rates" + "_N" + str(N) + ".pdf",
    #        width = len(gamma_label_list)*10+4, height = 10, units = "cm", verbose = False)