import platform, sys, os, re, subprocess
import numpy as np
import pandas as pd
from plotnine import *
os.chdir("/Users/mengbing/Documents/research/RL_nonstationary/cumsumrl/simulation_1d")
sys.path.append("/Users/mengbing/Documents/research/RL_nonstationary/cumsumrl/")

#%%
gamma_label_list = ["0.9", "0.95"]
method_list = ["_normalized", "", "_int_emp"]
method_label_list = ["Normalised", "Unnormalised", "Integral"]
# gamma_quantile_list = [0.05, 0.1, 0.15, 0.2]
gamma_quantile_list = [0.1]
Ns = [25,100]
T = 100
kappa_list = np.arange(25, 76, step=5)
default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
setting_list = [['homo', 'pwconst2'],
                ['homo', 'smooth'],
                ['pwconst2', 'homo'],
                ['smooth', 'homo']]
setting_label_list = ['Transition: Homo\nReward: PC',
                'Transition: Homo\nReward: Sm',
                'Transition: PC\nReward: Homo',
                'Transition: Sm\nReward: Homo']

for N in Ns:
    dir_label_list = []
    adjusted_p_values = []
    for s in range(len(setting_list)):
        setting = setting_list[s]
        setting_label = setting_label_list[s]
        for gamma in gamma_label_list:
            gamma_name = re.sub("\.", "", str(gamma))
            for gamma_quantile in gamma_quantile_list:
                dir_nm = 'output/combine_p_values_N' + str(N) + '/trans' + setting[0] + '_reward' + setting[1] + \
                         '_gamma' + gamma_name +  "_gquantile" + re.sub("\.", "", str(gamma_quantile))
                for j in range(len(method_list)):
                    rej = pd.read_csv(dir_nm + "/rejections" + method_list[j] + "_table.csv")
                    rej = rej.T
                    rej['method'] = method_label_list[j]
                    rej['gamma'] = gamma
                    rej['setting'] = setting_label
                    adjusted_p_values.append(rej)
    adjusted_p_values = pd.concat(adjusted_p_values, ignore_index=False)
    adjusted_p_values.insert(0, 'kappa', adjusted_p_values.index)
    adjusted_p_values = adjusted_p_values.rename({0: 'reject'}, axis=1)
    print(adjusted_p_values)
    # dat = adjusted_p_values.groupby(['setting', 'method', 'gamma', 'kappa'])['reject'].mean().reset_index()
    # print(dat)
    adjusted_p_values.to_csv('output/rej_data_N' + str(N) + '.csv', index=False)
    # os.system("module load R")
    subprocess.call("Rscript --vanilla p01_plot_combine_p_values.R " + str(N), shell=True)

    # # can also run the following code using plotnine
    # p = ggplot(dat,
    #            aes(x = 'kappa', y = 'reject', color = 'method'))+ \
    #     facet_grid(facets="gamma ~ setting", scales='free_y') + \
    #     geom_line(aes(group = 'method', linetype = 'method')) +\
    #     geom_point(aes(shape = 'method'), size = 4, alpha=0.6) +\
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
    #       axis_text_y=element_text(colour="black", size=9),
    #       ) + \
    #     guides(group = False)
    # # if len(method_list) > 1:
    # #     p += facet_grid(facets = ". ~ method", scales = 'free_y')
    # p.save("output/combine_p_values/1d_rejection_rates" + "_N" + str(N) + ".pdf",
    #        width = len(gamma_label_list)*10+4, height = 10, units = "cm", verbose = False)
    # facet_grid(facets="beta ~ method", scales='free_y') + \