import platform
import sys
import os, subprocess
import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd

plat = platform.platform()
print(plat)
if plat == 'macOS-12.5-x86_64-i386-64bit': ##local
    os.chdir("/Users/mengbing/Documents/research/RL_nonstationary/code2/simulation_nonstationary_changept_detection")
    sys.path.append("/Users/mengbing/Documents/research/RL_nonstationary/code2/")
elif plat == 'Linux-3.10.0-1160.42.2.el7.x86_64-x86_64-with-centos-7.6.1810-Core' or plat == 'Linux-3.10.0-1160.53.1.el7.x86_64-x86_64-with-centos-7.6.1810-Core':  # biostat cluster
    os.chdir("/home/mengbing/research/RL_nonstationary/code2/simulation_nonstationary_changept_detection")
    sys.path.append("/home/mengbing/research/RL_nonstationary/code2/")
elif plat == 'Linux-4.18.0-305.45.1.el8_4.x86_64-x86_64-with-glibc2.28':  # greatlakes
    os.chdir("/home/mengbing/research/RL_nonstationary/code2/simulation_nonstationary_changept_detection")
    sys.path.append("/home/mengbing/research/RL_nonstationary/code2/")

import re

# N = int(100)
setting_list = [['homo', 'pwconst2'],
                ['homo', 'smooth'],
                ['pwconst2', 'homo'],
                ['smooth', 'homo']]
setting_label_full_list = [['Homogeneous', 'Piecewise Constant'],
                ['Homogeneous', 'Smooth'],
                ['Piecewise Constant', 'Homogeneous'],
                ['Smooth', 'Homogeneous']]
setting_label_list = [['Hm', 'PC'],
                ['Hm', 'Sm'],
                ['PC', 'Hm'],
                ['Sm', 'Hm']]
effect_sizes = ["strong", "moderate", "weak"]
gamma_list = [0.9, 0.95]#, 0.9
gamma_label_list = [re.sub("\.", "", str(x)) for x in gamma_list]
# Ns = [25, 100]
date = '20221005'
# method = '_int_emp'
nsim = 50
# N=200

# gamma = gamma_list[0]


#%%

def replace_na(x):
    if x == -999:
        return np.nan
    else:
        return x

def load_replace_na(file_name):
    try:
        out = replace_na(pickle.load(open(file_name, "rb")))
        return out
    except:
        print("cannot open", file_name)
        return -999 # np.nan


Ns = [100]
file_name_parts = [
    ['proposed', ''],
    ['overall', ''],
    # ['behavior', ''],
    ['oracle', ''],
    # ['oracle80', ''],
    ['random', ''],
    ['kernel0', ''],
    ['kernel01', ''],
    ['kernel02', ''],
    ['kernel04', ''],
    ['kernel08', ''],
    ['kernel16', '']
]
file_name_labels = ['Proposed', #'Proposed (agnostic 0)', 'Proposed (agnostic 1)',
                    'Overall', #'Overall (agnostic 0)', 'Overall (agnostic 1)',
                    'Oracle', 'Random'] #

for N in Ns:
    df = pd.DataFrame(columns=['Setting', '$\\gamma$', 'Effect Size', 'seed',
                               'Method', 'Discounted Reward', 'Average Reward'])
    row_idx_table = 0
    row_idx = 0
    for i in range(len(setting_list)):
        setting = setting_list[i]
        print(setting)

        for effect_size in effect_sizes:
            print(effect_size)
            append_name = '_' + effect_size

            for j in range(len(gamma_list)):
                gamma = gamma_list[j]
                data_path0 = 'data/' + date + '_N' + str(N) + '_trans' + setting[0] + '_reward' + setting[1] + '_gamma' + re.sub(
                    "\.", "", str(gamma)) + append_name + '/gamma' + re.sub(
                    "\.", "", str(gamma)) + append_name
                for nrep in range(1, nsim + 1):
                    for type in file_name_parts:
                        file_name = data_path0 + "_" + str(nrep) + "/value_online_" + type[0] + '_gamma' + \
                                    re.sub("\.", "", str(gamma)) + ".dat"
                        try:
                            values = load_replace_na(file_name)
                            raw_reward = values['raw_reward']
                            # if nrep == 1:
                            #     print(raw_reward.shape)
                            # discounted reward
                            discounted_reward = 0.0
                            T_initial = 100
                            for t in range(T_initial, raw_reward.shape[1]):
                                discounted_reward += raw_reward[:, t] * gamma ** (t - T_initial)
                            discounted_reward = np.mean(discounted_reward)
                            average_reward = np.mean(raw_reward[:, 100:])
                            values['discounted_reward'] = discounted_reward
                            values['average_reward'] = average_reward
                            with open(file_name, "wb") as f:
                                pickle.dump(values, f)
                            df.loc[row_idx] = ["Transition: " + setting_label_list[i][0] + "\nReward: " + setting_label_list[i][1],
                                               gamma, effect_size,
                                               nrep, type[0], discounted_reward, average_reward]
                            row_idx += 1
                        except:
                            print(file_name + " does not exist.")
    print(df)

    df.to_csv('output/data_optvalue_online_dt_N' + str(N) + '.csv', index=False)












# #%%
# import matplotlib.lines as mlines
# import matplotlib.transforms as mtransforms
# # create a table to be saved to csv
# df_table = pd.DataFrame(columns=['Transition', 'Reward', 'N', '$\\gamma$', 'Effect Size',
#                                  'Overall', 'Proposed', 'Oracle', 'Random'])
#                                  # 'Kernel ($\\h$=0.2)', 'Kernel ($\\h$=0.3)',
#                                  # 'Kernel ($\\h$=0.4)'#, 'Kernel ($\\h$=0.5)'
# # create a table used for plotting
# df = pd.DataFrame(columns=['id', 'Setting', 'N', 'gamma', 'Effect Size',
#                            'proposed - oracle', 'proposed - overall', 'proposed - random',
#                            # 'proposed - kernel_02', 'proposed - kernel_03',
#                            # 'proposed - kernel_04', #'kernel_05 - overall',
#                            'changept_proposed'])
#
# row_idx_table = 0
# row_idx = 0
# for i in range(len(setting_list)):
#     setting = setting_list[i]
#     print(setting)
#
#     for effect_size in effect_sizes:
#         print(effect_size)
#         append_name = '_' + effect_size
#
#         for j in range(len(gamma_list)):
#             gamma = gamma_list[j]
#             setting_label = ["Transition: " + setting_label_list[i][0] + "\nReward: " + setting_label_list[i][1],
#                              N, r'$\gamma$ = ' + str(gamma), effect_size]
#
#             data_path0 = 'data/' + date + '_trans' + setting[0] + '_reward' + setting[1] + '_gamma' + re.sub(
#                 "\.", "", str(gamma)) + append_name
#
#             value_overall_list = []
#             value_proposed_list = []
#             # value_mean_list = []
#             # value_behavior_list = []
#             value_oracle_list = []
#             # value_oracle80_list = []
#             value_random_list = []
#             # value_kernel_01_list = []
#             # value_kernel_02_list = []
#             # value_kernel_03_list = []
#             # value_kernel_04_list = []
#             # value_kernel_05_list = []
#             for nrep in range(1, nsim + 1):
#                 data_path = data_path0 + \
#                             '/gamma' + re.sub("\.", "", str(gamma)) + append_name + '_' + str(nrep)
#
#                 value_overall = load_replace_na(data_path + '/value_online_overall_gamma' + re.sub("\.", "", str(gamma)) + '.dat')
#                 value_proposed = load_replace_na(data_path + '/value_online_proposed_gamma' + re.sub("\.", "", str(gamma)) + '.dat')
#                 # value_behavior = load_replace_na(data_path + '/value_observed_gamma' + re.sub("\.", "", str(gamma)) + '.dat')
#                 value_oracle = load_replace_na(data_path + '/value_online_oracle_gamma' + re.sub("\.", "", str(gamma)) + '.dat')
#                 # value_oracle80 = load_replace_na(data_path + '/value_oracle80_gamma' + re.sub("\.", "", str(gamma)) + '.dat')
#                 value_random = load_replace_na(data_path + '/value_online_random_gamma' + re.sub("\.", "", str(gamma)) + '.dat')
#                 # value_kernel_01 = load_replace_na(data_path + '/value_kernel_gamma' + re.sub("\.", "", str(gamma)) + '_bandwidth01.dat')
#                 # value_kernel_02 = load_replace_na(data_path + '/value_kernel_gamma' + re.sub("\.", "", str(gamma)) + '_bandwidth02.dat')
#                 # value_kernel_03 = load_replace_na(data_path + '/value_kernel_gamma' + re.sub("\.", "", str(gamma)) + '_bandwidth030000000000000004.dat')
#                 # value_kernel_04 = load_replace_na(data_path + '/value_kernel_gamma' + re.sub("\.", "", str(gamma)) + '_bandwidth04.dat')
#                 # value_kernel_05 = load_replace_na(data_path + '/value_kernel_gamma' + re.sub("\.", "", str(gamma)) + '_bandwidth05.dat')
#                 # changept_seq = load_replace_na(data_path + '/changept_sequential.dat')
#                 changept_seq = 50
#
#                 value_overall_list.append(value_overall)
#                 value_proposed_list.append(value_proposed)
#                 # value_mean_list.append(value_mean)
#                 # value_behavior_list.append(value_behavior)
#                 value_oracle_list.append(value_oracle)
#                 # value_oracle80_list.append(value_oracle80)
#                 value_random_list.append(value_random)
#                 # value_kernel_01_list.append(value_kernel_01)
#                 # value_kernel_02_list.append(value_kernel_02)
#                 # value_kernel_03_list.append(value_kernel_03)
#                 # value_kernel_04_list.append(value_kernel_04)
#                 # value_kernel_05_list.append(value_kernel_05)
#
#                 # print([value_proposed - value_oracle, value_mean - value_oracle, value_behavior - value_oracle])
#                 df.loc[row_idx] = [nrep] + setting_label + \
#                                   [value_proposed - value_oracle, value_proposed - value_overall, value_proposed - value_random,
#                                    # value_proposed - value_kernel_02,
#                                    # value_proposed - value_kernel_03, value_proposed - value_kernel_04, #value_kernel_05 - value_overall,
#                                    changept_seq]
#
#                 # except OSError:
#                 #     print('cannot open', data_path)
#
#                 row_idx += 1
#
#             # compute mean and standard deviation from raw data
#             means = np.array([np.nanmean(value_proposed_list), np.nanmean(value_oracle_list), np.nanmean(value_overall_list),
#                               np.nanmean(value_random_list)
#                               # np.nanmean(value_kernel_02_list), np.nanmean(value_kernel_03_list), np.nanmean(value_kernel_04_list)
#                               ])#, np.nanmean(value_kernel_05_list)
#             stds = np.array([np.nanstd(value_proposed_list), np.nanstd(value_oracle_list), np.nanstd(value_overall_list),
#                              np.nanstd(value_random_list),
#                              # np.nanstd(value_kernel_02_list), np.nanstd(value_kernel_03_list), np.nanstd(value_kernel_04_list)
#                              ])#, np.nanstd(value_kernel_05_list)
#             df_table.loc[row_idx_table] = setting_label_full_list[i] + [N, gamma, effect_size] + [i + " (" + j + ")" for i, j in zip([f"{mean:0.3f}" for mean in means], [f"{std:0.2f}" for std in stds])]
#             row_idx_table += 1
#
#
#
# stdoutOrigin = sys.stdout
# sys.stdout = open("output/optimal_rewards_online_table_latex" + str(N) + ".txt", "w")
# print(df_table.to_latex(index=True, sparsify=False, float_format="{:0.2f}".format,
#                            multirow=True, escape=False))
# sys.stdout.close()
# sys.stdout=stdoutOrigin
#
# # print(df_table)
#
# from math import sqrt
# #%% scatter plot matrix
# for gamma in gamma_list:
#
#     # import seaborn as sns
#     #
#     # def plot_unity(xdata, ydata, **kwargs):
#     #     mn = min(xdata.min(), ydata.min())
#     #     mx = max(xdata.max(), ydata.max())
#     #     points = np.linspace(mn, mx, 100)
#     #     plt.gca().plot(points, points, color='r', marker=None,
#     #                    linestyle='--', linewidth=2.0)
#     #
#     # grid = sns.pairplot(df[df["gamma"] == r'$\gamma$ = ' + str(gamma)][['proposed - oracle', 'proposed - overall', 'proposed - random'
#     #                               # 'proposed - kernel_02', 'proposed - kernel_03', 'proposed - kernel_04'
#     #                                                                     ]],#, 'kernel_05 - overall' 'kernel_01 - overall',
#     #                     height = 2, corner = True)
#     # grid.map_offdiag(plot_unity)
#     #
#     # # fig = grid.get_figure()
#     # grid.savefig("output/value_scatter_gamma" + re.sub("\.", "", str(gamma)) + ".png")
#
#     # my_scatter = pd.plotting.scatter_matrix(df[df["gamma"] == 'gamma = ' + str(gamma)][['seq - overall', 'oracle - overall', 'behavior - overall', 'random - overall',
#     #                               'oracle80 - overall', 'kernel_01 - overall', 'kernel_02 - overall',
#     #                               'kernel_03 - overall', 'kernel_04 - overall', 'kernel_05 - overall']],
#     #                                         alpha=0.2, figsize  = [15, 15])
#     # plt_count = 0
#     # nrow = 10
#     # i=1
#     # for ax in my_scatter.ravel():
#     #     # plt_count += 1
#     #     ax.set_xlabel(ax.get_xlabel(), fontsize=10, rotation=90)
#     #     ax.set_ylabel(ax.get_ylabel(), fontsize=10, rotation=0)
#     #     # add x=y line except for diagonal histograms
#     #     # if plt_count % nrow == i:
#     #     #     line = mlines.Line2D([0, 1], [0, 1], color='red')
#     #     #     transform = ax.transAxes
#     #     #     line.set_transform(transform)
#     #     #     ax.add_line(line)
#     #     #     i += 1
#     #     #     print(plt_count)
#     #     #     print("i = ", i)
#     #
#     # for i in range(np.shape(my_scatter)[0]):
#     #     for j in range(np.shape(my_scatter)[0]):
#     #         if i != j:
#     #             ax = my_scatter.ravel()[i]
#     #             line = mlines.Line2D([0, 1], [0, 1], color='red')
#     #             line.set_transform(ax.transAxes)
#     #             my_scatter[i, j].add_line(line)
#     #             ax.plot([0, 1], [0, 1], transform=ax.transAxes)
#     # # xpoints = ypoints = plt.xlim()
#     # # plt.plot(xpoints, ypoints, linestyle='--', color='k', lw=3, scalex=False, scaley=False)
#     # # May need to offset label when rotating to prevent overlap of figure
#     # [s.get_yaxis().set_label_coords(-0.8, 0.5) for s in my_scatter.reshape(-1)]
#     #
#     # plt.savefig("output/value_scatter_gamma" + re.sub("\.", "", str(gamma)) + ".png")
#     # plt.show()
#
#
#     #%% box plots of differences between estimated optimal rewards and the oracle optimal rewards
#     df_plot = df.reset_index()
#     df_plot = pd.melt(df_plot, id_vars=['Setting', 'N', 'gamma', 'Effect Size'],
#                       value_vars=['proposed - oracle', 'proposed - overall', 'proposed - random'
#                                   # 'proposed - kernel_02', 'proposed - kernel_03', 'proposed - kernel_04'
#                                   ])#, 'kernel_05 - overall' 'kernel_01 - overall',
#     df_plot['variable'] = df_plot['variable'].astype('category')
#     df_plot['variable'] = df_plot['variable'].cat.reorder_categories(['proposed - oracle', 'proposed - overall', 'proposed - random'
#                                   # 'proposed - kernel_02', 'proposed - kernel_03', 'proposed - kernel_04'
#                                                                       ])#, 'kernel_05 - overall' 'kernel_01 - overall',
#     print(df_plot)
#     df_plot.to_csv('output/data_optvalue_online_N' + str(N) + '_dt.csv', index=False)
#
#     # subprocess.call(["module", "load", "R"])
#     # subprocess.call(["Rscript", "--vanilla", "plot_opt_value.R", str(N)])
#
#     from plotnine import *
#     p = ggplot(df_plot, aes("variable", "value", fill="variable")) + \
#         geom_hline(yintercept = 0, size = 1, linetype="dashed", color = "red") +\
#         geom_boxplot() + \
#         xlab("") + \
#         ylab("Value Difference") + \
#         labs(fill="Method") +\
#         theme(
#             legend_direction="vertical",
#             legend_box_spacing=0.4,
#             axis_line=element_line(size=1, colour="black"),
#             panel_grid_major=element_line(colour="#d3d3d3"),
#             panel_grid_minor=element_blank(),
#             panel_background=element_blank(),
#             plot_title=element_text(size=18, face="bold"),
#             text=element_text(size=14),
#             axis_text_x=element_text(colour="black", size=13, rotation=90),
#             axis_text_y=element_text(colour="black", size=13),
#         ) +\
#         scale_fill_brewer(type="qual", palette="Accent") +\
#         facet_grid(facets = "gamma ~ Setting", scales = 'free_y')
#         # facet_grid(facets = "gamma ~ Setting", scales = "free_y")
#     # ggtitle("Monte Carlo difference vs the overall optimal rewards")
#     p.save("output/1d_box_optvalue_dt_N" + str(N) + ".pdf", width = 16, height = 6, verbose = False)
#
#
#     # #%% box plots of change points vs optimal rewards
#     # df_plot = df.reset_index()
#     # df_plot = pd.melt(df_plot, id_vars=['id', 'Setting', 'gamma'], value_vars=['proposed - overall', 'oracle - overall'])
#     # del df_plot['variable']
#     # df_plot.columns = ['id', 'Setting', 'gamma', 'est - overall']
#     # df_plot_cp = df.reset_index()
#     # df_plot_cp = pd.melt(df_plot_cp, id_vars=['id', 'Setting', 'gamma'], value_vars=['changept_proposed'])
#     # del df_plot_cp['variable']
#     # df_plot_cp.columns = ['id', 'Setting', 'gamma', 'changept']
#     # df_plot = df_plot.merge(df_plot_cp, how='inner', on=['id', 'Setting', 'gamma'])
#     # # print(df_plot)
#     #
#     # p = ggplot(df_plot, aes("changept", "est - overall")) + \
#     #     geom_hline(yintercept = 0, size = 1, linetype="dashed", color = "red") +\
#     #     geom_point(aes(color="Setting"), alpha = 0.3, size = 3) + \
#     #     xlab("Change point") + \
#     #     ylab("Estimated - Overall") + \
#     #     ggtitle("Monte Carlo difference vs the overall optimal rewards at different change points") + \
#     #     theme(
#     #         legend_direction="vertical",
#     #         legend_box_spacing=0.4,
#     #         axis_line=element_line(size=1, colour="black"),
#     #         panel_grid_major=element_line(colour="#d3d3d3"),
#     #         panel_grid_minor=element_blank(),
#     #         panel_border=element_blank(),
#     #         panel_background=element_blank(),
#     #         plot_title=element_text(size=18, face="bold"),
#     #         text=element_text(size=14),
#     #         axis_text_x=element_text(colour="black", size=13, rotation=90),
#     #         axis_text_y=element_text(colour="black", size=13),
#     #     ) +\
#     #     facet_grid(facets = "gamma ~ Setting", scales = "free_y")
#     #     # facet_grid(facets = "gamma ~ Setting", scales = "free_y")
#     # p.save("output/scatter_changept_diff_N" + str(N) + ".pdf", width = 16, height = 10, verbose = False)
#
