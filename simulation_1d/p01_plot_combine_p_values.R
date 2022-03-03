library(ggplot2)
library(data.table)
library(lemon)
curr_dir <- "/Users/mengbing/Documents/research/RL_nonstationary/cumsumrl/simulation_1d"
setwd(curr_dir)
args = commandArgs(trailingOnly=TRUE)
Ns <- as.integer(args[1])
methods <- c("Integral", "Normalised", "Unnormalised")
gammas <- c("0.9", "0.95")
settings <- c('Transition: Homo\nReward: PC',
              'Transition: Homo\nReward: Sm',
              'Transition: PC\nReward: Homo',
              'Transition: Sm\nReward: Homo')
default_colors = c('#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf')

for (N in Ns){
  dat <- fread(paste0("output/rej_data_N", N, ".csv"))
  dat$method <- factor(dat$method, levels = methods)
  dat$gamma <- factor(dat$gamma, levels = gammas, 
                      labels = expression(paste(gamma, " = 0.9"), paste(gamma, " = 0.95")))
  dat$setting <- factor(dat$setting, levels = settings)
  kappa_list <- unique(dat$kappa)
  # print(dat)
  dat <- dat[order(dat$method)]
  p = ggplot(dat, aes(x = kappa, y = reject, color = method)) + 
    facet_rep_grid(facets=gamma ~ setting, labeller = labeller(.rows = label_parsed, .multi_line = TRUE)) # label_parsed
  for (method in unique(dat$method)){
    p = p + geom_line(aes(x = kappa, y = reject, linetype = method),
                      data = dat[method == method,], size = 1.5) +
      geom_point(aes(x = kappa, y = reject, shape = method),
                 data = dat[method == method,], size = 4, alpha=0.8)
    
  }
  p = p +
    labs(x = expression(kappa), y = 'Rejection Probability', color = 'Method', 
         shape = 'Method', linetype = 'Method') +
    scale_x_continuous(breaks = kappa_list) + 
    scale_y_continuous(breaks = c(0, 0.05, 0.2, 0.4, 0.6, 0.8, 1.0)) + 
    scale_color_manual(values = default_colors[1:length(methods)]) + 
    geom_hline(yintercept = 0.8, size = 1, linetype='dashed') + 
    geom_hline(yintercept = 0.05, size = 1, linetype='dashed') +
    theme_bw() + 
    theme(legend.position="right",
          legend.direction="vertical",
          strip.text = element_text(size = 10),
          # legend.box.spacing=0.4,
          axis.line=element_line(size=0.5, colour="black"),
          # panel.grid.major=element_line(colour="#d3d3d3"),
          panel.grid.major=element_blank(),
          panel.grid.minor=element_blank(),
          # panel.border=element_blank(),
          panel.spacing=unit(0,'npc'),
          panel.background=element_blank(),
          plot.title=element_text(size=18, face="bold"),
          text=element_text(size=13),
          axis.text.x=element_text(colour="black", size=10, angle=0),
          axis.text.y=element_text(colour="black", size=10),
    ) 
  ggsave(paste0("output/combine_p_values_N", N, "/1d_rejection_rates_N", N, ".pdf"),
         width = length(settings)*8+4, height = length(gammas)*6+2, units = "cm")
}