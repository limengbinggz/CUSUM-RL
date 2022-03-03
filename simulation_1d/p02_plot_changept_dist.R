library(ggplot2)
library(data.table)
library(lemon)
curr_dir <- "/Users/mengbing/Documents/research/RL_nonstationary/cumsumrl/simulation_1d"
setwd(curr_dir)
args = commandArgs(trailingOnly=TRUE)
print(args)
N = as.integer(args[1])
dat <- fread(paste0("output/changept_data_N", N, ".csv"))
methods <- c("Integral", "Normalised", "Unnormalised")
gammas <- c("0.9", "0.95")
settings <- c('Transition: Homo\nReward: PC',
              'Transition: Homo\nReward: Sm',
              'Transition: PC\nReward: Homo',
              'Transition: Sm\nReward: Homo')
dat$method <- factor(dat$method, levels = methods)
dat$gamma <- factor(dat$gamma, levels = gammas, 
                    labels = expression(paste(gamma, " = 0.9"), paste(gamma, " = 0.95")))
dat$setting <- factor(dat$setting, levels = settings)
# remove 0
dat <- dat[dat$changept != 0,]
unique_cpt <- sort(unique(dat$changept))
x_breaks <- seq(25, 75, by = 5)
dat$changept <- factor(dat$changept, levels = x_breaks)
# factor(dat$changept, levels = x_breaks)
# dat <- dat[order(dat$type)]
default_colors = c('#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf')
p = ggplot(dat, aes(x=changept, fill = method)) + 
  geom_bar(position = "dodge", alpha = 0.9) +
  # stat_summary(geom = "bar", fun = "sum", position = "identity")
  # facet_rep_grid(facets=gamma~setting, labeller = label_parsed) +
  facet_rep_grid(facets=gamma ~ setting, labeller = labeller(.rows = label_parsed, .multi_line = TRUE)) +# label_parsed
  labs(x = expression(paste("Estimated Change Point T - ", kappa, "*")), y='Count', fill='Method') + 
  # scale_x_continuous(breaks=x_breaks) + 
  scale_x_discrete(drop=FALSE) +
  scale_fill_manual(values=default_colors[1:length(unique(dat$method))]) + 
  theme_bw() + 
  theme(legend.position="right",
        legend.direction="vertical",
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
ggsave(paste0("output/1d_bar_N", N, ".pdf"), width = length(settings)*8+4, height = length(gammas)*6+2, units = "cm")



