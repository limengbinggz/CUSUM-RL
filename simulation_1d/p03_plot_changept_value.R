#!/sw/arcts/centos7/stacks/gcc/8.2.0/R/4.1.0/bin/Rscript
library(data.table)
library(ggplot2)
library(dplyr)
library(ggbreak)
curr_dir <- "/Users/mengbing/Documents/research/RL_nonstationary/cumsumrl/simulation_1d"
setwd(curr_dir)
args = commandArgs(trailingOnly=TRUE)
N <- as.integer(args[1])
dat <- fread(paste0("data_optvalue_dt_N", N, ".csv"))
dat$variable <- factor(dat$variable, 
                       levels = c('proposed - oracle', 'proposed - overall', 'proposed - random',
                                  'proposed - kernel_02', 'proposed - kernel_04', 'proposed - kernel_08',
                                  'proposed - kernel_16'),
                       labels = c('Proposed - Oracle', 'Proposed - Overall', 'Proposed - Random',
                                  "Proposed - Kernel (h=0.2)",
                                  "Proposed - Kernel (h=0.4)",
                                  "Proposed - Kernel (h=0.8)",
                                  "Proposed - Kernel (h=1.6)"))

axis_limits <- data.table(gamma = c(0.9, 0.95),
                          ymax = c(6, 12),
                          ymin = c(-1, -2))
lfun <- function(limits) {
  grp <- dat$gamma[which(abs(dat$value - limits[1]) < 1e-7)]
  ulim <- axis_limits[gamma == grp,]$ymax
  llim <- axis_limits[gamma == grp,]$ymin
  return(c(llim, ulim))
}
n_gamma <- length(unique(dat$gamma))
dat$gamma <- factor(dat$gamma, levels = c(0.9, 0.95), labels = c("0.9", "0.95"))

gamma_names <- list(
  "0.9" = expression(gamma~"= 0.9"),
  "0.95" = expression(gamma~"= 0.95"),
  "Transition: Hm\nReward: PC" = "Transition: Hm\nReward: PC",
  "Transition: Hm\nReward: Sm" = "Transition: Hm\nReward: Sm",
  "Transition: PC\nReward: Hm" = "Transition: PC\nReward: Hm",
  "Transition: Sm\nReward: Hm" = "Transition: Sm\nReward: Hm"
)
gamma_labeller <- function(variable,value){
  return(gamma_names[value])
}

g1 <- ggplot(dat, aes(variable, value, fill=variable)) + 
  facet_grid(facets = gamma ~ Setting, scales = 'free', labeller = gamma_labeller) + 
  geom_hline(yintercept = 0, size = 1, linetype="dashed", color = "red") +
  geom_boxplot() +
  xlab("") + 
  ylab("Value Difference") +
  labs(fill="Method") +
  theme(
    legend.direction="horizontal",
    legend.position = "none",
    panel.border = element_rect(color = "black", fill = NA, size = 1),
    panel.grid.major=element_line(colour="#d3d3d3"),
    panel.grid.minor=element_line(colour="#d3d3d3"),
    panel.background=element_blank(),
    plot.title=element_text(size=18, face="bold"),
    text=element_text(size=14),
    axis.text.x=element_text(colour="black", size=13, angle = 90, vjust = 0.5, hjust = 0.1),
    axis.text.y=element_text(colour="black", size=13),
    plot.margin=grid::unit(c(0.3,0,0,0), "mm")
  ) +
  scale_y_continuous(limits = lfun, expand=expansion(0,0)) 
ggsave(paste0("1d_box_optvalue_dt_N", N, ".pdf"), width = 14, height = n_gamma * 2 + 4, plot = g1)