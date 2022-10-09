#!/sw/arcts/centos7/stacks/gcc/8.2.0/R/4.1.0/bin/Rscript
library(data.table)
library(ggplot2)
library(dplyr)
library(ggbreak)
args = commandArgs(trailingOnly=TRUE)
# N <- as.integer(args[1])
if(Sys.info()["nodename"] %in% c("sph-190219-bio.local", "0587432481.wireless.umich.net")){
  curr_dir <- "/Users/mengbing/Documents/research/RL_nonstationary/code2/simulation_nonstationary_changept_detection/output"
  setwd(curr_dir)
} else{ # greatlakes
  curr_dir <- "/home/mengbing/research/RL_nonstationary/code2/simulation_nonstationary_changept_detection/output"
  setwd(curr_dir)
}

N <- 100
set.seed(50)
dat <- fread(paste0("data_optvalue_online_dt_N", N, ".csv"))

dat_proposed <- dat[Method == "proposed", ]
colnames(dat_proposed)[6:7] <- c("proposed_discounted", "proposed_average")
dat <- merge(x = dat, y = dat_proposed[, -"Method"], 
             by = c("Setting","$\\gamma$","Effect Size","seed"), all.x = TRUE)
dat[,value_diff_discounted := proposed_discounted - `Discounted Reward`]
dat[,value_diff_average := proposed_average - `Average Reward`]

dat2 <- dat[Method != "proposed",]
dat2$Method <- paste0("proposed - ", dat2$Method)
settings <- unique(dat2$Setting)
gammas <- c("$\\gamma$ = 0.9", "$\\gamma$ = 0.95")
effect_sizes = c("strong", "moderate", "weak")

dat2$Method <- factor(dat2$Method, 
                       levels = c('proposed - oracle', 'proposed - overall', 'proposed - random',
                                  'proposed - kernel0', 'proposed - kernel01', 'proposed - kernel02', 
                                  'proposed - kernel04', 'proposed - kernel08', 'proposed - kernel16'),
                       labels = c('Proposed - Oracle', 'Proposed - Overall', 'Proposed - Random',
                                  'Proposed - Kernel (h=0)', 'Proposed - Kernel (h=0.1)', 'Proposed - Kernel (h=0.2)',
                                  'Proposed - Kernel (h=0.4)', 'Proposed - Kernel (h=0.8)', 'Proposed - Kernel (h=1.6)'))
dat2$`$\\gamma$` <- as.character(dat2$`$\\gamma$`)
axis_limits_discounted <- data.table(gamma = rep(c("0.9", "0.95"), 3),
                          effect_size = rep(c("strong", "moderate", "weak"), each = 2),
                          ymax = c(4, 7, 2, 3.5, 0.75, 1.25),
                          ymin = c(-0.5, -1, -0.5, -1, -0.75, -1))
lfun <- function(limits) {
  # print(limits)
  grp_gamma <- dat2$`$\\gamma$`[which(abs(dat2$value_diff_discounted - limits[1]) < 1e-7)]
  grp_effect_size <- dat2$`Effect Size`[which(abs(dat2$value_diff_discounted - limits[1]) < 1e-7)]
  lim_max <- axis_limits_discounted[gamma == grp_gamma & effect_size == grp_effect_size,]$ymax
  lim_min <- axis_limits_discounted[gamma == grp_gamma & effect_size == grp_effect_size,]$ymin
  # print(lim)
  # return(c(max(limits[1]-0.1, -2), lim)) #min(lim, 8)
  return(c(lim_min, lim_max)) #min(lim, 8)
  # return(c(-1, 8))
}
# dat$gamma <- factor(dat$gamma, levels = c(0.9, 0.95), labels = c("0.9", "0.95"))
# dat$gamma <- factor(dat$gamma, levels = c(0.9, 0.95),
#                     labels=c(expression(gamma~"= 0.9"), expression(gamma~"= 0.95")))

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
effect_sizes <- c("strong", "moderate", "weak")
for (effect_size in effect_sizes) {
  (p <- ggplot(dat2[`Effect Size` == effect_size,], aes(Method, value_diff_discounted, fill=Method)) + #, color=`Effect Size`
     geom_hline(yintercept = 0, size = 1, linetype="dashed", color = "red") +
     geom_boxplot() + 
     xlab("") + 
     ylab("Discounted Reward") +
     # labs(fill="Method") +
     theme(
       legend.direction="vertical",
       # panel.border=element_blank(),
       # legend.box.spacing=0.4,
       panel.border = element_rect(color = "black", fill = NA, size = 1),
       # axis.line=element_line(size=1, colour="black"),
       panel.grid.major=element_line(colour="#d3d3d3"),
       panel.grid.minor=element_line(colour="#d3d3d3"),
       panel.background=element_blank(),
       plot.title=element_text(size=18, face="bold"),
       text=element_text(size=14),
       axis.text.x=element_text(colour="black", size=13, angle = 90),
       axis.text.y=element_text(colour="black", size=13),
       plot.margin=grid::unit(c(0.3,0,-2,0), "mm")
     ) +
     # facet_grid(facets = gamma ~ Setting, scales = 'free_y')
     facet_grid(facets = `$\\gamma$` ~ Setting, scales = 'free_y', labeller = gamma_labeller) + #labeller = labeller(.rows = label_parsed, .multi_line = TRUE)
     # scale_y_discrete(labels = c("$\\gamma$ = 0.9" = expression(gamma~"= 0.9"),
     #                             "$\\gamma$ = 0.95" = expression(gamma~"= 0.9"))) +
     scale_fill_brewer(type="qual", palette="Accent") + #=+
     scale_y_continuous(limits = lfun, expand=expansion(0,0)))
  ggsave(paste0("1d_box_optvalue_online_dt_discounted_N", N, "_", effect_size, ".pdf"), width = 14, height = 6)
}



for (effect_size in effect_sizes) {
  (p <- ggplot(dat2[`Effect Size` == effect_size,], aes(Method, value_diff_average, fill=Method)) + #, color=`Effect Size`
     geom_hline(yintercept = 0, size = 1, linetype="dashed", color = "red") +
     geom_boxplot() + 
     xlab("") + 
     ylab("Average Value") +
     # labs(fill="Method") +
     theme(
       legend.direction="vertical",
       # panel.border=element_blank(),
       # legend.box.spacing=0.4,
       panel.border = element_rect(color = "black", fill = NA, size = 1),
       # axis.line=element_line(size=1, colour="black"),
       panel.grid.major=element_line(colour="#d3d3d3"),
       panel.grid.minor=element_line(colour="#d3d3d3"),
       panel.background=element_blank(),
       plot.title=element_text(size=18, face="bold"),
       text=element_text(size=14),
       axis.text.x=element_text(colour="black", size=13, angle = 90),
       axis.text.y=element_text(colour="black", size=13),
       plot.margin=grid::unit(c(0.3,0,-2,0), "mm")
     ) +
     # facet_grid(facets = `$\\gamma$` ~ Setting, scales = 'free_y') +
     facet_grid(facets = `$\\gamma$` ~ Setting, scales = 'free_y', labeller = gamma_labeller) + #labeller = labeller(.rows = label_parsed, .multi_line = TRUE)
     # scale_y_discrete(labels = c("$\\gamma$ = 0.9" = expression(gamma~"= 0.9"),
     #                             "$\\gamma$ = 0.95" = expression(gamma~"= 0.9"))) +
     scale_fill_brewer(type="qual", palette="Accent")) #+
  # scale_y_continuous(limits = lfun, expand=expansion(0,0)))
  ggsave(paste0("1d_box_optvalue_online_dt_average_N", N, "_", effect_size, ".pdf"), width = 14, height = 6)
}



# gamma_names <- list(
#   "0.9" = expression(gamma~"= 0.9"),
#   "0.95" = expression(gamma~"= 0.95"),
#   "Transition: Hm\nReward: PC" = "Transition: Hm\nReward: PC",
#   "Transition: Hm\nReward: Sm" = "Transition: Hm\nReward: Sm",
#   "Transition: PC\nReward: Hm" = "Transition: PC\nReward: Hm",
#   "Transition: Sm\nReward: Hm" = "Transition: Sm\nReward: Hm"
# )
# gamma_labeller <- function(variable,value){
#   return(gamma_names[value])
# }
# dat22 <- dat2
# dat22$`Effect Size` <- factor(dat22$`Effect Size`, levels = c("strong", "moderate", "weak"), 
#                               labels = c("Strong", "Moderate", "Weak"))
# dat22$`$\\gamma$` <- as.character(dat22$`$\\gamma$`)
# (p <- ggplot(dat22, aes(Method, value_diff_discounted, fill=`Effect Size`)) + #, color=`Effect Size`
#     geom_hline(yintercept = 0, size = 1, linetype="dashed", color = "red") +
#     geom_boxplot() + 
#     xlab("") + 
#     ylab("Discounted Reward") +
#     # labs(fill="Method") +
#     theme(
#       legend.direction="vertical",
#       # panel.border=element_blank(),
#       # legend.box.spacing=0.4,
#       panel.border = element_rect(color = "black", fill = NA, size = 1),
#       # axis.line=element_line(size=1, colour="black"),
#       panel.grid.major=element_line(colour="#d3d3d3"),
#       panel.grid.minor=element_line(colour="#d3d3d3"),
#       panel.background=element_blank(),
#       plot.title=element_text(size=18, face="bold"),
#       text=element_text(size=14),
#       axis.text.x=element_text(colour="black", size=13, angle = 90, vjust = 0.3),
#       axis.text.y=element_text(colour="black", size=13),
#       plot.margin=grid::unit(c(0.3,0,-2,0), "mm")
#     ) +
#     # facet_grid(facets = gamma ~ Setting, scales = 'free_y')
#     facet_grid(facets = `$\\gamma$` ~ Setting, scales = 'free_y', labeller = gamma_labeller) + #labeller = labeller(.rows = label_parsed, .multi_line = TRUE)
#     # scale_y_discrete(labels = c("$\\gamma$ = 0.9" = expression(gamma~"= 0.9"),
#     #                             "$\\gamma$ = 0.95" = expression(gamma~"= 0.9"))) +
#     scale_fill_brewer(type="qual", palette="Accent")) #+
# ggsave(paste0("1d_box_optvalue_online_dt_discounted_N", N, ".pdf"), width = 14, height = 8)
# 
# (p <- ggplot(dat22, aes(Method, value_diff_average, fill=`Effect Size`)) + #, color=`Effect Size`
#     geom_hline(yintercept = 0, size = 1, linetype="dashed", color = "red") +
#     geom_boxplot() + 
#     xlab("") + 
#     ylab("Average Value") +
#     # labs(fill="Method") +
#     theme(
#       legend.direction="vertical",
#       # panel.border=element_blank(),
#       # legend.box.spacing=0.4,
#       panel.border = element_rect(color = "black", fill = NA, size = 1),
#       # axis.line=element_line(size=1, colour="black"),
#       panel.grid.major=element_line(colour="#d3d3d3"),
#       panel.grid.minor=element_line(colour="#d3d3d3"),
#       panel.background=element_blank(),
#       plot.title=element_text(size=18, face="bold"),
#       text=element_text(size=14),
#       axis.text.x=element_text(colour="black", size=13, angle = 90, vjust = 0.3),
#       axis.text.y=element_text(colour="black", size=13),
#       plot.margin=grid::unit(c(0.3,0,-2,0), "mm")
#     ) +
#     # facet_grid(facets = gamma ~ Setting, scales = 'free_y')
#     facet_grid(facets = `$\\gamma$` ~ Setting, scales = 'free_y', labeller = gamma_labeller) + #gamma_labeller labeller = labeller(.rows = label_parsed, .multi_line = TRUE)
#     # scale_y_discrete(labels = c("$\\gamma$ = 0.9" = expression(gamma~"= 0.9"),
#     #                             "$\\gamma$ = 0.95" = expression(gamma~"= 0.9"))) +
#     scale_fill_brewer(type="qual", palette="Accent")) #+
# ggsave(paste0("1d_box_optvalue_online_dt_average_N", N, ".pdf"), width = 14, height = 8)










