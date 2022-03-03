library(ggplot2)
library(data.table)
library(lemon)
curr_dir <- "/Users/mengbing/Documents/research/RL_nonstationary/cumsumrl/simulation_ihs"
setwd(curr_dir)
dat <- fread("output/changept_data.csv")
methods <- c("Integral", "Normalised", "Unnormalised")
gammas <- c("0.9", "0.95")
dat$method <- factor(dat$method, levels = methods)
dat$gamma <- factor(dat$gamma, levels = gammas, 
                    labels = expression(paste(gamma, " = 0.9"), paste(gamma, " = 0.95")))
unique_cpt <- unique(dat$changept)
x_breaks <- seq(0, 50, by = 5)
dat$changept <- factor(dat$changept, levels = unique_cpt)
# factor(dat$changept, levels = x_breaks)
# dat <- dat[order(dat$type)]
default_colors = c('#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf')
p = ggplot(dat, aes(x=changept, fill = method)) + 
  geom_bar(position = "dodge", alpha = 0.9) +
  facet_rep_grid(facets=. ~ gamma, labeller = label_parsed) +
  labs(x = expression(paste("Estimated Change Point T - ", kappa, "*")), y='Count', fill='Method') + 
  scale_x_discrete(drop=FALSE) +
  scale_fill_manual(values=default_colors[1:length(unique(dat$method))]) + 
  theme_bw() + 
  theme(legend.position="right",
        legend.direction="vertical",
        axis.line=element_line(size=0.5, colour="black"),
        panel.grid.major=element_blank(),
        panel.grid.minor=element_blank(),
        panel.spacing=unit(0,'npc'),
        panel.background=element_blank(),
        plot.title=element_text(size=18, face="bold"),
        text=element_text(size=13),
        axis.text.x=element_text(colour="black", size=10, angle=0),
        axis.text.y=element_text(colour="black", size=10),
  ) 
# p
ggsave("output/ihs_bar.pdf", width=length(gammas)*10+4, height=10, units = "cm")