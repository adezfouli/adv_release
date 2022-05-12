#RNN vs ADV
dd = list()
indx = 1
for (i in c(0:20)){
  data = read.csv(paste0('../../nongit/archive/gonogo/state-reg/static-sims-1/gonog_sim_2000//RL_gonogo_2layers_0.01ent_256units_lof0.0/events_', i,".csv"))
  pol = read.csv(paste0('../../nongit/archive/gonogo/state-reg/policies/policies_', i,".csv"))
  data$ev = i
  data$pol0 = NA
  data$pol1 = NA
  data$pol0 = pol$X0
  data$pol1 = pol$X1
  dd[[indx]] =data
  indx = indx + 1
}

head(data)

require(data.table)
data = rbindlist(dd)
action_levels = levels(data$rnn.action)
data$rnn.action = as.character(data$rnn.action)
data = subset(data, T)
require(ggplot2)

library("RColorBrewer")
cc = brewer.pal(n = 8, name = "Set1")

ggplot() +
# geom_ribbon(data = subset(data, T), aes(x=X, ymax=pol0), fill="green", ymin=0) +
geom_vline(data = subset(data, learner.state == "[0. 1.]"), aes(xintercept= X), size=0.1, linetype = "longdash") +
# geom_line(data = subset(data, learner.state == "[0. 1.]"), aes(x=X, y=pol0), color=cc[1]) +
geom_point(data = subset(data, learner.state == "[0. 1.]"), aes(x=X, y=pol0), colour=cc[1],shape=21, fill=cc[1]) +
geom_point(data = subset(data, learner.state == "[1. 0.]"), aes(x=X, y=pol0), colour=cc[1], size=0.1, shape=21, fill=cc[1]) +
# geom_point(data =  subset(data, learner.state == "[1. 0.]"), aes(x=X, y = 0.2 - 0.5), fill = "blue", show.legend=FALSE, size=2, alpha=0.5, shape=22) +
geom_point(data =  subset(data, learner.state == "[0. 1.]"), aes(x=X, y = 1.3),
                                  fill = "blue", show.legend=FALSE, size=2, shape=25, color="blue") +

# geom_point(data =  subset(data, learner.action == "[1 0]"), aes(x=X, y = 1.2), fill = "black", show.legend=FALSE, size=1, shape=22) +
scale_color_manual(name="action", values=c("red", "blue")) +
# geom_hline(yintercept= 1- 0.5) +
# geom_hline(yintercept= 0.5- 0.5) +
# geom_hline(yintercept= 1.5- 0.5) +
theme_bw() +
# ylim(0- 0.5, NA) +
scale_y_continuous(breaks=c(0, 0.5, 1), limits=c(0, 1.3), expand = c(0.1, 0.1)) +
theme(axis.title.y=element_blank(),
        # axis.text.y=element_blank(),
        axis.ticks.y=element_blank())+
        theme(text = element_text(size=8)) +
        theme(axis.text=element_text(size=8)) +
theme(
  panel.grid.major.x = element_blank(),
  panel.grid.minor = element_blank()) +
        facet_grid(ev ~ .)+
        theme(
  strip.background = element_blank(),
  strip.text = element_blank(),
  # axis.line.x = element_line(color="black", size = 0.5),
  # panel.border = element_blank(),
) + xlab("trial")

require(cairo_pdf)
ggsave("../../doc/graphs/gonog_reward_plan2.pdf", width=16, height=3, unit="cm")

############ policy analysis ########
rm(list = ls())
#RNN vs ADV
dd = list()
indx = 1
for (i in c(0:100)){
  data = read.csv(paste0('../../nongit/archive/gonogo/state-reg/static-sims-1/gonog_sim_2000//RL_gonogo_2layers_0.01ent_256units_lof0.0/events_', i,".csv"))
  data$ev = i
  data$trail = seq(1:350)
  dd[[indx]] =data
  indx = indx + 1
}

require(data.table)
data = rbindlist(dd)
head(data)

data$s = (data$learner.state == '[1. 0.]') * 1
data$trial_g = floor((data$trail  - 0.0001) / 50) *  50

require(plyr)
dd2_group = ddply(subset(data, T), c("ev", "trial_g"),
            function(x){
              data.frame(cor = mean(x$s))
            }
)
nrow(dd2_group)

library("RColorBrewer")
require(ggplot2)
cc = brewer.pal(n = 8, name = "Set1")
dd2_group$cor = dd2_group$cor * 100
ggplot() +
  geom_bar(data = dd2_group, aes(x = as.factor(trial_g), y = cor),
  position = "dodge", stat = "summary", fun.y = "mean", fill=cc[3]
) +
stat_summary(data = dd2_group, aes(x = as.factor(trial_g), y = cor),
fun.data = mean_cl_normal, geom="linerange", colour="black",
             position=position_dodge(0.9),  fun.args = list(mult = 1), size=0.5) +
scale_x_discrete(labels=c("1-50", "50-100", "100-150", "150-200", "200-250", "250-300", "300-350")) +
scale_fill_brewer(palette="Set1") +
xlab("trial") +
ylab("%go trials") +
theme_bw() +
theme(legend.title = element_blank())+
theme(axis.text.x = element_text(angle = 90))

##### just for testing ######
require(plotrix)
std.error(subset(dd2_group, trial_g == 50)$cor)
std.error(adv_rew$X0)
#############################


ggsave("../../doc/graphs/go-precet.pdf", width=5, height=5.7, unit="cm")
