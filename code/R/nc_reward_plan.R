
#RNN vs ADV
dd = list()
indx = 1
for (i in c(1, 58)){

  # for human
  data = read.csv(paste0('../../nongit/archive/nc/human/run-cv/RL_human_dqn_vec_sim_sample/nc_sim_600000/RL_nc_dqn_buf_400000_eps_0.1_lr_0.0001/events_', i,".csv"))
  pol = read.csv(paste0("../../nongit/archive/nc/human/run-cv/policies//policies_", i,".csv"))

  # for ql
  # data = read.csv(paste0("../../nongit/archive/nc/ql/run-cv/RL_ql_dqn_vec_sim_rnn/nc_sim_500000/RL_nc_dqn_buf_400000_eps_0.01_lr_0.001/events_", i,".csv"))
  # pol = read.csv(paste0("../../nongit/archive/nc/ql/run-cv/policies/policies_", i,".csv"))


  data$ev = i
  data$pol0 = NA
  data$pol1 = NA
  data$pol0 = pol$X0
  data$pol1 = pol$X1
  dd[[indx]] =data
  indx = indx + 1
}

require(data.table)
data = rbindlist(dd)
action_levels = levels(data$rnn.action)
data$rnn.action = as.character(data$rnn.action)

data$ev2 = factor(data$ev, levels=rev(levels(as.factor(data$ev))))
require(ggplot2)
ggplot() +
scale_color_manual(name="action", values=c("red", "blue")) +
geom_ribbon(data = subset(data, T), aes(x=X, ymin=0.5, ymax=pol0), fill="green", alpha=0.5) +
geom_segment(data = subset(data, r1 == 1), aes(x=X, xend=X, y=0.5, yend=1), show.legend=FALSE, color="blue") +
geom_segment(data = subset(data, r2 == 1), aes(x=X, xend=X, y=0.5, yend=0), show.legend=FALSE, color="red") +
scale_y_continuous(breaks=c(0, 0.5, 1), limits = c(0,1), expand = c(0.1, 0.1)) +
scale_x_continuous(expand = c(0.02, 0.02)) +
geom_point(data = data, aes(x=X,
  color=as.factor(rnn.action),
  y = 0.5), show.legend=FALSE, size=1) +
theme_bw() +
theme(
        axis.title.y=element_blank(),
        # axis.text.y=element_blank(),
        # axis.ticks.y=element_blank()
      )+
        theme(text = element_text(size=8)) +
        theme(axis.text=element_text(size=8)) +
        theme(
          panel.grid.major.x = element_blank(),
        panel.grid.minor = element_blank()
      ) +
        facet_grid(ev2 ~ .)+
        theme(
  strip.background = element_blank(),
  strip.text = element_blank()
) + xlab("trial")

ggsave("../../doc/graphs/reward_plan_temp.pdf", width=16, height=5, unit="cm", useDingbats=FALSE)

#Human vs ADV

learner_path = '../models/archive/learner/gonogo/onto/learner_go_nogo_cells_6/model-7200.h5'
base_dirs = '/scratch1/dez004/results_gonogo_onto_6cells/'
