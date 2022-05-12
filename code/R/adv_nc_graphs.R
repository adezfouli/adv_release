rm(list = ls())

adv_rew = read.csv('../../nongit/archive/nc/human/run-cv/RL_human_dqn_vec_sim/nc_sim_600000//RL_nc_dqn_buf_400000_eps_0.1_lr_0.0001/adv_reward_.csv')
dirs1 = list.files(path = "../../nongit/archive/nc/human/run-cv/human_data_dqn_vec/random_0/", full.names = TRUE, recursive = FALSE)

# dirs = append(dirs1)

rews = list()
index = 1
for (d in dirs1){
  s_d = read.csv(d)
  # print(nrow(s_d))
  if(nrow(s_d) == 101){
    s_d = s_d[1:100,]
  }
  if(nrow(s_d) == 100){
    rews[[index]] = data.frame(id =d, bias_actions = sum(as.character(s_d$is_biased_choice) == " true"),
                                rewarded_choice = sum(as.character(s_d$observed_reward) == "1"),
                                left_choice = sum(as.character(s_d$side_choice) == " LEFT"),
                                right_choice = sum(as.character(s_d$side_choice) == " RIGHT")

  )
    index = index + 1
  }
}

require(data.table)
rews = rbindlist(rews)
nrow(rews)
mean(subset(rews, T)$bias_actions)

advs = rbind(data.frame(value = unlist(rews$bias_actions), system = "ADV vs\n SBJ"),
          data.frame(value = unlist(adv_rew$X0), system = "ADV vs\n LRN"))


##### just for testing ######
require(plotrix)
std.error(rews$bias_actions)
std.error(adv_rew$X0)
#############################

mean(subset(advs, system =="ADV vs\n SBJ")$value)
mean(subset(advs, system =="ADV vs\n LRN")$value)
nrow(subset(advs, system =="ADV vs\n SBJ"))

wilcox.test(subset(advs, system =="ADV vs\n SBJ")$value - 50, alternative="greater")

require(ggplot2)
ggplot() +
geom_bar(data=advs, aes(x=system, y=value, fill=system),
        position = "dodge", stat = "summary", fun.y = "mean"
) +
geom_point(data=advs, aes(x=as.numeric(system) + 0.2, y=value), alpha=1.0, size=0.1, stroke = 0.03, position=position_jitter(width=0.1, height = 0.0)) +
stat_summary(data=advs, aes(x=system, y=value, fill=system), fun.data = mean_cl_normal, geom="linerange", colour="black",
             position=position_dodge(.9),  fun.args = list(mult = 1), size=0.5) +

# geom_point(data=advs, aes(x = system + 0.1, y=value), color="red4",
#         position=position_jitterdodge(dodge.width=0.9, jitter.width=0.2), alpha=1.0, size=0.5, stroke = 0.05) +

scale_fill_brewer(palette="Set1") +
ylim(c(0, 100)) +
theme_bw() +
ylab("bias") +
theme(legend.position="None")  +
theme(text = element_text(size=8)) +
theme(axis.text=element_text(size=8), panel.grid.major.x = element_blank()) +
geom_hline(yintercept=50, linetype="dashed", color = "black") +
xlab("")

ggsave("../../doc/graphs/human_RNN.pdf", width=4, height=5, unit="cm")

wilcox.test(subset(advs, system =="ADV vs\n SBJ")$value - 50, alternative="greater")


########## for q-learning graphs ######################
rm(list = ls())

adv_rew = read.csv('../../nongit/archive/nc/ql/run-cv/sims/RL_ql_dqn_vec_sim_ql//nc_sim_500000//RL_nc_dqn_buf_400000_eps_0.01_lr_0.001/adv_reward_.csv')
adv_rew2 = read.csv('../../nongit/archive/nc/ql/run-cv/sims/RL_ql_dqn_vec_sim_rnn//nc_sim_500000//RL_nc_dqn_buf_400000_eps_0.01_lr_0.001/adv_reward_.csv')


require(data.table)
advs = rbind(data.frame(value = unlist(adv_rew$X0), system = "ADV vs\n QL"),
          data.frame(value = unlist(adv_rew2$X0), system = "ADV vs\n LRN"))


require(ggplot2)
require(ggplot2)
ggplot() +
geom_bar(data=advs, aes(x=system, y=value, fill=system),
        position = "dodge", stat = "summary", fun.y = "mean"
) +
geom_point(data=advs, aes(x=as.numeric(system) + 0.2, y=value), alpha=1.0, size=0.1, stroke = 0.03, position=position_jitter(width=0.1, height = 0.0)) +
stat_summary(data=advs, aes(x=system, y=value, fill=system), fun.data = mean_cl_normal, geom="linerange", colour="black",
             position=position_dodge(.9),  fun.args = list(mult = 1), size=0.5) +

# geom_point(data=advs, aes(x = system + 0.1, y=value), color="red4",
#         position=position_jitterdodge(dodge.width=0.9, jitter.width=0.2), alpha=1.0, size=0.5, stroke = 0.05) +

scale_fill_brewer(palette="Set1") +
ylim(c(0, 100)) +
theme_bw() +
ylab("bias") +
theme(legend.position="None")  +
theme(text = element_text(size=8)) +
theme(axis.text=element_text(size=8)) +
geom_hline(yintercept=50, linetype="dashed", color = "black") +
xlab("")

ggsave("../../doc/graphs/QRL_RNN.pdf", width=4, height=5, unit="cm")


mean(subset(advs, system =="ADV vs\n QL")$value)
mean(subset(advs, system =="ADV vs\n LRN")$value)

std.error(subset(advs, system =="ADV vs\n QL")$value)
std.error(subset(advs, system =="ADV vs\n LRN")$value)


wilcox.test(subset(advs, system =="ADV vs\n QL")$value - 50, alternative="greater")
