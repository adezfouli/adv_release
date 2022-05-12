
get_best = function(dir_s){
  output = list()
  index = 1

  for (pth in dir_s){
    dirs = list.dirs(path = pth, full.names = TRUE, recursive = FALSE)
    for (d in dirs){
      adv_rew = read.csv(paste0(d, '/adv_reward_.csv'))
      adv_rew$model = d
      output[[index]] = adv_rew
      index = index + 1
    }
  }

  require(data.table)
  all_d = as.data.frame(rbindlist(output))


  require(plyr)
  mms = ddply(subset(all_d, T), "model", function(x){data.frame(rew = mean(x$X0), v = var(x$X0))})
  mms = as.data.frame(mms)
  print(mms[order(mms$rew),])
  print(mms[which.max(mms$rew),])

}

dir_lists = list()
i = 1
for (it in as.character(c('50000', '200000', '500000', '1000000'))){
  # dir_lists[[i]] = paste0("../../nongit/archive/mrtt/RND-fair/RL_rmrtt_dqn_RND_fair_max_sim/rmrtt_sim_", it, "/" )
  dir_lists[[i]] = paste0("../../nongit/archive/mrtt/RND/RL_rmrtt_dqn_RND_sim/rmrtt_sim_", it, "/" )
  i = i + 1
}

dir_s = dir_lists
get_best(dir_lists)


# for hyper graph
output = list()
index = 1

for (buf in c('200000', '400000')){
  for (lr in c('0.001', '0.0001', '1e-05')){
    for (eps in c('0.1', '0.2', '0.01')){
      adv_rew = read.csv(paste0('../../nongit/archive/mrtt/RND-fair/RL_rmrtt_dqn_RND_fair_max_sim/rmrtt_sim_500000//RL_nc_dqn_buf_', buf, '_eps_', eps, '_lr_', lr, '/adv_reward_.csv'))
      adv_rew$eps = eps
      adv_rew$buf = buf
      adv_rew$lr = lr
      adv_rew$X0 = -adv_rew$X0
      adv_rew$adversary = 'FAIR'
      output[[index]] = adv_rew
      index = index + 1
  }
}
}

for (buf in c('200000', '400000')){
  for (lr in c('0.001', '0.0001', '1e-05')){
    for (eps in c('0.1', '0.2', '0.01')){
      adv_rew = read.csv(paste0('../../nongit/archive/mrtt/RND/RL_rmrtt_dqn_RND_sim/rmrtt_sim_500000//RL_nc_dqn_buf_', buf, '_eps_', eps, '_lr_', lr, '/adv_reward_.csv'))
      adv_rew$eps = eps
      adv_rew$buf = buf
      adv_rew$lr = lr
      adv_rew$adversary = 'MAX'
      output[[index]] = adv_rew
      index = index + 1
  }
}
}


require(data.table)
all_d = as.data.frame(rbindlist(output))

# lr_labels = list(
#   `0.001' = "learning rate: 0.001",
#   '0.0001' = "learning rate: 0.0001",
#   '1e-05' = "learning rate: 1e-05"
# )

require(ggplot2)
ggplot() +
geom_bar(data=subset(all_d, adversary == 'MAX'), aes(x=eps, y=X0, fill=buf),
        position = "dodge", stat = "summary", fun.y = "mean"
) +
stat_summary(data=subset(all_d, adversary == 'MAX'), aes(x=eps, y=X0, fill=buf), fun.data = mean_cl_normal, geom="linerange", colour="black",
             position=position_dodge(.9),  fun.args = list(mult = 1), size=0.5) +

# geom_point(data=advs, aes(x = system + 0.1, y=value), color="red4",
#         position=position_jitterdodge(dodge.width=0.9, jitter.width=0.2), alpha=1.0, size=0.5, stroke = 0.05) +

scale_fill_brewer(palette="Set1", name="buffer") +
theme_bw() +
theme(text = element_text(size=8)) +
theme(axis.text=element_text(size=8), panel.grid.major.x = element_blank()) +
xlab("epsilon")+
ylab("bias") +
coord_cartesian(ylim=c(250, 310))+
facet_grid(lr ~ ., labeller = label_both)

ggsave("../../doc/graphs/max_hyper.pdf", width=7, height=10, unit="cm")
