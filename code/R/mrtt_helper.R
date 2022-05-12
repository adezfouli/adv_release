rm(list = ls())
read_mturk_res = function(data_path){
  dirs = list.dirs(data_path, full.names = TRUE, recursive = FALSE)

  require(ramify)
  output = list()
  index = 1
  for (d in dirs){
    f = paste0(d, '/data/output.csv')
    df = read.csv(f)
    df$repay = as.numeric(gsub("\\[|\\]", "", df$repay))
    df$adv.action = as.numeric(gsub("\\[|\\]", "",df$adv.action))
    df$id = d
    df$trial = c(1:10)
    df$trustee_earn =  3 * df$investment - df$repay
    df$repay_precent =  100 * df$repay / (df$investment * 3 + 0.0000001)
    df$learner.discr = floor(clip(df$investment - 0.0001, 0, 1000) / 4)
    df$investor_earn = (20 - df$investment) + df$repay
    output[[index]] = df
    index = index + 1
  }
  require(data.table)
  all_s = as.data.table(as.data.frame(rbindlist(output)))
  all_s[, prev_repay_precent := shift(repay_precent, 1), by=.(id)]
  all_s$prev_repay_precent_disc = cut(all_s$prev_repay_precent, 5, method="length", na.omit=FALSE)
  all_s$investement.discr = floor(clip(all_s$investment - 0.0001, 0, 1000) / 4) + 1
  all_s
}

read_training_data = function(data_path){
  data = read.csv(data_path)
  data$repay_precent = data$reward / (data$action * 3) * 100
  data$repay = data$reward
  data = as.data.table(data)
  data$trial = ave(data$action, data$id, FUN = seq_along)
  data[, prev_repay_precent := shift(repay, 1), by=.(id)]
  data$investment = data$action
  data$prev_repay_precent_disc = cut(data$prev_repay_precent, 5, method="length", na.omit=FALSE)
  data$investement.discr = floor(clip(data$investment - 0.0001, 0, 1000) / 4) + 1
  data$trustee_earn =  3 * data$investment - data$repay
  data$investor_earn = (20 - data$investment) + data$repay
  data
}

read_sim_data = function(data_path, sim_count, adv_action_count=5){
  dd = list()
  indx = 1
  for (i in c(0:sim_count)){

    # for human
    data = read.csv(paste0(data_path, '/events_', i,".csv"))
    data$sbj = i
    data$trial = (data$X + 1 )
    data$percent_return = (data$adv.action * (100 / (adv_action_count - 1)))
    data$repay = data$learner.reward
    data$learner.discr = NA
    data$learner.discr[data$learner.action == '[0 0 0 0 1]'] = 5
    data$learner.discr[data$learner.action == '[0 0 0 1 0]'] = 4
    data$learner.discr[data$learner.action == '[0 0 1 0 0]'] = 3
    data$learner.discr[data$learner.action == '[0 1 0 0 0]'] = 2
    data$learner.discr[data$learner.action == '[1 0 0 0 0]'] = 1
    dd[[indx]] = data
    indx = indx + 1
  }

  require(data.table)
  all_d = as.data.frame(rbindlist(dd))
  all_d$repay_precent = all_d$percent_return
  all_d$investment = all_d$learner.action.cont
  all_d$id = all_d$sbj
  all_d = as.data.table(all_d)
  all_d[, prev_repay_precent := shift(repay_precent, 1), by=.(id)]
  all_d$prev_repay_precent_disc = cut(all_d$prev_repay_precent, 5, method="length", na.omit=FALSE)
  all_d$investement.discr = floor(clip(all_d$investment - 0.0001, 0, 1000) / 4) + 1
  all_d$trustee_earn =  3 * all_d$investment - all_d$repay
  all_d$investor_earn = (20 - all_d$investment) + all_d$repay
  all_d = as.data.frame(all_d)
}

plot_prev_repay = function(data){
  require(plyr)
  require(ggplot2)
  library("RColorBrewer")
  cc = brewer.pal(n = 8, name = "Set1")

  data = ddply(data, c("id", "prev_repay_precent_disc"), function(x){data.frame(investment=mean(x$investment))})
  ggplot(subset(data, !(data$prev_repay_precent_disc == "NA")), aes(y = investment, x =prev_repay_precent_disc)) +
  geom_bar(
  stat = "summary", fun.y = "mean", fill=cc[[2]]) +
  stat_summary(fun.data = mean_cl_normal, geom="linerange", colour="black",
               position=position_dodge(.9),  fun.args = list(mult = 1), size=0.2) +
  ylab("investment in current trial") +
  xlab("%repayment in \n previous trial") +
  scale_x_discrete(labels= c("0-20", "20-40", "40-60", "60-80", "80-100"))  +
  theme_bw() +
  scale_fill_brewer(name = "", palette="Set1") +
  guides(fill = guide_legend(keywidth = 0.5, keyheight = 3.0))+
  theme(text = element_text(size=8)) +
  theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1))

}

plot_adv_strategy = function(data){
  data_tile = ddply(subset(data, investment != 0), c("trial", "investement.discr"), function(x){data.frame(adv_mean=mean(x$repay_precent))})
  ggplot() +
  geom_tile(data=data_tile,
    aes(x=as.factor(trial), y = as.factor(investement.discr * 20), fill= adv_mean),
  ) +
  xlab("") +
  ylab("") +
  scale_fill_continuous(type = "viridis", name="advesary action (% repayment)",
    limits=c(0, 75), breaks = c(0, 25, 50, 75)) +
  theme_bw() +
  theme(legend.position="bottom", legend.box = "horizontal") +
  theme(text = element_text(size=8))
}


plot_investment_trial = function(data){
  data_i = ddply(data, c("trial", "id", "cat"), function(x){data.frame(investment=mean(x$investment))})
  library("RColorBrewer")
  cc = brewer.pal(n = 8, name = "Set1")

  ggplot(subset(data_i, T), aes(y = investment, x =trial, fill=cat)) +
  #  stat_summary(fun.y = "mean", geom = "bar", position = position_dodge(), fill=col) +
  geom_bar(stat = "summary", fun.y = "mean",position = "dodge") +
    stat_summary(fun.data = mean_cl_normal, geom="linerange", colour="black",
                 position=position_dodge(.9),  fun.args = list(mult = 1), size=0.2) +
    ylab("investment") +
    xlab("trial") +
    theme_bw() +
    scale_fill_brewer(name = "", palette="Set1", limits=c("MAX","", "FAIR")) +
    scale_x_discrete(limits = c(0, 5, 10)) +
    coord_cartesian(ylim=c(0, 16))+
    theme(text = element_text(size=8))+
    guides(fill = guide_legend(keywidth = 0.5, keyheight = 3.0))

}


plot_repay_trial = function(data){
  data_r = ddply(subset(data, investment != 0), c("trial", "id", "cat"), function(x){data.frame(repay=mean(x$repay_precent))})
  library("RColorBrewer")
  cc = brewer.pal(n = 8, name = "Set1")
  ggplot(subset(data_r, T), aes(y = repay, x =trial, fill=cat, group=cat)) +
  #  stat_summary(fun.y = "mean", geom = "bar", position = position_dodge(), fill=col) +
    geom_bar(stat = "summary", fun.y = "mean",position = "dodge") +
    stat_summary(fun.data = mean_cl_normal, geom="linerange", colour="black",
                 position=position_dodge(.9),  fun.args = list(mult = 1), size=0.2) +
    ylab("%repayment") +
    xlab("trial") +
    theme_bw() +
    scale_fill_brewer(name = "", palette="Set1", limits=c("MAX","", "FAIR")) +
    scale_x_discrete(limits = c(0, 5, 10)) +
    coord_cartesian(ylim=c(0, 60))+
    theme(text = element_text(size=8))+
    guides(fill = guide_legend(keywidth = 0.5, keyheight = 3.0))
}


plot_trustee_earn = function(data, x_order){
  data_r = ddply(subset(data, T), c("id", "group", "cat"), function(x){data.frame(trustee_earn=sum(x$trustee_earn))})
  library("RColorBrewer")
  cc = brewer.pal(n = 8, name = "Set1")
  ggplot(subset(data_r, T), aes(y = trustee_earn, x =group, fill=cat)) +
  #  stat_summary(fun.y = "mean", geom = "bar", position = position_dodge(), fill=col) +
    geom_bar(stat = "summary", fun.y = "mean") +
    stat_summary(fun.data = mean_cl_normal, geom="linerange", colour="black",
                 position=position_dodge(.9),  fun.args = list(mult = 1), size=0.2) +
    ylab("trustee earning") +
    xlab("") +
    theme_bw() +
    scale_fill_brewer(name = "", palette="Set1", limits=c("MAX","RND", "FAIR")) +
    scale_x_discrete(limits = x_order) +
    coord_cartesian(ylim=c(0, 320))+
    # blk_theme_grid_hor(legend_position ="none", margins = c(1,1,1,1), rotate_x = F) +
    guides(fill = guide_legend(keywidth = 0.5, keyheight = 3.0))+
    theme(legend.position="none")+
    theme(text = element_text(size=8))+
    theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1))
}


plot_investor_earn = function(data, x_order){
  data_r = ddply(subset(data, T), c("id", "group", "cat"), function(x){data.frame(investor_earn=sum(x$investor_earn))})
  library("RColorBrewer")
  cc = brewer.pal(n = 8, name = "Set1")
  ggplot(subset(data_r, T), aes(y = investor_earn, x =group, fill=cat)) +
  #  stat_summary(fun.y = "mean", geom = "bar", position = position_dodge(), fill=col) +
    geom_bar(stat = "summary", fun.y = "mean") +
    stat_summary(fun.data = mean_cl_normal, geom="linerange", colour="black",
                 position=position_dodge(.9),  fun.args = list(mult = 1), size=0.2) +
    ylab("investor earning") +
    xlab("") +
    theme_bw() +
    scale_fill_brewer(name = "", palette="Set1", limits=c("MAX","RND", "FAIR")) +
    scale_x_discrete(limits = x_order) +
    coord_cartesian(ylim=c(0, 320))+

    # blk_theme_grid_hor(legend_position ="none", margins = c(1,1,1,1), rotate_x = F) +
    guides(fill = guide_legend(keywidth = 0.5, keyheight = 3.0))+
    theme(legend.position="none") +
    theme(text = element_text(size=8))+
    theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1))
}

plot_fairness = function(data, x_order){
  data_r = ddply(subset(data, T), c("id", "group", "cat"), function(x){data.frame(diffs=abs(sum(x$trustee_earn - x$investor_earn)))})
  library("RColorBrewer")
  cc = brewer.pal(n = 8, name = "Set1")
  ggplot(subset(data_r, T), aes(y = diffs, x =group, fill=cat)) +
  #  stat_summary(fun.y = "mean", geom = "bar", position = position_dodge(), fill=col) +
    geom_bar(stat = "summary", fun.y = "mean") +
    stat_summary(fun.data = mean_cl_normal, geom="linerange", colour="black",
                 position=position_dodge(.9),  fun.args = list(mult = 1), size=0.2) +
    ylab("earning gap") +
    xlab("") +
    theme_bw() +
    scale_fill_brewer(name = "", palette="Set1", limits=c("MAX","RND", "FAIR")) +
    scale_x_discrete(limits = x_order) +
    # blk_theme_grid_hor(legend_position ="none", margins = c(1,1,1,1), rotate_x = F) +
    guides(fill = guide_legend(keywidth = 0.5, keyheight = 3.0))+
    theme(legend.position="none")+
    theme(text = element_text(size=8))+
    theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1))
}

#exporting random data
# mturk_d = read_mturk_res("../../nongit/archive/mrtt/Read/human_data/r25-adv-rnd/subject_data/")
# mturk_sumr = mturk_d[, c("id", "repay", "investment")]
# mturk_sumr$action = mturk_sumr$investment
# mturk_sumr$reward = floor(mturk_sumr$repay)
# mturk_sumr$repay = NULL
# mturk_sumr$investment = NULL
#
# write.csv(mturk_sumr, "../../data/MRTT/r25_rnd.csv")
###############################

# max earning models

## Read's data - not used in the paper
# training_d = read_training_data('../../data/MRTT/data_Read_summ.csv')
# mturk_d_adv = read_mturk_res('../../nongit/archive/mrtt/Read/human_data/r1-adv-Read/subject_data/')

###training on Read's adv
# sim_d = read_sim_data('../../nongit/archive/mrtt/Read/total earn/RL_rmrtt_dqn_vec_sim/rmrtt_sim_1000000//RL_nc_dqn_buf_200000_eps_0.2_lr_0.0001/', 14999)


mturk_d_rnd = read_mturk_res('../../nongit/archive/mrtt/RND/human_data/r25-adv-rnd/subject_data/')
mturk_d_rnd$group = "SBJ vs RND"
mturk_d_rnd$cat = "RND"

plot_prev_repay(mturk_d_rnd)
ggsave("../../doc/graphs/mrtt_rnd_plot_prev_repay.pdf", width=2.9, height=5.6, unit="cm", useDingbats=FALSE)

# RND data
mturk_d_adv_max = read_mturk_res('../../nongit/archive/mrtt/RND/human_data/r-876-adv-adv/subject_data/')
mturk_d_adv_max$group = "SBJ vs MAX ADV"
mturk_d_adv_max$cat = "MAX"

sim_d_adv_max = read_sim_data('../../nongit/archive/mrtt/RND/selected/RL_nc_dqn_buf_200000_eps_0.2_lr_0.001_sim/', 14999)
sim_d_adv_max$group = "LRN vs MAX ADV"
sim_d_adv_max$cat = "MAX"

ggsave("../../doc/graphs/mrtt_rnd_plot_trustee_earn.pdf", width=10, height=10, unit="cm", useDingbats=FALSE)

plot_adv_strategy(sim_d_adv_max)
ggsave("../../doc/graphs/mrtt_sim_plot_adv_strategy.pdf", width=4, height=4, unit="cm", useDingbats=FALSE)

plot_adv_strategy(mturk_d_adv_max)
ggsave("../../doc/graphs/mrtt_mturk_plot_adv_strategy.pdf", width=10, height=10, unit="cm", useDingbats=FALSE)

plot_repay_trial(mturk_d_adv_max)
ggsave("../../doc/graphs/mrtt_mturk_plot_repay_trial.pdf", width=10, height=10, unit="cm", useDingbats=FALSE)


plot_investment_trial(mturk_d_adv_max)
ggsave("../../doc/graphs/mrtt_mturk_plot_investment_trial.pdf", width=10, height=10, unit="cm", useDingbats=FALSE)

# statistics
# b = ddply(data, "id", function(x){sum(x$trustee_earn)})
# mean(b$V1)
# nrow(b)
#
# aa = ddply(mturk_d_adv, "id", function(x){
#   abs(sum(x$trustee_earn - x$investor_earn))
# })
#
# mean(aa$V1)


# fair adv
sim_d_adv_fair = read_sim_data('../../nongit/archive/mrtt/RND-fair/selected//RL_nc_dqn_buf_400000_eps_0.2_lr_1e-05_sim/', 14999, 5)
sim_d_adv_fair$group = "LRN vs FAIR ADV"
sim_d_adv_fair$cat = "FAIR"

mturk_d_adv_fair = read_mturk_res('../../nongit/archive/mrtt/RND-fair/human_data/r12-adv/subject_data/')
mturk_d_adv_fair$group = "SBJ vs FAIR ADV"
mturk_d_adv_fair$cat = "FAIR"

plot_adv_strategy(sim_d_adv_fair)
ggsave("../../doc/graphs/mrtt_sim_plot_adv_strategy-fair.pdf", width=10, height=10, unit="cm", useDingbats=FALSE)

plot_adv_strategy(mturk_d_adv_fair)
ggsave("../../doc/graphs/mrtt_sim_plot_adv_strategy-fair.pdf", width=10, height=10, unit="cm", useDingbats=FALSE)


alld_rnd = rbind(mturk_d_adv_max[,c("id", "repay_precent", "group", "investment", "cat", "trial")],
                mturk_d_adv_fair[,c("id", "repay_precent", "group", "investment", "cat", "trial")]
)

plot_repay_trial(alld_rnd)
ggsave("../../doc/graphs/mrtt_mturk_plot_repay_trial-fair.pdf", width=7.5, height=3, unit="cm", useDingbats=FALSE)

plot_investment_trial(alld_rnd)
ggsave("../../doc/graphs/mrtt_mturk_plot_investment_trial-fair.pdf", width=7.5, height=3, unit="cm", useDingbats=FALSE)

alld_rnd = rbind(mturk_d_adv_max[,c("id", "trustee_earn", "group", "investor_earn", "cat")],
                sim_d_adv_max[,c("id", "trustee_earn", "group", "investor_earn", "cat")],
                mturk_d_rnd[,c("id", "trustee_earn", "group", "investor_earn", "cat")],
                sim_d_adv_fair[,c("id", "trustee_earn", "group", "investor_earn", "cat")],
                mturk_d_adv_fair[,c("id", "trustee_earn", "group", "investor_earn", "cat")]
)

plot_fairness(alld_rnd, c(mturk_d_rnd$group[1],
                              sim_d_adv_fair$group[1],
                              mturk_d_adv_fair$group[1],
                              sim_d_adv_max$group[1],
                              mturk_d_adv_max$group[1]
                            ))
ggsave("../../doc/graphs/mrtt_sim_plot_plot_fairness.pdf", width=3, height=6, unit="cm", useDingbats=FALSE)


alld_rnd = rbind(mturk_d_adv_max[,c("id", "trustee_earn", "group", "cat")],
                mturk_d_adv_fair[,c("id", "trustee_earn", "group", "cat")],
                mturk_d_rnd[,c("id", "trustee_earn", "group", "cat")],
                sim_d_adv_max[,c("id", "trustee_earn", "group", "cat")],
                sim_d_adv_fair[,c("id", "trustee_earn", "group", "cat")]
)

plot_trustee_earn(alld_rnd, c(mturk_d_rnd$group[1],
                              sim_d_adv_fair$group[1],
                              mturk_d_adv_fair$group[1],
                              sim_d_adv_max$group[1],
                              mturk_d_adv_max$group[1]
                            ))

ggsave("../../doc/graphs/mrtt_sim_plot_plot_trutee_earn.pdf", width=3, height=6, unit="cm", useDingbats=FALSE)


alld_rnd = rbind(mturk_d_adv_max[,c("id", "investor_earn", "group", "cat")],
                mturk_d_adv_fair[,c("id", "investor_earn", "group", "cat")],
                mturk_d_rnd[,c("id", "investor_earn", "group", "cat")],
                sim_d_adv_max[,c("id", "investor_earn", "group", "cat")],
                sim_d_adv_fair[,c("id", "investor_earn", "group", "cat")]
)

plot_investor_earn(alld_rnd, c(mturk_d_rnd$group[1],
                              sim_d_adv_fair$group[1],
                              mturk_d_adv_fair$group[1],
                              sim_d_adv_max$group[1],
                              mturk_d_adv_max$group[1]
                            ))

ggsave("../../doc/graphs/mrtt_sim_plot_plot_investor_earn.pdf", width=3, height=6, unit="cm", useDingbats=FALSE)


###########
aa = read_mturk_res('../../nongit/archive/mrtt/RND-fair/human_data/r2-adv/subject_data/')

max(ddply(aa, "id", function(x){sum(x$investor_earn)})$V1)


######## number of subjects in random condition
mturk_d_adv_fair = read_mturk_res('../../nongit/archive/mrtt/RND/human_data/r12-adv/subject_data/')

length(unique(mturk_d_adv_fair$id))
length(unique(mturk_d_adv_max$id))

adv_fair = ddply(mturk_d_adv_fair, "id", function(x){data.frame(trustee_earn = sum(x$trustee_earn), gap = abs(sum(x$trustee_earn) - sum(x$investor_earn)))})
adv_max = ddply(mturk_d_adv_max, "id", function(x){data.frame(trustee_earn = sum(x$trustee_earn), gap = abs(sum(x$trustee_earn) - sum(x$investor_earn)))})
adv_rnd = ddply(mturk_d_rnd, "id", function(x){data.frame(trustee_earn = sum(x$trustee_earn), gap = abs(sum(x$trustee_earn) - sum(x$investor_earn)))})
mean(adv_fair$trustee_earn)
mean(adv_max$trustee_earn)
mean(adv_rnd$trustee_earn)

wilcox.test(adv_fair$trustee_earn, adv_max$trustee_earn)
wilcox.test(adv_rnd$trustee_earn, adv_max$trustee_earn)

mean(adv_fair$gap)
mean(adv_max$gap)
mean(adv_rnd$gap)

wilcox.test(adv_fair$gap, adv_max$gap)
wilcox.test(adv_rnd$gap, adv_fair$gap)




for (x in c(5, 6, 7, 8
    , 15, 16, 18)){
  p = ((20 + 2*x) / (6*x))
  print(1-p)
}

data_tile = ddply(subset(mturk_d_adv_fair, investment != 0 & investement.discr==3), 
  c("trial", "investement.discr"), function(x){data.frame(adv_mean=mean(x$repay_precent))})
