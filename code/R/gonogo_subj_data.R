rm(list = ls())

####### for geting stats from trainin data #######################
dirs1 = list.files(path = "../../nongit/archive/gonogo/state-reg/training-data/", full.names = TRUE, recursive = FALSE)

dirs = c(dirs1)

output = list()
index = 1
for (d in dirs){
  dd = read.csv(d)
  dd_prac = subset(dd, exp_stage == 'practice')
  dd = subset(dd, exp_stage == 'test')
  dd_prac = sum(dd_prac$correct == 'false')
  if (dd_prac > -1 ){
    output[[index]] = data.frame(id = d, total_sum = sum(dd$correct == 'false'))
    index = index + 1
}
}

require(data.table)
dd = rbindlist(output)
dr = data.frame(id = dd$id, total_sum = dd$total_sum, type="SJB vs\n RND")
nrow(dr)

quantile(dr$total_sum, .75)
mean(subset(dr, total_sum <= 31.75)$total_sum)
nrow(subset(dr, total_sum < 32))
mean(subset(dr, T)$total_sum)

######################
dirs1 = list.files(path = "../../nongit/archive/gonogo/state-reg/human_data/gonogo_data_1/", full.names = TRUE, recursive = FALSE)
dirs2 = list.files(path = "../../nongit/archive/gonogo/state-reg/human_data/gonogo_data_2/", full.names = TRUE, recursive = FALSE)


dirs = c(dirs1, dirs2)

output = list()
index = 1
for (d in dirs){
    dd = read.csv(d)
    dd_prac = subset(dd, exp_stage == 'practice')
    dd = subset(dd, exp_stage == 'test')
    if (dd$mode[1]== 'adv'){
    #   file.copy(d, "../../nongit/archive/gonogo/onto-cv/human_data/random_all/")
    dd_prac = sum(dd_prac$correct == 'false')
    if (dd_prac > -1 ){
      output[[index]] = data.frame(id = d, total_sum = sum(dd$correct == 'false'))
      index = index + 1
  }
}
}

require(data.table)
dd = rbindlist(output)
d1 = data.frame(id = dd$id, total_sum = dd$total_sum, type="SJB vs\n ADV")

x = 32
nrow(d1)
nrow(subset(d1, total_sum <x))
mean(subset(d1, total_sum < x)$total_sum)
mean(subset(d1, T)$total_sum)


wilcox.test(subset(dr, total_sum < x)$total_sum, subset(d1, total_sum < x)$total_sum)


data = rbind(d1, dr)
data = subset(data, total_sum < x)
require(ggplot2)

mean(subset(data, type=="SJB vs\n ADV")$total_sum)
mean(subset(data, type=="SJB vs\n RND")$total_sum)

require(plotrix)
std.error(subset(data, type=="SJB vs\n ADV")$total_sum)
std.error(subset(data, type=="SJB vs\n RND")$total_sum)

ggplot() +
# geom_boxplot(data=data, aes(x=type, y=total_sum, fill=type)
# ) +
geom_bar(data=data, aes(x=type, y=total_sum, fill=type),
        position = "dodge", stat = "summary", fun.y = "mean"
) +
# geom_point(data=data, aes(x=type, y=total_sum), size=1, stroke = 0.3, position=position_jitter(width=0.1, height = 0.0)) +
stat_summary(data=data, aes(x=type, y=total_sum, fill=total_sum), fun.data = mean_cl_normal, geom="linerange", colour="black",
             position=position_dodge(.9),  fun.args = list(mult = 1), size=0.5) +

# geom_point(data=advs, aes(x = system + 0.1, y=value), color="red4",
#         position=position_jitterdodge(dodge.width=0.9, jitter.width=0.2), alpha=1.0, size=0.5, stroke = 0.05) +

scale_fill_brewer(palette="Set1") +
theme_bw() +
ylab("#errors") +
theme(legend.position="None")  +
theme(text = element_text(size=8)) +
theme(axis.text=element_text(size=8), panel.grid.major.x = element_blank()) +
xlab("")

ggsave("../../doc/graphs/gono performance.pdf", width=4, height=5, unit="cm")


############### random vs adv in test time ##########
rm(list = ls())
dirs1 = list.files(path = "../../nongit/archive/gonogo/state-reg/human_data/gonogo_data_1/", full.names = TRUE, recursive = FALSE)
dirs2 = list.files(path = "../../nongit/archive/gonogo/state-reg/human_data/gonogo_data_2/", full.names = TRUE, recursive = FALSE)


dirs = c(dirs1, dirs2)

output = list()
index = 1
for (d in dirs){
    dd = read.csv(d)
    dd_prac = subset(dd, exp_stage == 'practice')
    dd = subset(dd, exp_stage == 'test')
    if (dd$mode[1]== 'adv'){
    #   file.copy(d, "../../nongit/archive/gonogo/onto-cv/human_data/random_all/")
    dd_prac = sum(dd_prac$correct == 'false')
    if (dd_prac > -1 ){
      output[[index]] = data.frame(id = d, total_sum = sum(dd$correct == 'false'))
      index = index + 1
  }
}
}

require(data.table)
dd = rbindlist(output)
d1 = data.frame(id = dd$id, total_sum = dd$total_sum, type="SJB vs\n ADV")

output = list()
index = 1
for (d in dirs){
    dd = read.csv(d)
    dd_prac = subset(dd, exp_stage == 'practice')
    dd = subset(dd, exp_stage == 'test')
    if (dd$mode[1]== 'random'){
    dd_prac = sum(dd_prac$correct == 'false')
    if (dd_prac > -1 ){
      output[[index]] = data.frame(id = d, total_sum = sum(dd$correct == 'false'))
      index = index + 1
  }
}
}

require(data.table)
dd = rbindlist(output)
d2 = data.frame(id = dd$id, total_sum = dd$total_sum, type="SBJ vs\nn RND (test)")

x = 32
nrow(d2)
quantile(d2$total_sum, 0.5)
nrow(subset(d2, total_sum <=x))
mean(subset(d2, total_sum <= x)$total_sum)
mean(subset(d2, T)$total_sum)

nrow(d1)
nrow(subset(d1, total_sum <=x))
mean(subset(d1, total_sum <= x)$total_sum)
mean(subset(d1, T)$total_sum)


wilcox.test(subset(d1, total_sum < x)$total_sum, subset(d2, total_sum < x)$total_sum)
wilcox.test(subset(d1, T)$total_sum, subset(d2, T)$total_sum)

######################### old codes ##############
dirs1 = list.files(path = "../../nongit/archive/gonogo/gonogo_human_data/onto/gonogo_data_random/", full.names = TRUE, recursive = FALSE)

# dirs3 = list.files(path = "../../nongit/archive/gonogo/onto-cv/human_data/gonogo_data_adv_3/", full.names = TRUE, recursive = FALSE)
# dirs4 = list.files(path = "../../nongit/archive/gonogo/onto-cv/human_data/gonogo_data_adv_4/", full.names = TRUE, recursive = FALSE)
# dirs2 = list.files(path = "../../nongit/archive/gonogo/onto-cv/human_data/gonogo_data_adv_2/", full.names = TRUE, recursive = FALSE)
# dirs3 = list.files(path = "../../nongit/archive/gonogo/onto-cv/human_data/gonogo_data_adv_3/", full.names = TRUE, recursive = FALSE)
# dirs1 = list.files(path = "../../nongit/archive/gonogo/onto-cv/human_data/gonogo_data_adv_1/", full.names = TRUE, recursive = FALSE)

dirs = c(dirs1)

output = list()
index = 1
for (d in dirs){
  dd = read.csv(d)
  dd_prac = subset(dd, exp_stage == 'practice')
  dd = subset(dd, exp_stage == 'test')
  dd_prac = sum(dd_prac$correct == 'false')
  if (dd_prac > -1){
  output[[index]] = data.frame(id = d, total_sum = sum(dd$correct == 'false'))
  index = index + 1
}
}

require(data.table)
dd = rbindlist(output)
d1 = data.frame(id = dd$id, total_sum = dd$total_sum, type="random")
nrow(d1)
mean(subset(d1, total_sum <= 175)$total_sum)
mean(subset(d1, T)$total_sum)

wilcox.test(subset(d1, total_sum <= 22)$total_sum, subset(d2, total_sum <=22)$total_sum)

wilcox.test(subset(d1, T)$total_sum, subset(d2, T)$total_sum)

########### for data from random  paper #################
# dirs = list.files(path = "../../nongit/archive/RL/gonogo/round2/gonogo_data/", full.names = TRUE, recursive = FALSE)
# dirs = list.files(path = "../../data/go-nogo/local/round2/raw/", full.names = TRUE, recursive = FALSE)
# dirs = list.files(path = "../../nongit/archive/gonogo_human_data/round2/adv - testing/", full.names = TRUE, recursive = FALSE)
dirs = list.files(path = "../../nongit/archive/gonogo/gonogo_human_data/onto/gonogo_data_random/", full.names = TRUE, recursive = FALSE)

output = list()
index = 1
for (d in dirs){
  dd = read.csv(d)
  dd = subset(dd, exp_stage == 'test')
  output[[index]] = data.frame(id = d, total_sum = sum(dd$correct == 'false'))
  index = index + 1
}

require(data.table)
dd = rbindlist(output)
d1 = data.frame(id = dd$id, total_sum = dd$total_sum, type="random")

mean(subset(d1, total_sum <= 22)$total_sum)

nrow(d1)


########### for data from AWS #################
dirs = list.files(path = "../../nongit/archive/gonogo_human_data/onto/gonogo_data_adv/", full.names = TRUE, recursive = FALSE)

output = list()
index = 1
for (d in dirs){
  dd = read.csv(d)
  dd = subset(dd, exp_stage == 'test')
  output[[index]] = data.frame(id = d, total_sum = sum(dd$correct == 'false'))
  index = index + 1
}
require(data.table)
dd = rbindlist(output)
d2 = data.frame(id = dd$id, total_sum = dd$total_sum, type="adv")
nrow(d2)


t.test(subset(d1, total_sum <= 22)$total_sum, subset(d2, total_sum <= 22)$total_sum, paired=FALSE)
t.test(subset(d1, T)$total_sum, subset(d2, T)$total_sum, paired=FALSE)

wilcox.test(subset(d1, total_sum <= 22)$total_sum, subset(d2, total_sum <= 22)$total_sum)



quantile(dd$total_sum, probs = c(0.05, 0.8))

# mean(subset(dd, T)$total_sum)
#
#
# nrow(dd)
# nrow(subset(dd, total_sum <= 22))
# mean(subset(dd, total_sum <= 22)$total_sum)
#
# mean(subset(dd, T)$total_sum)

############## for data from model simulations #######################
# d3 = read.csv("../../nongit/archive/RL/gonogo/sims_limit23/gonog_sim_25000/RL_gonogo_6cells_1layers_0.0005ent_128units/adv_reward_.csv")
# d3 = data.frame(id = d3$X, total_sum = d3$X0, type="adv vs RNN")

############## all data
require(ggplot2)
dd = rbind(d1, d2)
dd = subset(dd, total_sum <= 22)
ggplot() +
geom_bar(data=subset(dd, T), aes(x=type, y=total_sum, fill=type),
        position = "dodge", stat = "summary", fun.y = "mean"
) +

geom_point(data=dd, aes(x=as.numeric(type) + 0.2, y=total_sum), alpha=1.0, size=0.1, stroke = 0.5, position=position_jitter(width=0.1, height = 0.0)) +
stat_summary(data=dd, aes(x=type, y=total_sum, fill=type), fun.data = mean_cl_normal, geom="linerange", colour="black",
             position=position_dodge(.9),  fun.args = list(mult = 1), size=0.5) +

# geom_point(data=dd, aes(x=type, y=total_sum), size=0.2) +
scale_fill_brewer(palette="Set1") +
theme_bw() +
theme(legend.position="None")  +
xlab("") +
ylab("#errors")
ggsave("../../doc/graphs/gonogo_all.pdf", width=10, height=10, unit="cm")


# dd = subset(rbind(d1, d2, d3), total_sum <= 22)
# ggplot() +
# geom_bar(data=dd, aes(x=type, y=total_sum, fill=type),
#         position = "dodge", stat = "summary", fun.y = "mean"
# ) +
# geom_point(data=dd, aes(x=type, y=total_sum), size=0.2, position = position_jitter()) +
# scale_fill_brewer(palette="Set1") +
# theme_bw() +
# theme(legend.position="None")  +
# xlab("") +
# ylab("#errors")
# ggsave("../../doc/graphs/gonogo_some.pdf", width=10, height=10, unit="cm")

########
wilcox.test(d3$total_sum, d1$total_sum)


rm(list = ls())
########### for data from random nogo #################
require(plyr)
dd = read.csv("../../data/go-nogo/fromNCpaper/go_nogo.csv")
dd = subset(dd, exp_stage == 'test')


pre_dd = ddply(dd, "worker_id", function(x){data.frame(total_sum =
  sum(((x$correct_response != x$key_press) * 1)))})

sl = unique(subset(pre_dd, total_sum <= 22))
l = nrow(dd)

dd = subset(dd, worker_id %in% sl$worker_id)

sum((dd$correct_response != dd$key_press) * 1)
pre_dd = ddply(dd, "worker_id", function(x){data.frame(total_sum =
  sum(((x$correct_response != x$key_press) * 1)))})

nrow(pre_dd)

# for filtering subjects with too many errors
quantile(pre_dd$total_sum, probs = c(0.05, 0.95))



########## extractin random data #########
######################
# dirs1 = list.files(path = "../../nongit/archive/gonogo/onto-cv/human_data/gonogo_data_mix_1/", full.names = TRUE, recursive = FALSE)
dirs1 = list.files(path = "../../nongit/archive/gonogo/local-cv/round2/adv_data1/", full.names = TRUE, recursive = FALSE)
dirs2 = list.files(path = "../../nongit/archive/gonogo/local-cv/round2/adv_data2/", full.names = TRUE, recursive = FALSE)
dirs3 = list.files(path = "../../nongit/archive/gonogo/local-cv/round2/training_data/", full.names = TRUE, recursive = FALSE)

dirs = c(dirs1, dirs2, dirs3)

output = list()
index = 1
for (d in dirs){
    dd = read.csv(d)
    dd_prac = subset(dd, exp_stage == 'practice')
    dd = subset(dd, exp_stage == 'test')
    if (dd$mode[1]== 'random'){
      # file.copy(d, "../../nongit/archive/gonogo/local-cv/round3/training-data/")
    dd_prac = sum(dd_prac$correct == 'false')
    if (dd_prac > -1 ){
      output[[index]] = data.frame(id = d, total_sum = sum(dd$correct == 'false'))
      index = index + 1
  }
}
}

require(data.table)
dd = rbindlist(output)
d1 = data.frame(id = dd$id, total_sum = dd$total_sum, type="random")
nrow(d1)
quantile(d1$total_sum, probs = c(0.05, 0.75))

length(unique(d1$id))
length(unique(subset(d1, total_sum <=31.75)$id))
mean(subset(d1, total_sum <= 20.75)$total_sum)


######################
