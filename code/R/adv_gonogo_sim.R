dir_lists = list()
i = 1
for (it in as.character(c('3000', '2000','1000', '500', '0'))){
  dir_lists[[i]] = paste0("../../nongit/archive/gonogo/state-reg/static-sims-1/gonog_sim_", it, "/")
  i = i + 1
}


output = list()
index = 1

for (pth in as.array(dir_lists)){
  dirs = list.dirs(path = pth, full.names = TRUE, recursive = FALSE)
  for (d in dirs){
    adv_rew = read.csv(paste0(d, '/psudo_reward_.csv'))
    adv_rew$model = d
    output[[index]] = adv_rew
    index = index + 1
  }
}

require(data.table)
all_d = rbindlist(output)
nrow(all_d)
head(all_d)

require(plyr)
require(plotrix)

# mms = ddply(subset(all_d, X0 <= 20.75), "model", function(x){data.frame(rew = mean(x$X0), std.error=std.error(x$X0))})
require(plyr)
mms = ddply(subset(all_d, T), "model", function(x){data.frame(rew = mean(x$X0), std.error=std.error(x$X0))})
nrow(mms)

mms[order(mms$rew),]
mms[which.max(mms$rew),]

require(ggplot2)
ggplot() +
geom_bar(data = all_d, aes(x=model, y= X0),stat = "summary", fun.y = "mean", show.legend=FALSE) +
scale_fill_brewer(palette="Set1")

###### for plotting but not used in the paper
output = list()
index = 1

for (d in c('../../nongit/archive/gonogo/state-reg/static-sims-1/gonog_sim_1000//RL_gonogo_2layers_0.05ent_256units_lof0.0/', 
      '../../nongit/archive/gonogo/state-reg/static-sims-1/gonog_sim_2000//RL_gonogo_2layers_0.01ent_256units_lof0.0/'
)){
  adv_rew = read.csv(paste0(d, '/psudo_reward_.csv'))
  adv_rew$model = d
  output[[index]] = adv_rew
  index = index + 1
}

require(data.table)
all_d = rbindlist(output)
require(plyr)
mms = ddply(subset(all_d, T), "model", function(x){data.frame(rew = mean(x$X0), std.error=std.error(x$X0))})


ggplot() +
geom_bar(data=subset(all_d, T), aes(x=model, y=X0, fill=model),
        position = "dodge", stat = "summary", fun.y = "mean"
) +
stat_summary(data=subset(all_d, T), aes(x=model, y=X0, fill=model), fun.data = mean_cl_normal, geom="linerange", colour="black",
             position=position_dodge(.9),  fun.args = list(mult = 1), size=0.5) +
scale_fill_brewer(palette="Set1") +
theme_bw() +
theme(legend.position="None")  +
xlab("") +
ylab("#errors")+
theme(legend.position="None")

ggsave("../../doc/graphs/gonogo_hyper.pdf", width=10, height=10, unit="cm")

#########$$$$$$$$$$$$$$$$ temp not used

dirs = list.dirs(path = "../../nongit/archive/gonogo/state-reg/RL_gonogo_dqn_vec/", full.names = TRUE, recursive = FALSE)


output = list()
index = 1
for(d in dirs){
  logs = read.table(paste0(d, "/run.log"), fill = TRUE, skip=50)


  rewws = strsplit(as.character(logs$V15), ',')

  all_loss = data.frame(iter = c(1:(length(logs$V9)-1)),
  loss=as.numeric(rewws[1:(length(logs$V15)-1)]))

  nr = nrow(all_loss)

  output[[index]] = data.frame(d = d, m = mean(all_loss[(nr - 100):nr, ]$loss, na.rm=T))
  index = index + 1

}

require(data.table)
all_d = rbindlist(output)
all_d[order(all_d$m),]
