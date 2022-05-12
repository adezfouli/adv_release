cells = c(5,8,10)
all_d = list()
indx = 1
for (cell in cells){
  for (f in c(0:9)){
  # path = paste0("../../nongit/archive/learner/nc/learner_human_nc/learner_nc_cells_", cell, "/events.csv")
  # path = paste0("../../nongit/archive/learner/nc/old/learner_qrl_nc/learner_synth_nc_cells_", cell, "/events.csv")
  # path = paste0("../../nongit/archive/gonogo/onto-cv/gonogo_learner/cells_", cell, "/fold_", f, "/events.csv")
  path = paste0("../../nongit/archive/gonogo/state-reg/gonogo_learner_sreg/cells_", cell, "/fold_", f, "/events.csv")
  d = read.csv(path)
  d$cell = cell
  d$fold = f
  all_d[[indx]] = d
  indx = indx + 1
}
}
require(data.table)
d = as.data.frame(rbindlist(all_d))

d = subset(d, epoch <= 14000)
nrow(d)

require(plyr)
d_summr = ddply(subset(d, T), c("epoch", "cell"), function(x){data.frame(loss = mean(x$sum.loss))})
d_summr[order(d_summr$loss),]
d_summr[which.min(d_summr$loss), ]

# dd = subset(d, epoch %% 1000 == 0)
dd = subset(d, T)
require(ggplot2)
ggplot() +
geom_bar(data=dd, aes(x=epoch, y=sum.loss),
 position = "dodge", stat = "summary", fun.y = "mean", fill = "red"
) +
stat_summary(data=dd, aes(x=epoch, y=sum.loss),
fun.data = mean_cl_normal, geom="linerange", colour="black",
             position=position_dodge(.9),  fun.args = list(mult = 1), size=0.5) +
             facet_grid(cell ~ .)
