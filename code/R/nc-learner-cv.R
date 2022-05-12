######### for ql models ######################
cells = c(5,8,10)
all_d = list()
indx = 1
for (cell in cells){
  for (f in c(0:9)){
  path = paste0("../../nongit/archive/nc/ql/run-cv/nc_ql_learner/cells_", cell, "/fold_", f, "/events.csv")
  d = read.csv(path)
  d$cell = cell
  d$fold = f
  all_d[[indx]] = d
  indx = indx + 1
}
}
require(data.table)
d = rbindlist(all_d)

nrow(d)

require(plyr)
d_summr = ddply(subset(d, epoch < 49000), c("epoch", "cell"), function(x){data.frame(loss = mean(x$loss))})
d_summr[order(d_summr$loss),]
d_summr[which.min(d_summr$loss), ]

dd = subset(d, epoch < 49000 & epoch %% 1000 == 0 )
require(ggplot2)
ggplot() +
geom_bar(data=dd, aes(x=epoch, y=sum.loss),
 position = "dodge", stat = "summary", fun.y = "mean", fill = "red"
) +
stat_summary(data=dd, aes(x=epoch, y=sum.loss),
fun.data = mean_cl_normal, geom="linerange", colour="black",
             position=position_dodge(.9),  fun.args = list(mult = 1), size=0.5) +
             facet_grid(cell ~ .)


######### for human models ######################
cells = c(5,8,10)
all_d = list()
indx = 1
for (cell in cells){
 for (f in c(0:9)){
 path = paste0("../../nongit/archive/nc/human/run-cv/nc_human_learner/cells_", cell, "/fold_", f, "/events.csv")
 d = read.csv(path)
 d$cell = cell
 d$fold = f
 all_d[[indx]] = d
 indx = indx + 1
}
}
require(data.table)
d = rbindlist(all_d)

nrow(d)

require(plyr)
d_summr = ddply(subset(d, epoch < 49000), c("epoch", "cell"), function(x){data.frame(loss = mean(x$loss))})
d_summr[order(d_summr$loss),]
d_summr[which.min(d_summr$loss), ]

dd = subset(d, epoch < 2000 & epoch %% 100 == 0 )
require(ggplot2)
ggplot() +
geom_bar(data=dd, aes(x=epoch, y=sum.loss),
position = "dodge", stat = "summary", fun.y = "mean", fill = "red"
) +
stat_summary(data=dd, aes(x=epoch, y=sum.loss),
fun.data = mean_cl_normal, geom="linerange", colour="black",
            position=position_dodge(.9),  fun.args = list(mult = 1), size=0.5) +
            facet_grid(cell ~ .)
