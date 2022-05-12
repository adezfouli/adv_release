######### for ql models ######################
cells = c(2,3,4,5,8,10,15)
all_d = list()
indx = 1
for (cell in cells){
  for (f in c(0:9)){
  path = paste0("../../nongit/archive/mrtt/RND/mrtt_RND_learner/cells_", cell, "/fold_", f, "/events.csv")
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
