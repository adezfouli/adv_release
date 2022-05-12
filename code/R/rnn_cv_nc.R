cells = c(4,5,6)
all_d = list()
indx = 1
for (cell in cells){
  path = paste0("../../nongit/archive/learner/nc/learner_human_nc/learner_nc_cells_", cell, "/events.csv")
  d = read.csv(path)
  d$cell = cell
  all_d[[indx]] = d
  indx = indx + 1
}

library(data.table)
all_d = rbindlist(all_d)

all_d = subset(all_d, cell == 5)

all_d[order(all_d$loss),]

all_d[which.min(all_d$sum.loss), ]

require(ggplot2)

ggplot() +
geom_line(data = all_d, aes(x=epoch, color=as.factor(cell), y = loss)) +
scale_color_brewer(name="reward", palette="Set1") +
xlab("loss") +
ylab("iteration")
