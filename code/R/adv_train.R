logs = read.table("../../nongit/archive/nc/ql/dqn/RL_ql_dqn/RL_nc_dqn_buf_10000_eps_0.01_lr_0.0001/run.log", fill = TRUE, skip=50)

#logs = read.table("../../nongit/results/run.log",  fill = TRUE, skip=100)

#logs = read.table("../../nongit/results/dqn/run.log", fill = TRUE, skip=25)


rewws = strsplit(as.character(logs$V15), ',')

all_loss = data.frame(iter = c(1:(length(logs$V9)-1)),
loss=as.numeric(rewws[1:(length(logs$V15)-1)]))
# 
# all_loss = data.frame(iter = as.numeric(as.character(logs$V9[1:(length(logs$V9)-1)])),
# loss=as.numeric(as.character(logs$V15[1:(length(logs$V15)-1)])))

# all_loss = data.frame(loss = logs$V15[logs$V15 != ''])
# all_loss$iter = c(1:nrow(all_loss))
# all_loss$loss = as.numeric(strsplit(as.character(all_loss$loss), ','))

# summary(subset(all_loss, iter < 100)$loss)
# all_loss$iter = c(1:nrow(all_loss))
# mean(subset(all_loss, iter  > 10000)$loss)

require(ggplot2)
ggplot(data=subset(all_loss, TRUE), aes(x = iter, y = loss)) +
    geom_line() +
    geom_smooth() +
    geom_hline(yintercept=50, linetype="dashed", color = "red") +
    scale_color_brewer(breaks = c("train", "test", "rand"),
                            labels=c("Train", "Test", "Random"), palette="Set1", name="")+
    xlab('training iteration') +
    ylab('reward') +
    # ylim(c(0, 10)) +
    theme(panel.grid.major.x = element_blank()) +
    theme(panel.grid.major.y = element_line(size = 0.1)) +
    theme(text=element_text(size=10, family="Times"))

ggsave("../../doc/graphs/RL_train.pdf", width=14, height=5, unit="cm")
