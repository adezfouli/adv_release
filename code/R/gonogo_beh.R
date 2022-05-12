rm(list = ls())
########### for data from random nogo #################
require(plyr)
dd = read.csv("../../data/go-nogo/local/state-reg/merged.csv")
dd = subset(dd, exp_stage == 'test')


pre_dd = ddply(dd, "worker_id", function(x){data.frame(total_sum =
  sum(((x$correct_response != x$key_press) * 1)))})

sl = unique(subset(pre_dd, total_sum <= 31.75))
l = nrow(dd)

dd = subset(dd, worker_id %in% sl$worker_id)

head(dd$trial_num)
head(dd$correct)

dd$bcorrect = 1 * (dd$correct == 'true')
dd$trial_g = floor((dd$trial_num + 1  - 0.0001) / 50) *  50
dd2_group = ddply(subset(dd, T), c("worker_id", "trial_g", "condition"),
            function(x){data.frame(cor = mean(x$bcorrect))})



require(ggplot2)
dd2_group$cor = dd2_group$cor * 100
head(dd2_group)
ggplot() +
  geom_bar(data = dd2_group, aes(x = as.factor(trial_g), y = cor, fill=condition, group=condition),
  position = "dodge", stat = "summary", fun.y = "mean"
) +
stat_summary(data = dd2_group, aes(x = as.factor(trial_g), y = cor, group=condition),
fun.data = mean_cl_normal, geom="linerange", colour="black",
             position=position_dodge(0.9),  fun.args = list(mult = 1), size=0.5) +

scale_x_discrete(labels=c("1-50", "50-100", "100-150", "150-200", "200-250", "250-300", "300-350")) +

# geom_point(data=dd, aes(x=type, y=total_sum), size=0.2) +
scale_fill_brewer(palette="Set1") +
xlab("trial") +
ylab("%correct") +
theme_bw() +
theme(axis.text.x = element_text(
                           angle = 90))+
theme(legend.title = element_blank())

ggsave("../../doc/graphs/subj-correct.pdf", width=8, height=5.7, unit="cm")

require(plotrix)
std.error(subset(dd2_group, condition == "nogo" & trial_g == 300)$cor)

########################## old analysis ######################

require(plyr)
dd = ddply(dd, "worker_id", function(x){
  x$sum_nogo = NA
  x$sum_nogo = cumsum((x$condition == 'nogo')*1)
  x
})

dd2 = dd[3:l,]
dd2$prev_condition2 = dd$condition[1:(l-2)]
dd2$prev_condition1 = dd$condition[2:(l-1)]
dd2$prev_correct1 = dd$correct[2:(l-1)]
dd2$prev_correct2 = dd$correct[1:(l-2)]
dd2$correct = 1 * (dd2$correct == 'True')

cond1 = dd2$condition == 'nogo' & dd2$prev_condition1 == 'go' &
dd2$prev_condition2 == 'go' &
dd2$prev_correct1 == 'True' &
dd2$prev_correct2 == 'True'

cond2 = dd2$condition == 'nogo' & dd2$prev_condition1 == 'nogo' &
dd2$prev_condition2 == 'go' &
dd2$prev_correct1 == 'False' &
dd2$prev_correct2 == 'True'

dd2$cond1 = NA
dd2$cond1[cond1] = 'go'
dd2$cond1[cond2] = 'nogo'


dd2$sum_nogo_cond = floor((dd2$sum_nogo - 0.0001) / 5) *  5

dd2_group = ddply(subset(dd2, !is.na(cond1)), c("worker_id", "cond1", "sum_nogo_cond"),
            function(x){
              data.frame(cor = mean(x$correct))
            }
)

require(ggplot2)
ggplot() +
  geom_bar(data = dd2_group, aes(x = as.factor(sum_nogo_cond), y = cor, fill = cond1),
  position = "dodge", stat = "summary", fun.y = "mean"
) +
stat_summary(data = dd2_group, aes(x = as.factor(sum_nogo_cond), y = cor, group= cond1),
fun.data = mean_cl_normal, geom="linerange", colour="black",
             position=position_dodge(.9),  fun.args = list(mult = 1), size=0.5) +

# geom_point(data=dd, aes(x=type, y=total_sum), size=0.2) +
scale_fill_brewer(palette="Set1") +
xlab("#no-go trials") +
ylab("correct probability") +
theme_bw() +
theme(legend.title = element_blank())


##################### for early trials #######################################
dd2$trial_basket = dd2$sum_nogo  == 1

require(plyr)
dd3 = ddply(dd2, "worker_id", function(x){
  x$before_5 = NA
  x$before_5 = x$sum_nogo[6]
  x
})

head(dd3)
mean(subset(dd2, trial_num < 6 & condition == 'nogo')$correct)

dd_early = ddply(subset(dd3, condition == 'nogo'), c("worker_id", "before_5"),
            function(x){
              data.frame(cor = mean(x$correct))
            }
)


mean(subset(dd_early, before_5 == 5)$cor)

require(ggplot2)
ggplot() +
  geom_bar(data = dd_early, aes(x = as.factor(trial_basket), y = cor),
  position = "dodge", stat = "summary", fun.y = "mean"
) +
# stat_summary(data = dd2_group, aes(x = as.factor(sum_nogo_cond), y = cor, group= cond1),
# fun.data = mean_cl_normal, geom="linerange", colour="black",
#              position=position_dodge(.9),  fun.args = list(mult = 1), size=0.5) +
#
# # geom_point(data=dd, aes(x=type, y=total_sum), size=0.2) +
scale_fill_brewer(palette="Set1") +
xlab("#no-go trials") +
ylab("correct probability") +
theme_bw() +
theme(legend.title = element_blank())

head(dd2)


################################################################################


require(ggplot2)
ggplot() +
  geom_bar(data = subset(dd2, !is.na(cond1)), aes(x = as.factor(sum_nogo_cond), y = correct, fill = cond1),
  position = "dodge", stat = "summary", fun.y = "mean"
) +
stat_summary(data = subset(dd2, !is.na(cond1)), aes(x = as.factor(sum_nogo_cond), y = correct, group= cond1),
fun.data = mean_cl_normal, geom="linerange", colour="black",
             position=position_dodge(.9),  fun.args = list(mult = 1), size=0.5) +

# geom_point(data=dd, aes(x=type, y=total_sum), size=0.2) +
scale_fill_brewer(palette="Set1") +
theme_bw()



head(dd2)

mean(subset(dd2,
  condition == 'nogo'
  & prev_condition1 == 'go'
  & prev_condition2 == 'go'
  & prev_correct1 == 'True'
  & prev_correct2 == 'True'
  & sum_nogo  > 20
)$correct)


mean(subset(dd2,
  condition == 'nogo'
  & prev_condition1 == 'nogo'
  & prev_condition2 == 'go'
  & prev_correct1 == 'False'
  & prev_correct2 == 'True'
  & sum_nogo > 20
)$correct)

require(plyr)
res = ddply(subset(dd2, worker_id %in% sl$worker_id), "worker_id", function(x){
mean(subset(x,
  condition == 'nogo'
  & prev_condition1 == 'go'
  & prev_condition2 == 'go'
  & prev_correct1 == 'True'
  & trial_num < 50
)$correct)})
mean(res$V1, na.rm=T)



require(plyr)
res = ddply(subset(dd2, worker_id %in% sl$worker_id), "worker_id", function(x){
mean(subset(x,
  condition == 'nogo'
  & prev_condition1 == 'go'
  & prev_condition2 == 'go'
  & prev_correct1 == 'True'
  & trial_num < 50
)$correct)})
mean(res$V1, na.rm=T)

res = ddply(subset(dd2, worker_id %in% sl$worker_id), "worker_id", function(x){
mean(subset(x,
  condition == 'nogo'
  & prev_condition1 == 'nogo'
  & prev_condition2 == 'go'
  & prev_correct1 == 'False'
  & prev_correct2 == 'True'
  & trial_num < 50
)$correct)})
mean(res$V1, na.rm=T)
