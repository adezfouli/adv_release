ql_h = read.csv('../../nongit/archive/nc/cross-sims/ql_adv_vs_human_lrn/adv_reward_.csv')
ql_h$type = "QL adversary vs human learner"
ql_ql = read.csv('../../nongit/archive/nc/cross-sims/ql_adv_vs_ql_lrn/adv_reward_.csv')
ql_ql$type = "QL adversary vs QL learner"
h_ql = read.csv('../../nongit/archive/nc/cross-sims/human_adv_vs_ql_lrn/adv_reward_.csv')
h_ql$type = "Human adversary vs QL learner"
h_h = read.csv('../../nongit/archive/nc/cross-sims/human_adv_vs_human_lrn/adv_reward_.csv')
h_h$type = "Human adversary vs human learner"

advs = rbind(ql_h, h_ql)

require(plyr)
ddply(advs, "type", function(x){m = mean(x$X0)})


require(ggplot2)
ggplot() +
geom_bar(data=advs, aes(x=type, y=X0, fill=type),
        position = "dodge", stat = "summary", fun.y = "mean"
) +
geom_point(data=advs, aes(x=as.factor(type) + 0.2, y=X0, group=type), alpha=1.0, size=0.1, stroke = 0.03, position=position_jitter(width=0.1, height = 0.0)) +
stat_summary(data=advs, aes(x=type , y=X0, fill=type), fun.data = mean_cl_normal, geom="linerange", colour="black",
             position=position_dodge(.9),  fun.args = list(mult = 1), size=0.5) +

# geom_point(data=advs, aes(x = system + 0.1, y=value), color="red4",
#         position=position_jitterdodge(dodge.width=0.9, jitter.width=0.2), alpha=1.0, size=0.5, stroke = 0.05) +

scale_fill_brewer(palette="Set1", name="") +
ylim(c(0, 100)) +
theme_bw() +
ylab("bias") +
theme(legend.position="right")  +
theme(text = element_text(size=8)) +
theme(axis.text=element_text(size=8), panel.grid.major.x = element_blank()) +
theme(axis.text.x=element_blank()) +
geom_hline(yintercept=50, linetype="dashed", color = "black") +
xlab("")

ggsave("../../doc/graphs/cross.pdf", width=12, height=6, unit="cm")
