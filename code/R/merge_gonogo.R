dirs = list.files(path = "../../nongit/archive/gonogo/state-reg/training-data/", full.names = TRUE, recursive = FALSE)

output = list()
index = 1
for (d in dirs){
  dd = read.csv(d)
  dd$worker_id = d
  output[[index]] = dd
  index = index + 1
}

require(data.table)
d_data = rbindlist(output)

write.csv(d_data, "../../data/go-nogo/local/state-reg/merged.csv")
