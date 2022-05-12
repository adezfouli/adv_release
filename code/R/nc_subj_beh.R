rm(list = ls())
dirs1 = list.files(path= "../../nongit/archive/nc/human/run-cv/human_data_dqn_vec/random_0/", full.names = TRUE, recursive = FALSE)

correct_subj  = 0
bias = 0
for (d in dirs1){
  s_d = read.csv(d)
  print(nrow(s_d))
  if(nrow(s_d) == 101){
    s_d = s_d[1:100,]
  }
  if(nrow(s_d) == 100){
    correct_subj  = correct_subj + 1
    bias = bias + sum(as.character(s_d$is_biased_choice) == " true")
  }
}


bias / correct_subj


##################### old code $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
dirs = list.files(path = "../../nongit/archive/nc_human_data/batch_2/random_0/", full.names = TRUE, recursive = FALSE)

correct_subj  = 0
bias = 0
for (d in dirs){
  s_d = read.csv(d)
  print(nrow(s_d))
  if(nrow(s_d) == 101){
    s_d = s_d[1:100,]
  }
  if(nrow(s_d) == 100){
    correct_subj  = correct_subj + 1
    bias = bias + sum(as.character(s_d$is_biased_choice) == " true")
  }
}

dirs = list.files(path = "../../nongit/archive/nc_human_data/batch_1/random_0/", full.names = TRUE, recursive = FALSE)
for (d in dirs){
  s_d = read.csv(d)
  print(nrow(s_d))
  if(nrow(s_d) == 101){
    s_d = s_d[1:100,]
  }
  if(nrow(s_d) == 100){
    correct_subj  = correct_subj + 1
    bias = bias + sum(as.character(s_d$is_biased_choice) == " true")
  }
  # print(nrow(s_d))
  # print(sum(as.character(s_d$is_biased_choice) == " true"))
}


dirs = list.files(path = "../../nongit/archive/nc_human_data/batch_3/random_0/", full.names = TRUE, recursive = FALSE)
for (d in dirs){
  s_d = read.csv(d)
#  print(nrow(s_d))
  if(nrow(s_d) == 101){
    s_d = s_d[1:100,]
  }
  if(nrow(s_d) == 100){
    correct_subj  = correct_subj + 1
    print(sum(as.character(s_d$is_biased_choice) == " true"))
    bias = bias + sum(as.character(s_d$is_biased_choice) == " true")
  }
  # print(nrow(s_d))
  # print(sum(as.character(s_d$is_biased_choice) == " true"))
}


bias / correct_subj
