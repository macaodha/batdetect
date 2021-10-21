#tensorflow bat detector

getwd()
setwd('C:/Users/Anthony/Documents/GitHub/batdetect/bat_train')

test_set      <- 'bulgaria'  # can be one of: bulgaria, uk, norfolk
data_set      <- paste0('data/train_test_split/test_set_', test_set, '.npz')
raw_audio_dir <- 'data/wav/'
base_line_dir <- 'data/baselines/'
result_dir    <- 'results/'
model_dir     <- 'data/models/'

dir('bat_train/data/train_test_split')

if(!file.exists(result_dir)){
  dir.create(result_dir)
}
if(!file.exists(model_dir)){
  dir.create(model_dir)
}

cat('test set:', test_set)
#plt.close('all')

library(reticulate)

np <- import('numpy')

#np$load()
# train and test_pos are in units of seconds
loaded_data_tr  <- np$load(data_set, allow_pickle = TRUE, encoding = 'latin1')
train_pos       <- loaded_data_tr['train_pos']
train_files     <- loaded_data_tr['train_files']
train_durations <- loaded_data_tr['train_durations']
test_pos        <- loaded_data_tr['test_pos']
test_files      <- loaded_data_tr['test_files']
test_durations  <- loaded_data_tr['test_durations']


#
# CNN
#print('\ncnn')

#np$hs
extract_train_position_from_file <- function(gt_pos, duration){

  num_neg_calls = length(gt_pos)
  window_size   = 0.230
  shift         = 0.015
  pos_window    = window_size / 2  # window around GT that is not sampled from
  pos           = gt_pos
  
  # augmentation
  num_neg_calls = 3*num_neg_calls
  
  pos = np$hstack(c(lapply(gt_pos, function(x) x - shift), 
                    gt_pos, 
                    lapply(gt_pos, function(x) x + shift)))
  
  # sample a set of negative locations - need to be sufficiently far away from GT
  #c(0 - window_size, gt_pos[[1]], duration - window_size)
  #pos
  pos_pad = np$hstack(c(0 - window_size, gt_pos[[1]], duration - window_size))
  neg     = c()
  cnt     = 0
  browser()
  while(cnt < num_neg_calls){
    rand_pos = np$random$random()*max(pos_pad)
    if(mean((np$abs(pos_pad - rand_pos) > (pos_window + shift))) == 1){
      neg = c(neg, rand_pos)
      cnt = cnt + 1
      neg = np$asarray(neg)
    }
  }
  browser()
  # sort them
  positions   = np$hstack(c(pos, neg))
  sorted_inds = np$argsort(positions)
  positions   = positions[sorted_inds]
  
  # create labels
  
  class_labels = np$vstack(c(np$ones(c(pos$shape[1],1)), np$zeros(c(pos$shape[1],1))))
  class_labels = class_labels[sorted_inds]
  
  return(list(positions, class_labels))
}

generate_training_positions <- function(files, gt_pos, durations){
  pos_list = list()
  for(ii in 1:length(files)){
    pos_list = append(pos_list, extract_train_position_from_file(gt_pos[ii], durations[ii]))
  }
  #pos_list = extract_train_position_from_file()
  return(pos_list)
}

generate_training_positions(train_files, train_pos, train_durations)

#model.train(train_files, train_pos, train_durations)

train_pos[[1000]][4]
