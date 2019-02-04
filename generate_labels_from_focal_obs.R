#Process focal recordings into data frames of training and testing data

#Generate a data table of labels from all the focal observations in the subdirectories of a main folder 'soundfocs_dir'
#INPUTS:
# soundfocs_dir: directory where files are stored (can be stored in subdirectories)
# require_start_stop: whether or not to require that each file have a start and stop marker label (in this case those without will be excluded)
#OUTPUTS:
# labels: data frame containing
#     $t0: initial time of marker
#     $tf: final time of marker
#     $label: label of marker
#     $filename: base filename of where the labels are from
#     $type: categorized marker type
#     $iscall: whether the marker is a call or not
get.focal.labels <- function(soundfocs_dir,require_start_stop = F){
  
  #get all text files in folder (and subfolders, if present)
  files <- list.files(soundfocs_dir,pattern='*[.]txt$',recursive=TRUE,include.dirs = TRUE)
  
  #load labels from all files
  labels <- data.frame()
  for(i in 1:length(files)){
    print(files[i])
    file <- paste(soundfocs_dir,files[i],sep='/')
    labs.curr <- read.csv(file=file,stringsAsFactors=F,header=F,sep='\t')
    colnames(labs.curr) <- c('t0','tf','label')
    labs.curr$filename <- basename(files[i])
    labels <- rbind(labels,labs.curr)
  }
  
  #parse labels into categories
  regexps <- c('cc.*','sn.*','mo.*','nn:.*','start.*','end.*','pause.*','resume.*','agg.*','alarm.*','unk.*','ld.*','hyb.*','qv.*','bv.*','sc.*','pf.*','chat.*','[?].*','beg.*','rec on','rec off','sv.*','ea.*')
  types <- c('cc','sn','mo','nn','start','end','pause','resume','agg','alarm','unk','ld','hyb','qv','bv','sc','pf','chat','?','beg','recon','recoff','sv','ea')
  iscall <- c(T,T,T,F,F,F,F,F,T,T,T,T,T,F,F,T,F,T,F,F,F,F,F,F)
  
  parse.table <- data.frame(regex = regexps,type=types,iscall=iscall,stringsAsFactors = F)
  
  labels$type <- NA
  labels$iscall <- F
  for(i in 1:nrow(parse.table)){
    idxs <- which(grepl(pattern=parse.table$regex[i],labels$label,ignore.case = TRUE))
    labels$type[idxs] <- parse.table$type[i]
    labels$iscall[idxs] <- parse.table$iscall[i]
  }
  
  #if needed, remove files with no start and stop marker
  if(require_start_stop){
    files <- unique(labels$filename)
    new_labels <- data.frame()
    for(i in 1:length(files)){
      curr <- labels[which(labels$filename==files[i]),]
      if(nrow(curr)>2){
        starts <- sum(as.character(curr$type) %in% c('start','begin'),na.rm=T)
        ends <- sum(as.character(curr$type) %in% c('end','stop'),na.rm=T)
        if(starts == 1 & ends == 1){
          new_labels <- rbind(new_labels, curr)
        }
      }
    }
    labels <- new_labels
  }
  
  
  return(labels)
}

#MAIN

#Directories
dir_all <- '/Volumes/Elements/Sound Files/Pups VS No Pups/'
dir_testset <- '/Volumes/Elements/Sound Files/Ari Examined files/'
dir_testset2 <- '/Users/astrandb/Dropbox/meerkats/data/sound_focals/labels/'
outdir <- '/Users/astrandb/Dropbox/meerkats/data/sound_focals/labels_list/'

#Get labels for test set
labels_all <- get.focal.labels(soundfocs_dir = dir_all)
labels_test <- get.focal.labels(soundfocs_dir = dir_testset)
labels_test2 <- rbind(labels_test,get.focal.labels(soundfocs_dir = dir_testset2))

#Get trianing data (make sure to exclude files in the test set)
files_all <- unique(labels_all$filename)
files_test <- unique(labels_test$filename)
files_test <- c(files_test,unique(labels_test2$filename))

idxs_train <- which(!(labels_all$filename %in% files_test))
labels_train <- labels_all[idxs_train,]

write.csv(x=labels_train,file=paste(outdir,'focal_labels_train.csv',sep=''))
write.csv(x=labels_test,file=paste(outdir,'focal_labels_test.csv',sep=''))


