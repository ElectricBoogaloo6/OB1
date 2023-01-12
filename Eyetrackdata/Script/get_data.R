
dall_selection = d.all[,c('FFDr','FFDp','SFD','GZD','TVT','SKP','nfp','nsp','nap','s1','s2','nlett1','ISL','OSL','f','pred','ILP','haveFixation','haveFirstPass')]
dall_selection[is.na(dall_selection)] <- 0
dall_selection = dall_selection*1

write.table(dall_selection, "Fixation_durations_complete2.txt", sep="\t",quote = FALSE,row.names = FALSE)

