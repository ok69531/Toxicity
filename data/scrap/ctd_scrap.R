rm(list = ls())
gc(reset = T)

library(stringr)
library(httr)
library(rvest)

setwd('C:/Users/SOYOUNG/Desktop/toxic/ctd')

url = 'http://ctdbase.org/downloads/'

# ctd_line = readLines(url, encoding = 'UTF-8')
# 
# 
# idx = str_detect(ctd_line, '<li><a href="#')
# sum(idx) ## 18
# idx = which(idx)
# 
# length(idx)
# 
# ctd_listup = c(str_extract(ctd_line[idx], '#[a-z]*'))
# 
# url에다가 ctd_listup 하나씩 붙여서 파일 다운로드하기! 되겠지..?
# 
# file_url = paste0(url, ctd_listup)
# readLines(file_url[1], encoding = 'UTF-8')
# a = readLines('http://ctdbase.org/downloads/#cg', encoding = 'UTF-8')


file_url_get = httr::GET(url) ## 여기
if(file_url_get$status_code != 200) print('Fail!')


file_url = read_html(file_url_get) %>% 
  html_nodes('body#downloadsdownloads')

elim_vec = c('top', 'header', 'nav', 'content', 'pgheading', 'pagetoc', 'footer', 'gd')
file_class = file_url %>% html_children() %>% html_attr('id')
div_class_name = file_class[!file_class %in% elim_vec & !is.na(file_class)]

for(i in 1:length(div_class_name))
{
  if(!file.exists(div_class_name[i])) dir.create(div_class_name[i])
  file_url %>% html_nodes(paste0('div', '#', div_class_name[i])) %>% 
    html_nodes('table.filelisting a') %>%
    html_attr('href')
  file_url_path = paste0('http://ctdbase.org', file_url)
  file_name = gsub('/reports/', '', file_url)
  for(j in 1:length(file_url_path))
  {
    download.file(file_url_path[j], paste0(div_class_name[j], '/', file_name[j]))
  }
}


# div_class_name
# 
# 
# file_url_path = paste0('http://ctdbase.org', file_url)
# file_name = gsub('/reports/', '', file_url)






file_idx = str_detect(file_url[1], '<a href="/reports/')
