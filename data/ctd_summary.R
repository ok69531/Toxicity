rm(list = ls())
gc(reset = T)

if(!require(RMySQL)) install.packages('RMySQL'); library(RMySQL)
if(!require(dplyr)) install.packages('dplyr'); library(dplyr)

conn = dbConnect(dbDriver('MySQL'), 
                 host = '127.0.0.1', dbname = 'pyctd', port = 3306,
                 user = 'root', password = 'nrz5oloF71')


########## all chemicals ##########
chems = dbGetQuery(conn, 'select * from allchems')
dim(chems); str(chems)

chem_len = apply(chems, 2, function(x) length(unique(x)))
chem_na_len = apply(chems, 2, function(x) sum(is.na(x)))
chem_len + chem_na_len

length(chems)
head(chems$Definition[complete.cases(chems$Definition)])


# ChemicalName과 ChemicalID 일대일대응 ?
# CasRN? -> CAS 등록 번호는 이제까지 알려진 모든 화합물, 중합체 등을 기록하는 번호
# parentid가 무엇 ?
# treenumbers (identifiers of the chemical's nodes;) 가 무엇 ? -> chemical(분자)의 구성 요소 ? -> 원자들(?) 
# parentTreeNumbers ?
# synonyms는 chemicalname의 동의어 ? -> 구분자 '|'가 '또는'을 의미 ?



########## all disease  ##########
dis = dbGetQuery(conn, 'select * from alldiseases')
dim(dis); str(dis)

dis_len = apply(dis, 2, function(x) length(unique(x)))
dis_na_len = apply(dis, 2, function(x) sum(is.na(x)))
dis_len + dis_na_len

slim_map = strsplit(dis$SlimMappings, '\\|')
d = slim_map %>% unlist()
table(d)[which.max(table(d))]; table(d)[which.min(table(d))]



# DiseaseName과 DiseaseID 일대일대응?

# SlimMappings: 상위 -> 하위 관계?



########## chemical-disease ##########
cd = dbGetQuery(conn, 'select * from cd')
dim(cd); str(cd)

cd_len = apply(cd, 2, function(x) length(unique(x)))
cd_na_len = apply(cd, 2, function(x) sum(is.na(x)))
cd_len + cd_na_len

summary(cd$InferenceScore)



########## chemical-gene ##########
cg = dbGetQuery(conn, 'select * from cg')
dim(cg); str(cg)

cg_len = apply(cg, 2, function(x) length(unique(x)))
cg_na_len = apply(cg, 2, function(x) sum(is.na(x)))
cg_len + cg_na_len

unique(cg[, which.min(cg_len)])
table(cg$GeneForms)


# PubMedIDs 가 무엇 ?


