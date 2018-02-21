## Predicting EFW using regression-based and machine learning algorithms
library(quantregForest)
library(sampling)
library(ggplot2)
library(quantreg)
library(splines)
library(ipred)
library(gbm)
library(matrixStats)
library(MASS)

## Run this separately, as the Java needs a fresh R session
# library(rJava)
# options(java.parameters = "-Xmx20000m")
# library(bartMachine)
## 

## DATA MANAGEMENT: GOAL IS TO GENERATE TRAIN AND TEST DATA
library(data.table);library(beepr)
m <- read.table(file="~/FGRFetalDeath/momi_odn_rf.txt",sep="\t",header = T,na.strings=c(" "))
head(m,15)

j1<-m$gestweek
j2<-jitter(m$vid,factor=1)
min(j1);max(j1)
pdf(file="~/PredictFGR/visits.pdf",width=10,height=8)
par(mar=c(5.1,4.1,4.1,2.75),cex=2)
plot(j1,j2,xlab="Gestational Week",
     ylab="Ultrasound Visit Number",
     pch=20,cex=.5,col="blue",axes=T,xaxt="n",yaxt="n",bty="l",las=1,ylim=c(1,8))
axis(2,seq(1,8,1),label=c("1","2","3","4","5","6","7","8"),las=1,tcl=-.15)
axis(1,seq(6,42,4),label=c("6","10","14","18","22","26","30","34","38","42"),las=1,tcl=-.15)
axis(4,seq(1,8,1),label=table(m$vid),las=1,line=-1,lty=0)
dev.off()

names(m)
table(m$race)/sum(table(m$race))
aggregate(as.numeric(m$gab<37),list(m$race),mean)
m2<-subset(m,m$last==1)
nrow(m2);nrow(m)

mm <- read.table(file="~/FGRFetalDeath/odn_momi_rf.txt",sep="\t",header = T,na.strings=c(" "))
head(mm,100)
mm<-na.omit(mm)

## SPLIT INTO TRAIN AND TEST DATA
head(m)
m$deliverydate<-as.Date(m$deliverydate)
m$yob<-as.numeric(format(m$deliverydate, "%Y"))
table(m$yob)
names(m)
train<-subset(m,m$yob<2010);
test<-subset(m,m$yob>=2010);
nrow(as.matrix(unique(m$rank_pregid)))
nrow(as.matrix(m$rank_pregid))
nrow(as.matrix(unique(train$rank_pregid)))
nrow(as.matrix(train$rank_pregid))
nrow(as.matrix(unique(test$rank_pregid)))
nrow(as.matrix(test$rank_pregid))

(nrow(test)+nrow(train)==nrow(m))

m$vid2<-ifelse(m$vid<4,m$vid,4)
table(m$vid2)/sum(table(m$vid2))

## NEED TO COMPARE VARIABLES BTW DATASETS
names(mm)
hist(test$gab);hist(mm$gab);
hist(test$EFW);hist(mm$EFW);
hist(test$gestweek);hist(mm$gestweek);
hist(test$apgar1);hist(mm$apgar1);
hist(test$apgar5);hist(mm$apgar5);
hist(test$momage);hist(as.numeric(mm$momage));
table(test$black)/sum(table(test$black));table(mm$black);
table(test$white)/sum(table(test$white));table(mm$white);
table(test$hispanic)/sum(table(test$hispanic));table(mm$hispanic);

head(test);mean(mm$apgar1);mean(mm$apgar5)

plot(test$gab,test$birthweight,col="blue",pch=20,cex=.5)
points(mm$gab,mm$birthweight,col="red",pch=20,cex=.5)

plot(test$gestweek,test$EFW,col="blue",pch=20,cex=.5,xlim=c(20,42),ylim=c(0,6000))
points(mm$gestweek,mm$EFW,col="red",pch=20,cex=.5,xlim=c(20,42),ylim=c(0,6000))
points(test$gab,test$birthweight,col="cyan",pch=20,cex=.5,xlim=c(20,42),ylim=c(0,6000))

mm$gestweek<-(mm$gestweek-mean(train$gestweek))/sd(train$gestweek)
mm$gab<-(mm$gab-mean(train$gab))/sd(train$gab)
mm$birthweight<-(mm$birthweight-mean(train$birthweight))/sd(train$birthweight)
mm$apgar1<-(mm$apgar1-mean(train$apgar1))/sd(train$apgar1)
mm$apgar5<-(mm$apgar5-mean(train$apgar5))/sd(train$apgar5)
mm$momage<-(as.numeric(mm$momage)-mean(train$momage))/sd(train$momage)
mm$yob<-(mm$yob-mean(train$yob))/sd(train$yob)

head(mm)

test$gestweek<-(test$gestweek-mean(train$gestweek))/sd(train$gestweek)
test$gab<-(test$gab-mean(train$gab))/sd(train$gab)
test$birthweight<-(test$birthweight-mean(train$birthweight))/sd(train$birthweight)
test$apgar1<-(test$apgar1-mean(train$apgar1))/sd(train$apgar1)
test$apgar5<-(test$apgar5-mean(train$apgar5))/sd(train$apgar5)
test$momage<-(test$momage-mean(train$momage))/sd(train$momage)
test$yob<-(test$yob-mean(train$yob))/sd(train$yob)

train$gestweek<-scale(train$gestweek)
train$gab<-scale(train$gab)
train$birthweight<-scale(train$birthweight)
train$apgar1<-scale(train$apgar1)
train$apgar5<-scale(train$apgar5)
train$momage<-scale(train$momage)
train$yob<-scale(train$yob)

head(train)

## END DATA MANAGEMENT

## TABLE 1 DESCRIPTIVES

attach(m)
r1<-cbind(median(EFW),quantile(EFW, 1/4), quantile(EFW, 3/4))
r2<-cbind(median(birthweight),quantile(birthweight, 3/4), quantile(birthweight, 3/4))
r3<-cbind(median(momage),quantile(momage, 1/4), quantile(momage, 3/4))
r4<-cbind(median(gab),quantile(gab, 1/4), quantile(gab, 3/4))
r5<-cbind(median(apgar1),quantile(apgar1, 1/4), quantile(apgar1, 3/4))
r6<-cbind(median(apgar5),quantile(apgar5,1/4), quantile(apgar5, 3/4))
r7<-cbind(median(yob),quantile(yob, 1/4), quantile(yob, 3/4))
detach(m);r1
attach(mm)
s1<-cbind(median(EFW),quantile(EFW, 1/4), quantile(EFW, 3/4))
s2<-cbind(median(birthweight),quantile(birthweight, 1/4), quantile(birthweight, 3/4))
s3<-cbind(median(momage),quantile(momage, 1/4), quantile(momage, 3/4))
s4<-cbind(median(gab),quantile(gab, 1/4), quantile(gab, 3/4))
s5<-cbind(median(apgar1),quantile(apgar1, 1/4), quantile(apgar1, 3/4))
s6<-cbind(median(apgar5),quantile(apgar5,1/4), quantile(apgar5, 3/4))
s7<-cbind(median(yob),quantile(yob, 1/4), quantile(yob, 3/4))
detach(mm);s7

rs1<-cbind("EFW",r1,s1);rs2<-cbind("Birthweight",r2,s2);rs3<-cbind("Maternal Age",r3,s3);
rs4<-cbind("Gestational Age at Birth",r4,s4);rs5<-cbind("1 min Apgar",r5,s5);
rs6<-cbind("5 min Apgar",r6,s6);rs7<-cbind("Year of Birth",r7,s7);

## TABLE 1
library(xtable)
xtable(rbind(rs1,rs2,rs3,rs4,rs5,rs6,rs7))

## eAPPENDIX SUMMARY RESULTS
names(m)
summary(m$gestweek);summary(m$gab);summary(m$birthweight);table(m$race);
summary(subset(m$momage,m$momage<60));summary(m$apgar5);table(m$smoke);summary(m$yob)


## RANDOM FOREST
set.seed(123)
# y<-as.matrix(train$EFW)
# X<-as.matrix(cbind(train$gestweek,train$birthweight,train$momage,train$gab,train$apgar1,train$apgar5,as.numeric(train$yob),
#          as.factor(train$gender),as.factor(train$smoke),as.factor(train$livebirth),as.factor(train$white),
#          as.factor(train$black),as.factor(train$hispanic)))
# y<-y[1:500,]
# X<-X[1:500,];names(X)<-c("gestweek","birthweight","momage","gab","apgar1","apgar5","yob","gender","smoke","livebirth","white","black","hispanic")
# rf <- quantregForest(X,y,importance = T, mtry=8, ntree=5000)
# importance(rf, quantiles=.5)
head(train)
y0<-subset(train$EFW,train$smoke==0);length(y0)
y1<-subset(train$EFW,train$smoke==1);length(y1)
train0<-subset(train,train$smoke==0)
train1<-subset(train,train$smoke==1)
X0<-cbind(train0$gestweek,train0$birthweight,train0$momage,train0$gab,train0$apgar1,train0$apgar5,as.numeric(train0$rank_pregid),as.numeric(train0$yob),
                       as.factor(train0$gender),as.factor(train0$white),as.factor(train0$black),as.factor(train0$hispanic))
X1<-cbind(train1$gestweek,train1$birthweight,train1$momage,train1$gab,train1$apgar1,train1$apgar5,as.numeric(train1$rank_pregid),as.numeric(train1$yob),
          as.factor(train1$gender),as.factor(train1$white),as.factor(train1$black),as.factor(train1$hispanic))
rf0 <- quantregForest(X0,y0,importance = F, mtry=8, ntree=5000)
rf1 <- quantregForest(X1,y1,importance = F, mtry=8, ntree=5000)
beep(8)

saveRDS(rf,"~/PredictFGR/RandomForest_v2_smk0.rds")
rf1<-readRDS("~/PredictFGR/RandomForest_v2_smk1.rds")
rf0<-readRDS("~/PredictFGR/RandomForest_v2_smk0.rds")

test0<-subset(test,test$smoke==0);nrow(test0)
test1<-subset(test,test$smoke==1);nrow(test1);names(test1)
X2_0<-cbind(test0$gestweek,test0$birthweight,test0$momage,test0$gab,test0$apgar1,test0$apgar5,as.numeric(test0$rank_pregid),as.numeric(test0$yob),
         as.factor(test0$gender),as.factor(test0$white), as.factor(test0$black),as.factor(test0$hispanic))
X2_1<-cbind(test1$gestweek,test1$birthweight,test1$momage,test1$gab,test1$apgar1,test1$apgar5,as.numeric(test1$rank_pregid),as.numeric(test1$yob),
            as.factor(test1$gender),as.factor(test1$white),as.factor(test1$black),as.factor(test1$hispanic))
predRF0 <- predict(rf0,X2_0,quantiles=0.1*(1:9))
predRF1 <- predict(rf1,X2_1,quantiles=0.1*(1:9))

col.list0<-round(runif(nrow(test0),min=.1,max=.9),1)*10
col.list1<-round(runif(nrow(test1),min=.1,max=.9),1)*10
nrow(predRF1)
EFW_rfQ0<-as.matrix(predRF0[cbind(1:nrow(predRF0),col.list0)])
EFW_rfQ1<-as.matrix(predRF1[cbind(1:nrow(predRF1),col.list1)])

tt<-rbind(EFW_rfQ0,EFW_rfQ1)
nrow(test)
test$EFW_rfQ<-tt

head(test)
plot(test$EFW_rfQ,test$EFW_rf)

test$EFW_rf<-predRF[,2]
test$UPL_rf<-predRF[,3]
test$LPL_rf<-predRF[,1]

head(mm)
X3<-cbind(mm$gestweek,mm$birthweight,mm$momage,mm$gab,mm$apgar1,mm$apgar5,as.numeric(mm$yob),
          as.factor(mm$gender),as.factor(mm$smoke),as.factor(mm$white),
          as.factor(mm$black),as.factor(mm$hispanic))
OSpred<-predict(rf,X3,predict.all=F,quantiles=c(.025,.5,.975))

mm$EFW_rf<-OSpred[,2]
mm$diffind<-as.numeric(abs(mm$EFW - mm$EFW_rf)>150)+1;table(mm$diffind)
test$cover_rf<-ifelse(abs(test$EFW-test$EFW_rf)<1000,1,2)

plot(test$EFW,test$EFW_rfQ,xlim=c(0,6000),ylim=c(0,6000),tcl=-.1,las=1,ylab="Predicted  EFW (kg)",xlab="Actual EFW (kg)",pch=20,cex=.5,col=c("blue","red")[test$cover_rf])
points(mm$EFW,mm$EFW_rf,pch=20,cex=.5,col="red",xlim=c(0,6000),ylim=c(0,6000))
lines(x = c(0,6000), y = c(0,6000),lty=2)
loess1<-loess.smooth(test$EFW,test$EFW_rfQ,span=.75,degree=2,family="gaussian")
lines(loess1,lwd=2,col="black")

head(test);head(mm)
write.table(mm,file="~/PredictFGR/ODNSR_data.txt",sep="\t")
write.table(test,file="~/PredictFGR/test_data_v2.txt",sep="\t")

## QUANTILE REGRESSION
qr <- rq(EFW ~ ns(gestweek,df=2)+ns(birthweight,df=2)+ns(apgar5,df=3)+ns(apgar5,df=3)+white+black+ns(momage,df=3)+
            ns(gab,df=3)+gender+smoke+ns(yob,df=3), 
            tau = c(.025,.5,.975), data = train)
predQR <- predict(qr,test)
test$EFW_qr<-predQR[,2]
test$LPL_qr<-predQR[,1]
test$UPL_qr<-predQR[,3]

OSpredQR<-predict(qr,mm)

mm$EFW_qr<-OSpredQR[,2]
mm$diffind<-as.numeric(abs(mm$EFW - mm$EFW_qr)>150)+1;table(mm$diffind)
test$cover_qr<-ifelse(abs(test$EFW-test$EFW_qr)<1000,1,2)

plot(test$EFW,test$EFW_qr,xlim=c(0,6000),ylim=c(0,6000),tcl=-.1,las=1,ylab="Predicted  EFW (kg)",xlab="Actual EFW (kg)",pch=20,cex=.5,col=c("blue","red")[test$cover_rf])
points(mm$EFW,mm$EFW_qr,pch=20,col="red")
lines(x = c(0,6000), y = c(0,6000),lty=2)

saveRDS(qr,"~/PredictFGR/QuantileRegression.rds")

head(test)
head(mm)
write.table(mm,file="~/PredictFGR/ODNSR_data.txt",sep="\t")
write.table(test,file="~/PredictFGR/test_data_v2.txt",sep="\t")

## GENERALIZED BOOSTED MODELS
Y0<-as.matrix(log(train$EFW))
X0<-as.matrix(cbind(train$gestweek,train$birthweight,train$momage,train$gab,train$apgar1,train$apgar5,as.numeric(train$yob),
                    as.factor(train$gender),as.factor(train$smoke),as.factor(train$white),
                    as.factor(train$black),as.factor(train$hispanic)))
X2<-as.matrix(cbind(test$gestweek,test$birthweight,test$momage,test$gab,test$apgar1,test$apgar5,as.numeric(test$yob),
                    as.factor(test$gender),as.factor(test$smoke),as.factor(test$white),
                    as.factor(test$black),as.factor(test$hispanic)))
X3<-as.matrix(cbind(mm$gestweek,mm$birthweight,mm$momage,mm$gab,mm$apgar1,mm$apgar5,as.numeric(mm$yob),
                    as.factor(mm$gender),as.factor(mm$smoke),as.factor(mm$white),
                    as.factor(mm$black),as.factor(mm$hispanic)))
Z<-as.data.frame(cbind(Y0,X0))
head(Z)
bm<-gbm(V1~V2+V3+V4+V5+V6+V7+V8+V9+V10+V11+V12+V13,
        data=Z,
        var.monotone=c(0,0,0,0,0,0,0,0,0,0,0,0),
        distribution="gaussian",
        n.trees=5000,
        shrinkage=0.005,
        interaction.depth=3,
        bag.fraction = 0.5,
        train.fraction = 0.5,
        n.minobsinnode = 10,
        cv.folds = 5,
        keep.data=TRUE,
        verbose=TRUE,
        n.cores=1)
X2<-cbind(1,X2);X2<-as.data.frame(X2);X2$V1<-NULL
X3<-cbind(1,X3);X3<-as.data.frame(X3);X3$V1<-NULL

gbm_b <- predict(bm,X2)
test$EFW_gbm<-exp(gbm_b)

head(test)

gbm_b0<-predict(bm,X3)
mm$EFW_gbm<-exp(gbm_b0)

mm$diffind<-as.numeric(abs(mm$EFW - mm$EFW_gbm)>150)+1;table(mm$diffind)
test$cover_gbm<-ifelse(abs(test$EFW-test$EFW_gbm)<1000,1,2)

plot(test$EFW,test$EFW_gbm,xlim=c(0,6000),ylim=c(0,6000),tcl=-.1,las=1,ylab="Predicted  EFW (kg)",xlab="Actual EFW (kg)",pch=20,cex=.5,col=c("blue","red")[test$cover_rf])
points(mm$EFW,mm$EFW_gbm,pch=20,col="red")
lines(x = c(0,6000), y = c(0,6000),lty=2)

saveRDS(bm,"~/PredictFGR/GeneralizedBoostedModels.rds")

head(test)
head(mm)
write.table(mm,file="~/PredictFGR/ODNSR_data.txt",sep="\t")
write.table(test,file="~/PredictFGR/test_data_v2.txt",sep="\t")

## GLM REGRESSION
lreg<-lm(EFW~ns(gestweek,df=2)+ns(birthweight,df=2)+ns(apgar5,df=3)+ns(apgar5,df=3)+white+black+ns(momage,df=3)+
           ns(gab,df=3)+gender+smoke+ns(yob,df=3),data=train)
GLM<-predict(lreg,test,interval=c("prediction"))
predGLM<-GLM

test$EFW_glm<-predGLM[,1]
test$LPL_glm<-predGLM[,2]
test$UPL_glm<-predGLM[,3]

GLM0<-predict(lreg,mm)
mm$EFW_glm<-GLM0

mm$diffind<-as.numeric(abs(mm$EFW - mm$EFW_glm)>150)+1;table(mm$diffind)
test$cover_glm<-ifelse(abs(test$EFW-test$EFW_glm)<1000,1,2)

plot(test$EFW,test$EFW_glm,xlim=c(0,6000),ylim=c(0,6000),tcl=-.1,las=1,ylab="Predicted  EFW (kg)",xlab="Actual EFW (kg)",pch=20,cex=.5,col=c("blue","red")[test$cover_rf])
points(mm$EFW,mm$EFW_glm,xlim=c(0,6000),pch=20,col="red")
lines(x = c(0,6000), y = c(0,6000),lty=2)

saveRDS(lreg,"~/PredictFGR/GeneralizedLinearModel.rds")

head(test)
head(mm)
write.table(mm,file="~/PredictFGR/ODNSR_data.txt",sep="\t")
write.table(test,file="~/PredictFGR/test_data_v2.txt",sep="\t")


mm <- read.table(file="~/PredictFGR/ODNSR_data.txt",sep="\t",header = T,na.strings=c(" "))
test <- read.table(file="~/PredictFGR/test_data_v2.txt",sep="\t",header = T,na.strings=c(" "))

head(mm)
head(test)
##
##
##

## BAYESIAN ADDITIVE REGRESSION TREES
set_bart_machine_num_cores(4)
y<-log(train$EFW)
X<-as.data.frame(cbind(train$gestweek,train$birthweight,train$momage,train$gab,train$apgar1,train$apgar5,as.numeric(train$yob),
                       as.factor(train$gender),as.factor(train$smoke),as.factor(train$white),
                       as.factor(train$black),as.factor(train$hispanic)))
names(X)<-c("gestweek","birthweight","momage","gab","apgar1","apgar5","yob","gender","smoke","white","black","hispanic")

#y<-as.data.frame(log(train$EFW))
#index<-sample(1:nrow(as.matrix(y)),2000)
#bmCV<-bartMachineCV(X[index,],y[index,])

brt<-bartMachine(X,y,verbose=TRUE,k=5,nu=10,q=0.75,num_trees=50)
beep(8)
summary(brt)

X2<-as.data.frame(cbind(test$gestweek,test$birthweight,test$momage,test$gab,test$apgar1,test$apgar5,as.numeric(test$yob),
                        as.factor(test$gender),as.factor(test$smoke),as.factor(test$white),
                        as.factor(test$black),as.factor(test$hispanic)))
names(X2)<-c("gestweek","birthweight","momage","gab","apgar1","apgar5","yob","gender","smoke","white","black","hispanic")
X3<-as.data.frame(cbind(mm$gestweek,mm$birthweight,mm$momage,mm$gab,mm$apgar1,mm$apgar5,as.numeric(mm$yob),
                        as.factor(mm$gender),as.factor(mm$smoke),as.factor(mm$white),
                        as.factor(mm$black),as.factor(mm$hispanic)))
names(X3)<-c("gestweek","birthweight","momage","gab","apgar1","apgar5","yob","gender","smoke","white","black","hispanic")
predBART <- predict(brt,X2)
predBART_lim<-calc_prediction_intervals(brt,X2,pi_conf=0.95,num_samples_per_data_point=5000)
predBART<-cbind(predBART,predBART_lim)

test<-read.table(file="~/PredictFGR/test_data_v2.txt",sep="\t",header = T)
mm<-read.table(file="~/PredictFGR/ODNSR_data.txt",sep="\t",header = T)

head(test);head(mm)

test$EFW_brt<-exp(predBART[,1])
test$UPL_brt<-exp(predBART[,3])
test$LPL_brt<-exp(predBART[,2])

predBART0 <- predict(brt,X3)
mm$EFW_brt<-exp(predBART0)

mm$diffind<-as.numeric(abs(mm$EFW - mm$EFW_brt)>150)+1;table(mm$diffind)
test$cover_brt<-ifelse(abs(test$EFW-test$EFW_brt)<1000,1,2)

plot(test$EFW,test$EFW_brt,xlim=c(0,6000),ylim=c(0,6000),tcl=-.1,las=1,ylab="Predicted  EFW (kg)",xlab="Actual EFW (kg)",pch=20,cex=.5,col=c("blue","red")[test$cover_brt])
points(mm$EFW,mm$EFW_brt,pch=20,col="red")
lines(x = c(0,6000), y = c(0,6000),lty=2)

saveRDS(brt,"~/PredictFGR/BARTMachine.rds")

## MSE, MAD, AND CONFIDENCE INTERVALS
## REGRESSION-BASED
rmsq <- function(data,indices) {
  d <- data[indices,] # allows boot to select sample 
  rmse<-sqrt(mean((d$EFW_glm-d$EFW)^2))
  MAD<-median(abs(d$EFW_glm-d$EFW))
  res<-cbind(rmse,MAD)
  return(res)
}
set.seed(123)
boot1<-boot(data=test,statistic=rmsq,R=2000)
plot(boot1,index=1);plot(boot1,index=2)
glm_rmse_ci<-boot.ci(boot1,type="perc",index=1)
RMSE_glm<-sqrt(mean((test$EFW_glm-test$EFW)^2))
glm_mad_ci<-boot.ci(boot1,type="perc",index=2)
MAD_glm<-median(abs(test$EFW_glm-test$EFW))

rmsq <- function(data,indices) {
  d <- data[indices,] # allows boot to select sample 
  rmse<-sqrt(mean((d$EFW_qr-d$EFW)^2))
  MAD<-median(abs(d$EFW_qr-d$EFW))
  res<-cbind(rmse,MAD)
  return(res)
}
set.seed(123)
boot1<-boot(data=test,statistic=rmsq,R=2000)
plot(boot1,index=1);plot(boot1,index=2)
qr_rmse_ci<-boot.ci(boot1,type="perc",index=1)
RMSE_qr<-sqrt(mean((test$EFW_qr-test$EFW)^2))
qr_mad_ci<-boot.ci(boot1,type="perc",index=2)
MAD_qr<-median(abs(test$EFW_qr-test$EFW))


reg_res<-as.data.frame(rbind(cbind("GLM",round(RMSE_glm,2),round(t(glm_rmse_ci$percent[1,4:5]),2),round(MAD_glm,2),round(t(glm_mad_ci$percent[1,4:5]),2)),
                             cbind("QR",round(RMSE_qr,2),round(t(qr_rmse_ci$percent[1,4:5]),2),round(MAD_qr,2),round(t(qr_mad_ci$percent[1,4:5]),2))))
names(reg_res)<-c("Method","RMSE","95% LCL","95% UCL","MAD","95% LCL","95% UCL")
reg_res

## MACHINE LEARNING
library(boot)
rmsq <- function(data,indices) {
  d <- data[indices,] # allows boot to select sample 
  rmse<-sqrt(mean((d$EFW_rf-d$EFW)^2))
  MAD<-median(abs(d$EFW_rf-d$EFW))
  res<-cbind(rmse,MAD)
  return(res)
}
set.seed(123)
boot1<-boot(data=test,statistic=rmsq,R=2000)
plot(boot1,index=1);plot(boot1,index=2)
rf_rmse_ci<-boot.ci(boot1,type="perc",index=1)
RMSE_rf<-sqrt(mean((test$EFW_rf-test$EFW)^2))
rf_mad_ci<-boot.ci(boot1,type="perc",index=2)
MAD_rf<-median(abs(test$EFW_rf-test$EFW))

rmsq <- function(data,indices) {
  d <- data[indices,] # allows boot to select sample 
  rmse<-sqrt(mean((d$EFW_brt-d$EFW)^2))
  MAD<-median(abs(d$EFW_brt-d$EFW))
  res<-cbind(rmse,MAD)
  return(res)
}
set.seed(123)
boot1<-boot(data=test,statistic=rmsq,R=2000)
plot(boot1,index=1);plot(boot1,index=2)
brt_rmse_ci<-boot.ci(boot1,type="perc",index=1)
RMSE_brt<-sqrt(mean((test$EFW_brt-test$EFW)^2))
brt_mad_ci<-boot.ci(boot1,type="perc",index=2)
MAD_brt<-median(abs(test$EFW_brt-test$EFW))

rmsq <- function(data,indices) {
  d <- data[indices,] # allows boot to select sample 
  rmse<-sqrt(mean((d$EFW_gbm-d$EFW)^2))
  MAD<-median(abs(d$EFW_gbm-d$EFW))
  res<-cbind(rmse,MAD)
  return(res)
}
set.seed(123)
boot1<-boot(data=test,statistic=rmsq,R=2000)
plot(boot1,index=1);plot(boot1,index=2)
gbm_rmse_ci<-boot.ci(boot1,type="perc",index=1)
RMSE_gbm<-sqrt(mean((test$EFW_gbm-test$EFW)^2))
gbm_mad_ci<-boot.ci(boot1,type="perc",index=2)
MAD_gbm<-median(abs(test$EFW_gbm-test$EFW))

ml_res<-as.data.frame(rbind(cbind("RF",round(RMSE_rf,2),round(t(rf_rmse_ci$percent[1,4:5]),2),round(MAD_rf,2),round(t(rf_mad_ci$percent[1,4:5]),2)),
                            cbind("BART",round(RMSE_brt,2),round(t(brt_rmse_ci$percent[1,4:5]),2),round(MAD_brt,2),round(t(brt_mad_ci$percent[1,4:5]),2)),
                            cbind("GBM",round(RMSE_gbm,2),round(t(gbm_rmse_ci$percent[1,4:5]),2),round(MAD_gbm,2),round(t(gbm_mad_ci$percent[1,4:5]),2))))
names(ml_res)<-c("Method","RMSE","95% LCL","95% UCL","MAD","95% LCL","95% UCL")
ml_res
library(xtable)
xtable(rbind(reg_res,ml_res))
beep(8)

## MSE, MAD, AND CONFIDENCE INTERVALS
## REGRESSION-BASED
rmsq <- function(data,indices) {
  d <- data[indices,] # allows boot to select sample 
  rmse<-sqrt(mean((d$EFW_glm-d$EFW)^2))
  MAD<-median(abs(d$EFW_glm-d$EFW))
  res<-cbind(rmse,MAD)
  return(res)
}
set.seed(123)
boot1<-boot(data=mm,statistic=rmsq,R=2000)
plot(boot1,index=1);plot(boot1,index=2)
glm_rmse_ci<-boot.ci(boot1,type="perc",index=1)
RMSE_glm<-sqrt(mean((mm$EFW_glm-mm$EFW)^2))
glm_mad_ci<-boot.ci(boot1,type="perc",index=2)
MAD_glm<-median(abs(mm$EFW_glm-mm$EFW))

rmsq <- function(data,indices) {
  d <- data[indices,] # allows boot to select sample 
  rmse<-sqrt(mean((d$EFW_qr-d$EFW)^2))
  MAD<-median(abs(d$EFW_qr-d$EFW))
  res<-cbind(rmse,MAD)
  return(res)
}
set.seed(123)
boot1<-boot(data=mm,statistic=rmsq,R=2000)
plot(boot1,index=1);plot(boot1,index=2)
qr_rmse_ci<-boot.ci(boot1,type="perc",index=1)
RMSE_qr<-sqrt(mean((mm$EFW_qr-mm$EFW)^2))
qr_mad_ci<-boot.ci(boot1,type="perc",index=2)
MAD_qr<-median(abs(mm$EFW_qr-mm$EFW))

reg_res<-as.data.frame(rbind(cbind("GLM",round(RMSE_glm,2),round(t(glm_rmse_ci$percent[1,4:5]),2),round(MAD_glm,2),round(t(glm_mad_ci$percent[1,4:5]),2)),
                             cbind("QR",round(RMSE_qr,2),round(t(qr_rmse_ci$percent[1,4:5]),2),round(MAD_qr,2),round(t(qr_mad_ci$percent[1,4:5]),2))))
names(reg_res)<-c("Method","RMSE","95% LCL","95% UCL","MAD","95% LCL","95% UCL")
reg_res

## MACHINE LEARNING
rmsq <- function(data,indices) {
  d <- data[indices,] # allows boot to select sample 
  rmse<-sqrt(mean((d$EFW_rf-d$EFW)^2))
  MAD<-median(abs(d$EFW_rf-d$EFW))
  res<-cbind(rmse,MAD)
  return(res)
}
set.seed(123)
boot1<-boot(data=mm,statistic=rmsq,R=2000)
plot(boot1,index=1);plot(boot1,index=2)
rf_rmse_ci<-boot.ci(boot1,type="perc",index=1)
RMSE_rf<-sqrt(mean((mm$EFW_rf-mm$EFW)^2))
rf_mad_ci<-boot.ci(boot1,type="perc",index=2)
MAD_rf<-median(abs(mm$EFW_rf-mm$EFW))

rmsq <- function(data,indices) {
  d <- data[indices,] # allows boot to select sample 
  rmse<-sqrt(mean((d$EFW_brt-d$EFW)^2))
  MAD<-median(abs(d$EFW_brt-d$EFW))
  res<-cbind(rmse,MAD)
  return(res)
}
set.seed(123)
boot1<-boot(data=mm,statistic=rmsq,R=2000)
plot(boot1,index=1);plot(boot1,index=2)
brt_rmse_ci<-boot.ci(boot1,type="perc",index=1)
RMSE_brt<-sqrt(mean((mm$EFW_brt-mm$EFW)^2))
brt_mad_ci<-boot.ci(boot1,type="perc",index=2)
MAD_brt<-median(abs(mm$EFW_brt-mm$EFW))

rmsq <- function(data,indices) {
  d <- data[indices,] # allows boot to select sample 
  rmse<-sqrt(mean((d$EFW_gbm-d$EFW)^2))
  MAD<-median(abs(d$EFW_gbm-d$EFW))
  res<-cbind(rmse,MAD)
  return(res)
}
set.seed(123)
boot1<-boot(data=mm,statistic=rmsq,R=2000)
plot(boot1,index=1);plot(boot1,index=2)
gbm_rmse_ci<-boot.ci(boot1,type="perc",index=1)
RMSE_gbm<-sqrt(mean((mm$EFW_gbm-mm$EFW)^2))
gbm_mad_ci<-boot.ci(boot1,type="perc",index=2)
MAD_gbm<-median(abs(mm$EFW_gbm-mm$EFW))

ml_res<-as.data.frame(rbind(cbind("RF",round(RMSE_rf,2),round(t(rf_rmse_ci$percent[1,4:5]),2),round(MAD_rf,2),round(t(rf_mad_ci$percent[1,4:5]),2)),
                            cbind("BART",round(RMSE_brt,2),round(t(brt_rmse_ci$percent[1,4:5]),2),round(MAD_brt,2),round(t(brt_mad_ci$percent[1,4:5]),2)),
                            cbind("GBM",round(RMSE_gbm,2),round(t(gbm_rmse_ci$percent[1,4:5]),2),round(MAD_gbm,2),round(t(gbm_mad_ci$percent[1,4:5]),2))))
names(ml_res)<-c("Method","RMSE","95% LCL","95% UCL","MAD","95% LCL","95% UCL")
ml_res
library(xtable)
xtable(rbind(reg_res,ml_res))
beep(8)

## PLOTS
pdf(file="~/PredictFGR/scatter_table_v5.pdf",width=8,height=11.5)
#png(file="~/PredictFGR/scatter_table_v3.png",width=540,height=600)
test<-read.table(file="~/PredictFGR/test_data_v2.txt",sep="\t",header = T)
mm<-read.table(file="~/PredictFGR/ODNSR_data.txt",sep="\t",header = T)

test$EFW<-test$EFW/1000;test$EFW_rf<-test$EFW_rf/1000;test$EFW_gbm<-test$EFW_gbm/1000;test$EFW_brt<-test$EFW_brt/1000;
test$EFW_glm<-test$EFW_glm/1000;test$EFW_qr<-test$EFW_qr/1000;

mm$EFW<-mm$EFW/1000;mm$EFW_rf<-mm$EFW_rf/1000;mm$EFW_gbm<-mm$EFW_gbm/1000;mm$EFW_brt<-mm$EFW_brt/1000;
mm$EFW_glm<-mm$EFW_glm/1000;mm$EFW_qr<-mm$EFW_qr/1000;

test$cover_rf<-ifelse(abs(test$EFW-test$EFW_rf)<1,1,2)
test$cover_qr<-ifelse(abs(test$EFW-test$EFW_qr)<1,1,2)
test$cover_brt<-ifelse(abs(test$EFW-test$EFW_brt)<1,1,2)
test$cover_glm<-ifelse(abs(test$EFW-test$EFW_glm)<1,1,2)
test$cover_gbm<-ifelse(abs(test$EFW-test$EFW_gbm)<1,1,2)

pp<-c(1,4);c=c("black");c2=c("gray50","red")

par(cex=1.05,mai=c(.95,.95,.25,.25))
#split screen in two columns:
split.screen(c(1, 2))
#split first column in three rows:
split.screen(c(3, 1), screen = 1)
#split second column in two screens with specific dimensions:
coord <- matrix(c(c(0, 1, .15, 0.5), c(0, 1, 0.5, .85)), byrow=T, ncol=4)
split.screen(coord,screen=2)

screen(3)
plot(0,0,xlim=c(0,6),ylim=c(0,6),tcl=-.1,las=1,ylab="Predicted  EFW (kg)",xlab="Actual EFW (kg)",cex=.5,
     col="white")
points(test$EFW,test$EFW_rf,pch=pp[test$cover_rf],cex=.25,col=c2[test$cover_rf])
points(mm$EFW,mm$EFW_rf,pch=17,cex=.5,col=c("cyan"))
lines(x = c(0,6), y = c(0,6),col="black",lty=1)
loess1<-loess.smooth(test$EFW,test$EFW_rf,span=.75,degree=2,family="gaussian")
lines(loess1,lwd=1,col="blue")
text(4,.5,substitute(paste(rho,"=",v),list(v=round(cor(test$EFW,test$EFW_rf),3))))


# x.coord<-c(loess1$x,rev(seq(0,max(test$EFW),length.out=nrow(test))))
# y.coord<-c(loess1$y,rev(seq(0,max(test$EFW),length.out=nrow(test))))
# polygon(x.coord,y.coord,col=rgb(0, 1, 1,0.25))
text(.85,5,"Random \n     Forest")

screen(4)
plot(0,0,xlim=c(0,6),ylim=c(0,6),tcl=-.1,las=1,ylab="Predicted  EFW (kg)",xlab="Actual EFW (kg)",cex=.5,
     col="white")
points(test$EFW,test$EFW_brt,pch=pp[test$cover_brt],cex=.25,col=c2[test$cover_brt])
points(mm$EFW,mm$EFW_brt,pch=17,cex=.5,col=c("cyan"))
lines(x = c(0,6), y = c(0,6),col="black",lty=1)
loess1<-loess.smooth(test$EFW,test$EFW_brt,span=.75,degree=2,family="gaussian")
lines(loess1,lwd=1,col="blue")
text(4,.5,substitute(paste(rho,"=",v),list(v=round(cor(test$EFW,test$EFW_brt),3))))

# x.coord<-c(loess1$x,rev(seq(0,max(test$EFW),length.out=nrow(test))))
# y.coord<-c(loess1$y,rev(seq(0,max(test$EFW),length.out=nrow(test))))
# polygon(x.coord,y.coord,col=rgb(0, 1, 1,0.25))
text(.85,5,"BART")

screen(5)
plot(0,0,xlim=c(0,6),ylim=c(0,6),tcl=-.1,las=1,ylab="Predicted  EFW (kg)",xlab="Actual EFW (kg)",cex=.5,
     col="white")
points(test$EFW,test$EFW_gbm,pch=pp[test$cover_gbm],cex=.25,col=c2[test$cover_gbm])
points(mm$EFW,mm$EFW_gbm,pch=17,cex=.5,col=c("cyan"))
lines(x = c(0,6), y = c(0,6),col="black",lty=1)
loess1<-loess.smooth(test$EFW,test$EFW_gbm,span=.75,degree=2,family="gaussian")
lines(loess1,lwd=1,col="blue")
text(4,.5,substitute(paste(rho,"=",v),list(v=round(cor(test$EFW,test$EFW_gbm),3))))

# x.coord<-c(loess1$x,rev(seq(0,max(test$EFW),length.out=nrow(test))))
# y.coord<-c(loess1$y,rev(seq(0,max(test$EFW),length.out=nrow(test))))
# polygon(x.coord,y.coord,col=rgb(0, 1, 1,0.25))
text(1,5,"Generalized \n     Boosted \n               Models")

screen(6)
plot(0,0,xlim=c(0,6),ylim=c(0,6),tcl=-.1,las=1,ylab="Predicted  EFW (kg)",xlab="Actual EFW (kg)",cex=.5,
     col="white")
points(test$EFW,test$EFW_qr,pch=pp[test$cover_qr],cex=.25,col=c2[test$cover_qr])
points(mm$EFW,mm$EFW_qr,pch=17,cex=.5,col=c("cyan"))
lines(x = c(0,6), y = c(0,6),col="black",lty=1)
loess1<-loess.smooth(test$EFW,test$EFW_qr,span=.75,degree=2,family="gaussian")
lines(loess1,lwd=1,col="blue")
text(4,.5,substitute(paste(rho,"=",v),list(v=round(cor(test$EFW,test$EFW_qr),3))))

# x.coord<-c(ifelse(loess1$x<3,3,loess1$x),rev(seq(3,max(test$EFW),length.out=nrow(test))))
# y.coord<-c(ifelse(loess1$y<3,3,loess1$y),rev(seq(3,max(test$EFW),length.out=nrow(test))))
# polygon(x.coord,y.coord,col=rgb(1, 0, 1,0.25))
# 
# x.coord<-c(ifelse(loess1$x>3,3,loess1$x),rev(seq(min(test$EFW),3,length.out=nrow(test))))
# y.coord<-c(ifelse(loess1$y>3,3,loess1$y),rev(seq(min(test$EFW),3,length.out=nrow(test))))
# polygon(x.coord,y.coord,col=rgb(0, 1, 1,0.25))
text(.85,5,"Quantile \n          Regression")

screen(7)
plot(0,0,xlim=c(0,6),ylim=c(0,6),tcl=-.1,las=1,ylab="Predicted  EFW (kg)",xlab="Actual EFW (kg)",cex=.5,
     col="white")
points(test$EFW,test$EFW_glm,pch=pp[test$cover_glm],cex=.25,col=c2[test$cover_glm])
points(mm$EFW,mm$EFW_glm,pch=17,cex=.5,col=c("cyan"))
lines(x = c(0,6), y = c(0,6),col="black",lty=1)
loess1<-loess.smooth(test$EFW,test$EFW_qr,span=.75,degree=2,family="gaussian")
lines(loess1,lwd=1,col="blue")
text(4,.5,substitute(paste(rho,"=",v),list(v=round(cor(test$EFW,test$EFW_glm),3))))

# x.coord<-c(ifelse(loess1$x<3,3,loess1$x),rev(seq(3,max(test$EFW),length.out=nrow(test))))
# y.coord<-c(ifelse(loess1$y<3,3,loess1$y),rev(seq(3,max(test$EFW),length.out=nrow(test))))
# polygon(x.coord,y.coord,col=rgb(1, 0, 0,0.25),border=NULL)
# 
# x.coord<-c(ifelse(loess1$x>3,3,loess1$x),rev(seq(min(test$EFW),3,length.out=nrow(test))))
# y.coord<-c(ifelse(loess1$y>3,3,loess1$y),rev(seq(min(test$EFW),3,length.out=nrow(test))))
# polygon(x.coord,y.coord,col=rgb(0, 1, 1,0.25),border=NULL)
text(1,5,"Generalized \n     Linear \n             Model")
dev.off()
close.screen(all = TRUE) 

max(test$gestweek2)
xtxt<-c("20","24","28","32","36","30","32","42")

library(quantreg);library(sampling)
pdf(file="~/PredictFGR/qr_plot_v3sens.pdf",width=8,height=11.5)
test<-read.table(file="~/PredictFGR/test_data_v2.txt",sep="\t",header = T)
#cll<-cluster(data=test,clustername="rank_pregid",size=length(unique(test$rank_pregid)),method="srswr")
#test<-getdata(test,cll)

#split screen in two columns:
split.screen(c(1,2))
#split first column in three rows:
split.screen(c(3,1),screen=1)
#split second column in three rows:
split.screen(c(3,1),screen=2)

screen(3)
ylims<-c(-.2,.6);B<-200;test$y<-test$EFW/1000;spn<-.75
test0<-subset(test,test$smoke==0)
test1<-subset(test,test$smoke==1)
rq0<-rqss(y~qss(gestweek,lambda=.75),tau=quant,data=test0)
rq1<-rqss(y~qss(gestweek,lambda=.75),tau=quant,data=test1)
x1<-unique(test1$gestweek)
x1<-x1[order(x1)]
length(x1)
qdiff<-rq0$coef[2:length(x1)]-rq1$coef[2:length(x1)]
lo<-loess(qdiff~x1[-1],span=spn)
plot(predict(lo),ylim=ylims,type="l",las=1,tcl=-.1,lwd=3,col="white",
     ylab="Quantile Difference (kg)",xlab="Gestational Week",xaxt="n")
rect(par("usr")[1],par("usr")[3],par("usr")[2],par("usr")[4],col = "gray95")
mtext("Empirical EFW", 3, line=1)
axis(1,at=seq(0,140,20),labels=c("20","24","28","32","36","30","32","42"),tcl=-.1)
quant<-.5;col1a<-rgb(128/255,128/255,128/255,alpha=0.3);
for(b in 1:B){
  clust<-cluster(data=test,clustername="rank_pregid",size=length(unique(test$rank_pregid)),method="srswr")
  testB<-getdata(test,clust)
  test0<-subset(testB,testB$smoke==0)
  test1<-subset(testB,testB$smoke==1)
  rq0<-rqss(y~qss(gestweek,lambda=.75),tau=quant,data=test0)
  rq1<-rqss(y~qss(gestweek,lambda=.75),tau=quant,data=test1)
  x1<-unique(test1$gestweek)
  x1<-x1[order(x1)]
  length(x1)
  qdiff<-rq0$coef[2:length(x1)]-rq1$coef[2:length(x1)]
  lo<-loess(qdiff~x1[-1],span=spn)
  lines(predict(lo),ylim=ylims,lwd=1,col=col1a)
}
quant<-.25;col1a<-rgb(1,20/255,60/255,alpha=0.2);
for(b in 1:B){
  clust<-cluster(data=test,clustername="rank_pregid",size=length(unique(test$rank_pregid)),method="srswr")
  testB<-getdata(test,clust)
  test0<-subset(testB,testB$smoke==0)
  test1<-subset(testB,testB$smoke==1)
  rq0<-rqss(y~qss(gestweek,lambda=.75),tau=quant,data=test0)
  rq1<-rqss(y~qss(gestweek,lambda=.75),tau=quant,data=test1)
  x1<-unique(test1$gestweek)
  x1<-x1[order(x1)]
  length(x1)
  qdiff<-rq0$coef[2:length(x1)]-rq1$coef[2:length(x1)]
  lo<-loess(qdiff~x1[-1],span=spn)
  lines(predict(lo),ylim=ylims,lwd=1,col=col1a)
}
quant<-.75;col1a<-rgb(0,191/255,1,alpha=0.2);
for(b in 1:B){
  clust<-cluster(data=test,clustername="rank_pregid",size=length(unique(test$rank_pregid)),method="srswr")
  testB<-getdata(test,clust)
  test0<-subset(testB,testB$smoke==0)
  test1<-subset(testB,testB$smoke==1)
  rq0<-rqss(y~qss(gestweek,lambda=.75),tau=quant,data=test0)
  rq1<-rqss(y~qss(gestweek,lambda=.75),tau=quant,data=test1)
  x1<-unique(test1$gestweek)
  x1<-x1[order(x1)]
  length(x1)
  qdiff<-rq0$coef[2:length(x1)]-rq1$coef[2:length(x1)]
  lo<-loess(qdiff~x1[-1],span=spn)
  lines(predict(lo),ylim=ylims,lwd=1,col=col1a)
}

quant<-.5;col1b<-"black"
test0<-subset(test,test$smoke==0)
test1<-subset(test,test$smoke==1)
rq0<-rqss(y~qss(gestweek,lambda=.75),tau=quant,data=test0)
rq1<-rqss(y~qss(gestweek,lambda=.75),tau=quant,data=test1)
x1<-unique(test1$gestweek)
x1<-x1[order(x1)]
length(x1)
qdiff<-rq0$coef[2:length(x1)]-rq1$coef[2:length(x1)]
lo<-loess(qdiff~x1[-1],span=spn)
lines(predict(lo),ylim=ylims,lwd=3,col=col1b)
quant<-.25;col1b<-"red"
test0<-subset(test,test$smoke==0)
test1<-subset(test,test$smoke==1)
rq0<-rqss(y~qss(gestweek,lambda=.75),tau=quant,data=test0)
rq1<-rqss(y~qss(gestweek,lambda=.75),tau=quant,data=test1)
x1<-unique(test1$gestweek)
x1<-x1[order(x1)]
length(x1)
qdiff<-rq0$coef[2:length(x1)]-rq1$coef[2:length(x1)]
lo<-loess(qdiff~x1[-1],span=spn)
lines(predict(lo),ylim=ylims,lwd=3,col=col1b)
quant<-.75;col1b<-"blue"
test0<-subset(test,test$smoke==0)
test1<-subset(test,test$smoke==1)
rq0<-rqss(y~qss(gestweek,lambda=.75),tau=quant,data=test0)
rq1<-rqss(y~qss(gestweek,lambda=.75),tau=quant,data=test1)
x1<-unique(test1$gestweek)
x1<-x1[order(x1)]
length(x1)
qdiff<-rq0$coef[2:length(x1)]-rq1$coef[2:length(x1)]
lo<-loess(qdiff~x1[-1],span=spn)
lines(predict(lo),ylim=ylims,lwd=3,col=col1b)
legend("topleft", inset=.05, title="Quantile",lty=c(1,1,1),lwd=c(3,3,3),col=c("red","black","blue"),
       c("0.25","0.50","0.75"), horiz=F,bty="n")

screen(4)
test$y<-test$EFW_glm/1000
test0<-subset(test,test$smoke==0)
test1<-subset(test,test$smoke==1)
rq0<-rqss(y~qss(gestweek,lambda=.75),tau=quant,data=test0)
rq1<-rqss(y~qss(gestweek,lambda=.75),tau=quant,data=test1)
x1<-unique(test1$gestweek)
x1<-x1[order(x1)]
length(x1)
qdiff<-rq0$coef[2:length(x1)]-rq1$coef[2:length(x1)]
lo<-loess(qdiff~x1[-1],span=spn)
plot(predict(lo),ylim=ylims,type="l",las=1,tcl=-.1,lwd=3,col="white",
     ylab="Quantile Difference (kg)",xlab="Gestational Week",xaxt="n")
mtext("Predicted EFW (GLM)", 3, line=1)
axis(1,at=seq(0,140,20),labels=c("20","24","28","32","36","30","32","42"),tcl=-.1)
quant<-.5;col1a<-rgb(128/255,128/255,128/255,alpha=0.3);
for(b in 1:B){
  clust<-cluster(data=test,clustername="rank_pregid",size=length(unique(test$rank_pregid)),method="srswr")
  testB<-getdata(test,clust)
  test0<-subset(testB,testB$smoke==0)
  test1<-subset(testB,testB$smoke==1)
  rq0<-rqss(y~qss(gestweek,lambda=.75),tau=quant,data=test0)
  rq1<-rqss(y~qss(gestweek,lambda=.75),tau=quant,data=test1)
  x1<-unique(test1$gestweek)
  x1<-x1[order(x1)]
  length(x1)
  qdiff<-rq0$coef[2:length(x1)]-rq1$coef[2:length(x1)]
  lo<-loess(qdiff~x1[-1],span=spn)
  lines(predict(lo),ylim=ylims,lwd=1,col=col1a)
}
quant<-.25;col1a<-rgb(1,20/255,60/255,alpha=0.2);
for(b in 1:B){
  clust<-cluster(data=test,clustername="rank_pregid",size=length(unique(test$rank_pregid)),method="srswr")
  testB<-getdata(test,clust)
  test0<-subset(testB,testB$smoke==0)
  test1<-subset(testB,testB$smoke==1)
  rq0<-rqss(y~qss(gestweek,lambda=.75),tau=quant,data=test0)
  rq1<-rqss(y~qss(gestweek,lambda=.75),tau=quant,data=test1)
  x1<-unique(test1$gestweek)
  x1<-x1[order(x1)]
  length(x1)
  qdiff<-rq0$coef[2:length(x1)]-rq1$coef[2:length(x1)]
  lo<-loess(qdiff~x1[-1],span=spn)
  lines(predict(lo),ylim=ylims,lwd=1,col=col1a)
}
quant<-.75;col1a<-rgb(0,191/255,1,alpha=0.2);
for(b in 1:B){
  clust<-cluster(data=test,clustername="rank_pregid",size=length(unique(test$rank_pregid)),method="srswr")
  testB<-getdata(test,clust)
  test0<-subset(testB,testB$smoke==0)
  test1<-subset(testB,testB$smoke==1)
  rq0<-rqss(y~qss(gestweek,lambda=.75),tau=quant,data=test0)
  rq1<-rqss(y~qss(gestweek,lambda=.75),tau=quant,data=test1)
  x1<-unique(test1$gestweek)
  x1<-x1[order(x1)]
  length(x1)
  qdiff<-rq0$coef[2:length(x1)]-rq1$coef[2:length(x1)]
  lo<-loess(qdiff~x1[-1],span=spn)
  lines(predict(lo),ylim=ylims,lwd=1,col=col1a)
}

quant<-.5;col1b<-"black"
test0<-subset(test,test$smoke==0)
test1<-subset(test,test$smoke==1)
rq0<-rqss(y~qss(gestweek,lambda=.75),tau=quant,data=test0)
rq1<-rqss(y~qss(gestweek,lambda=.75),tau=quant,data=test1)
x1<-unique(test1$gestweek)
x1<-x1[order(x1)]
length(x1)
qdiff<-rq0$coef[2:length(x1)]-rq1$coef[2:length(x1)]
lo<-loess(qdiff~x1[-1],span=spn)
lines(predict(lo),ylim=ylims,lwd=3,col=col1b)
quant<-.25;col1b<-"red"
test0<-subset(test,test$smoke==0)
test1<-subset(test,test$smoke==1)
rq0<-rqss(y~qss(gestweek,lambda=.75),tau=quant,data=test0)
rq1<-rqss(y~qss(gestweek,lambda=.75),tau=quant,data=test1)
x1<-unique(test1$gestweek)
x1<-x1[order(x1)]
length(x1)
qdiff<-rq0$coef[2:length(x1)]-rq1$coef[2:length(x1)]
lo<-loess(qdiff~x1[-1],span=spn)
lines(predict(lo),ylim=ylims,lwd=3,col=col1b)
quant<-.75;col1b<-"blue"
test0<-subset(test,test$smoke==0)
test1<-subset(test,test$smoke==1)
rq0<-rqss(y~qss(gestweek,lambda=.75),tau=quant,data=test0)
rq1<-rqss(y~qss(gestweek,lambda=.75),tau=quant,data=test1)
x1<-unique(test1$gestweek)
x1<-x1[order(x1)]
length(x1)
qdiff<-rq0$coef[2:length(x1)]-rq1$coef[2:length(x1)]
lo<-loess(qdiff~x1[-1],span=spn)
lines(predict(lo),ylim=ylims,lwd=3,col=col1b)
legend("topleft", inset=.05, title="Quantile",lty=c(1,1,1),lwd=c(3,3,3),col=c("red","black","blue"),
       c("0.25","0.50","0.75"), horiz=F,bty="n")

screen(5)
test$y<-test$EFW_qr/1000
test0<-subset(test,test$smoke==0)
test1<-subset(test,test$smoke==1)
rq0<-rqss(y~qss(gestweek,lambda=.75),tau=quant,data=test0)
rq1<-rqss(y~qss(gestweek,lambda=.75),tau=quant,data=test1)
x1<-unique(test1$gestweek)
x1<-x1[order(x1)]
length(x1)
qdiff<-rq0$coef[2:length(x1)]-rq1$coef[2:length(x1)]
lo<-loess(qdiff~x1[-1],span=spn)
plot(predict(lo),ylim=ylims,type="l",las=1,tcl=-.1,lwd=3,col="white",
     ylab="Quantile Difference (kg)",xlab="Gestational Week",xaxt="n")
mtext("Predicted EFW (QR)", 3, line=1)
axis(1,at=seq(0,140,20),labels=c("20","24","28","32","36","30","32","42"),tcl=-.1)
quant<-.5;col1a<-rgb(128/255,128/255,128/255,alpha=0.3);
for(b in 1:B){
  clust<-cluster(data=test,clustername="rank_pregid",size=length(unique(test$rank_pregid)),method="srswr")
  testB<-getdata(test,clust)
  test0<-subset(testB,testB$smoke==0)
  test1<-subset(testB,testB$smoke==1)
  rq0<-rqss(y~qss(gestweek,lambda=.75),tau=quant,data=test0)
  rq1<-rqss(y~qss(gestweek,lambda=.75),tau=quant,data=test1)
  x1<-unique(test1$gestweek)
  x1<-x1[order(x1)]
  length(x1)
  qdiff<-rq0$coef[2:length(x1)]-rq1$coef[2:length(x1)]
  lo<-loess(qdiff~x1[-1],span=spn)
  lines(predict(lo),ylim=ylims,lwd=1,col=col1a)
}
quant<-.25;col1a<-rgb(1,20/255,60/255,alpha=0.2);
for(b in 1:B){
  clust<-cluster(data=test,clustername="rank_pregid",size=length(unique(test$rank_pregid)),method="srswr")
  testB<-getdata(test,clust)
  test0<-subset(testB,testB$smoke==0)
  test1<-subset(testB,testB$smoke==1)
  rq0<-rqss(y~qss(gestweek,lambda=.75),tau=quant,data=test0)
  rq1<-rqss(y~qss(gestweek,lambda=.75),tau=quant,data=test1)
  x1<-unique(test1$gestweek)
  x1<-x1[order(x1)]
  length(x1)
  qdiff<-rq0$coef[2:length(x1)]-rq1$coef[2:length(x1)]
  lo<-loess(qdiff~x1[-1],span=spn)
  lines(predict(lo),ylim=ylims,lwd=1,col=col1a)
}
quant<-.75;col1a<-rgb(0,191/255,1,alpha=0.2);
for(b in 1:B){
  clust<-cluster(data=test,clustername="rank_pregid",size=length(unique(test$rank_pregid)),method="srswr")
  testB<-getdata(test,clust)
  test0<-subset(testB,testB$smoke==0)
  test1<-subset(testB,testB$smoke==1)
  rq0<-rqss(y~qss(gestweek,lambda=.75),tau=quant,data=test0)
  rq1<-rqss(y~qss(gestweek,lambda=.75),tau=quant,data=test1)
  x1<-unique(test1$gestweek)
  x1<-x1[order(x1)]
  length(x1)
  qdiff<-rq0$coef[2:length(x1)]-rq1$coef[2:length(x1)]
  lo<-loess(qdiff~x1[-1],span=spn)
  lines(predict(lo),ylim=ylims,lwd=1,col=col1a)
}

quant<-.5;col1b<-"black"
test0<-subset(test,test$smoke==0)
test1<-subset(test,test$smoke==1)
rq0<-rqss(y~qss(gestweek,lambda=.75),tau=quant,data=test0)
rq1<-rqss(y~qss(gestweek,lambda=.75),tau=quant,data=test1)
x1<-unique(test1$gestweek)
x1<-x1[order(x1)]
length(x1)
qdiff<-rq0$coef[2:length(x1)]-rq1$coef[2:length(x1)]
lo<-loess(qdiff~x1[-1],span=spn)
lines(predict(lo),ylim=ylims,lwd=3,col=col1b)
quant<-.25;col1b<-"red"
test0<-subset(test,test$smoke==0)
test1<-subset(test,test$smoke==1)
rq0<-rqss(y~qss(gestweek,lambda=.75),tau=quant,data=test0)
rq1<-rqss(y~qss(gestweek,lambda=.75),tau=quant,data=test1)
x1<-unique(test1$gestweek)
x1<-x1[order(x1)]
length(x1)
qdiff<-rq0$coef[2:length(x1)]-rq1$coef[2:length(x1)]
lo<-loess(qdiff~x1[-1],span=spn)
lines(predict(lo),ylim=ylims,lwd=3,col=col1b)
quant<-.75;col1b<-"blue"
test0<-subset(test,test$smoke==0)
test1<-subset(test,test$smoke==1)
rq0<-rqss(y~qss(gestweek,lambda=.75),tau=quant,data=test0)
rq1<-rqss(y~qss(gestweek,lambda=.75),tau=quant,data=test1)
x1<-unique(test1$gestweek)
x1<-x1[order(x1)]
length(x1)
qdiff<-rq0$coef[2:length(x1)]-rq1$coef[2:length(x1)]
lo<-loess(qdiff~x1[-1],span=spn)
lines(predict(lo),ylim=ylims,lwd=3,col=col1b)
legend("topleft", inset=.05, title="Quantile",lty=c(1,1,1),lwd=c(3,3,3),col=c("red","black","blue"),
       c("0.25","0.50","0.75"), horiz=F,bty="n")

screen(6)
test$y<-test$EFW_brt/1000
test0<-subset(test,test$smoke==0)
test1<-subset(test,test$smoke==1)
rq0<-rqss(y~qss(gestweek,lambda=.75),tau=quant,data=test0)
rq1<-rqss(y~qss(gestweek,lambda=.75),tau=quant,data=test1)
x1<-unique(test1$gestweek)
x1<-x1[order(x1)]
length(x1)
qdiff<-rq0$coef[2:length(x1)]-rq1$coef[2:length(x1)]
lo<-loess(qdiff~x1[-1],span=spn)
plot(predict(lo),ylim=ylims,type="l",las=1,tcl=-.1,lwd=3,col="white",
     ylab="Quantile Difference (kg)",xlab="Gestational Week",xaxt="n")
mtext("Predicted EFW (BART)", 3, line=1)
axis(1,at=seq(0,140,20),labels=c("20","24","28","32","36","30","32","42"),tcl=-.1)
quant<-.5;col1a<-rgb(128/255,128/255,128/255,alpha=0.3);
for(b in 1:B){
  clust<-cluster(data=test,clustername="rank_pregid",size=length(unique(test$rank_pregid)),method="srswr")
  testB<-getdata(test,clust)
  test0<-subset(testB,testB$smoke==0)
  test1<-subset(testB,testB$smoke==1)
  rq0<-rqss(y~qss(gestweek,lambda=.75),tau=quant,data=test0)
  rq1<-rqss(y~qss(gestweek,lambda=.75),tau=quant,data=test1)
  x1<-unique(test1$gestweek)
  x1<-x1[order(x1)]
  length(x1)
  qdiff<-rq0$coef[2:length(x1)]-rq1$coef[2:length(x1)]
  lo<-loess(qdiff~x1[-1],span=spn)
  lines(predict(lo),ylim=ylims,lwd=1,col=col1a)
}
quant<-.25;col1a<-rgb(1,20/255,60/255,alpha=0.2);
for(b in 1:B){
  clust<-cluster(data=test,clustername="rank_pregid",size=length(unique(test$rank_pregid)),method="srswr")
  testB<-getdata(test,clust)
  test0<-subset(testB,testB$smoke==0)
  test1<-subset(testB,testB$smoke==1)
  rq0<-rqss(y~qss(gestweek,lambda=.75),tau=quant,data=test0)
  rq1<-rqss(y~qss(gestweek,lambda=.75),tau=quant,data=test1)
  x1<-unique(test1$gestweek)
  x1<-x1[order(x1)]
  length(x1)
  qdiff<-rq0$coef[2:length(x1)]-rq1$coef[2:length(x1)]
  lo<-loess(qdiff~x1[-1],span=spn)
  lines(predict(lo),ylim=ylims,lwd=1,col=col1a)
}
quant<-.75;col1a<-rgb(0,191/255,1,alpha=0.2);
for(b in 1:B){
  clust<-cluster(data=test,clustername="rank_pregid",size=length(unique(test$rank_pregid)),method="srswr")
  testB<-getdata(test,clust)
  test0<-subset(testB,testB$smoke==0)
  test1<-subset(testB,testB$smoke==1)
  rq0<-rqss(y~qss(gestweek,lambda=.75),tau=quant,data=test0)
  rq1<-rqss(y~qss(gestweek,lambda=.75),tau=quant,data=test1)
  x1<-unique(test1$gestweek)
  x1<-x1[order(x1)]
  length(x1)
  qdiff<-rq0$coef[2:length(x1)]-rq1$coef[2:length(x1)]
  lo<-loess(qdiff~x1[-1],span=spn)
  lines(predict(lo),ylim=ylims,lwd=1,col=col1a)
}

quant<-.5;col1b<-"black"
test0<-subset(test,test$smoke==0)
test1<-subset(test,test$smoke==1)
rq0<-rqss(y~qss(gestweek,lambda=.75),tau=quant,data=test0)
rq1<-rqss(y~qss(gestweek,lambda=.75),tau=quant,data=test1)
x1<-unique(test1$gestweek)
x1<-x1[order(x1)]
length(x1)
qdiff<-rq0$coef[2:length(x1)]-rq1$coef[2:length(x1)]
lo<-loess(qdiff~x1[-1],span=spn)
lines(predict(lo),ylim=ylims,lwd=3,col=col1b)
quant<-.25;col1b<-"red"
test0<-subset(test,test$smoke==0)
test1<-subset(test,test$smoke==1)
rq0<-rqss(y~qss(gestweek,lambda=.75),tau=quant,data=test0)
rq1<-rqss(y~qss(gestweek,lambda=.75),tau=quant,data=test1)
x1<-unique(test1$gestweek)
x1<-x1[order(x1)]
length(x1)
qdiff<-rq0$coef[2:length(x1)]-rq1$coef[2:length(x1)]
lo<-loess(qdiff~x1[-1],span=spn)
lines(predict(lo),ylim=ylims,lwd=3,col=col1b)
quant<-.75;col1b<-"blue"
test0<-subset(test,test$smoke==0)
test1<-subset(test,test$smoke==1)
rq0<-rqss(y~qss(gestweek,lambda=.75),tau=quant,data=test0)
rq1<-rqss(y~qss(gestweek,lambda=.75),tau=quant,data=test1)
x1<-unique(test1$gestweek)
x1<-x1[order(x1)]
length(x1)
qdiff<-rq0$coef[2:length(x1)]-rq1$coef[2:length(x1)]
lo<-loess(qdiff~x1[-1],span=spn)
lines(predict(lo),ylim=ylims,lwd=3,col=col1b)
legend("topleft", inset=.05, title="Quantile",lty=c(1,1,1),lwd=c(3,3,3),col=c("red","black","blue"),
       c("0.25","0.50","0.75"), horiz=F,bty="n")

screen(7)
quant<-.5;
test$y<-test$EFW_rf/1000
test0<-subset(test,test$smoke==0)
test1<-subset(test,test$smoke==1)
rq0<-rqss(y~qss(gestweek,lambda=.75),tau=quant,data=test0)
rq1<-rqss(y~qss(gestweek,lambda=.75),tau=quant,data=test1)
x1<-unique(test1$gestweek)
x1<-x1[order(x1)]
length(x1)
qdiff<-rq0$coef[2:length(x1)]-rq1$coef[2:length(x1)]
lo<-loess(qdiff~x1[-1],span=spn)
plot(predict(lo),ylim=ylims,type="l",las=1,tcl=-.1,lwd=3,col="white",
     ylab="Quantile Difference (kg)",xlab="Gestational Week",xaxt="n")
mtext("Predicted EFW (RF)", 3, line=1)
axis(1,at=seq(0,140,20),labels=c("20","24","28","32","36","30","32","42"),tcl=-.1)
col1a<-rgb(128/255,128/255,128/255,alpha=0.3);
for(b in 1:B){
  clust<-cluster(data=test,clustername="rank_pregid",size=length(unique(test$rank_pregid)),method="srswr")
  testB<-getdata(test,clust)
  test0<-subset(testB,testB$smoke==0)
  test1<-subset(testB,testB$smoke==1)
  rq0<-rqss(y~qss(gestweek,lambda=.75),tau=quant,data=test0)
  rq1<-rqss(y~qss(gestweek,lambda=.75),tau=quant,data=test1)
  x1<-unique(test1$gestweek)
  x1<-x1[order(x1)]
  length(x1)
  qdiff<-rq0$coef[2:length(x1)]-rq1$coef[2:length(x1)]
  lo<-loess(qdiff~x1[-1],span=spn)
  lines(predict(lo),ylim=ylims,lwd=1,col=col1a)
}
quant<-.25;col1a<-rgb(1,20/255,60/255,alpha=0.2);
for(b in 1:B){
  clust<-cluster(data=test,clustername="rank_pregid",size=length(unique(test$rank_pregid)),method="srswr")
  testB<-getdata(test,clust)
  test0<-subset(testB,testB$smoke==0)
  test1<-subset(testB,testB$smoke==1)
  rq0<-rqss(y~qss(gestweek,lambda=.75),tau=quant,data=test0)
  rq1<-rqss(y~qss(gestweek,lambda=.75),tau=quant,data=test1)
  x1<-unique(test1$gestweek)
  x1<-x1[order(x1)]
  length(x1)
  qdiff<-rq0$coef[2:length(x1)]-rq1$coef[2:length(x1)]
  lo<-loess(qdiff~x1[-1],span=spn)
  lines(predict(lo),ylim=ylims,lwd=1,col=col1a)
}
quant<-.75;col1a<-rgb(0,191/255,1,alpha=0.2);
for(b in 1:B){
  clust<-cluster(data=test,clustername="rank_pregid",size=length(unique(test$rank_pregid)),method="srswr")
  testB<-getdata(test,clust)
  test0<-subset(testB,testB$smoke==0)
  test1<-subset(testB,testB$smoke==1)
  rq0<-rqss(y~qss(gestweek,lambda=.75),tau=quant,data=test0)
  rq1<-rqss(y~qss(gestweek,lambda=.75),tau=quant,data=test1)
  x1<-unique(test1$gestweek)
  x1<-x1[order(x1)]
  length(x1)
  qdiff<-rq0$coef[2:length(x1)]-rq1$coef[2:length(x1)]
  lo<-loess(qdiff~x1[-1],span=spn)
  lines(predict(lo),ylim=ylims,lwd=1,col=col1a)
}

quant<-.5;col1b<-"black"
test0<-subset(test,test$smoke==0)
test1<-subset(test,test$smoke==1)
rq0<-rqss(y~qss(gestweek,lambda=.75),tau=quant,data=test0)
rq1<-rqss(y~qss(gestweek,lambda=.75),tau=quant,data=test1)
x1<-unique(test1$gestweek)
x1<-x1[order(x1)]
length(x1)
qdiff<-rq0$coef[2:length(x1)]-rq1$coef[2:length(x1)]
lo<-loess(qdiff~x1[-1],span=spn)
lines(predict(lo),ylim=ylims,lwd=3,col=col1b)
quant<-.25;col1b<-"red"
test0<-subset(test,test$smoke==0)
test1<-subset(test,test$smoke==1)
rq0<-rqss(y~qss(gestweek,lambda=.75),tau=quant,data=test0)
rq1<-rqss(y~qss(gestweek,lambda=.75),tau=quant,data=test1)
x1<-unique(test1$gestweek)
x1<-x1[order(x1)]
length(x1)
qdiff<-rq0$coef[2:length(x1)]-rq1$coef[2:length(x1)]
lo<-loess(qdiff~x1[-1],span=spn)
lines(predict(lo),ylim=ylims,lwd=3,col=col1b)
quant<-.75;col1b<-"blue"
test0<-subset(test,test$smoke==0)
test1<-subset(test,test$smoke==1)
rq0<-rqss(y~qss(gestweek,lambda=.75),tau=quant,data=test0)
rq1<-rqss(y~qss(gestweek,lambda=.75),tau=quant,data=test1)
x1<-unique(test1$gestweek)
x1<-x1[order(x1)]
length(x1)
qdiff<-rq0$coef[2:length(x1)]-rq1$coef[2:length(x1)]
lo<-loess(qdiff~x1[-1],span=spn)
lines(predict(lo),ylim=ylims,lwd=3,col=col1b)
legend("topleft", inset=.05, title="Quantile",lty=c(1,1,1),lwd=c(3,3,3),col=c("red","black","blue"),
       c("0.25","0.50","0.75"), horiz=F,bty="n")

screen(8)
test$y<-test$EFW_gbm/1000
test0<-subset(test,test$smoke==0)
test1<-subset(test,test$smoke==1)
rq0<-rqss(y~qss(gestweek,lambda=.75),tau=quant,data=test0)
rq1<-rqss(y~qss(gestweek,lambda=.75),tau=quant,data=test1)
x1<-unique(test1$gestweek)
x1<-x1[order(x1)]
length(x1)
qdiff<-rq0$coef[2:length(x1)]-rq1$coef[2:length(x1)]
lo<-loess(qdiff~x1[-1],span=spn)
plot(predict(lo),ylim=ylims,type="l",las=1,tcl=-.1,lwd=3,col="white",
     ylab="Quantile Difference (kg)",xlab="Gestational Week",xaxt="n")
mtext("Predicted EFW (GBM)", 3, line=1)
axis(1,at=seq(0,140,20),labels=c("20","24","28","32","36","30","32","42"),tcl=-.1)
quant<-.5;col1a<-rgb(128/255,128/255,128/255,alpha=0.3);
for(b in 1:B){
  clust<-cluster(data=test,clustername="rank_pregid",size=length(unique(test$rank_pregid)),method="srswr")
  testB<-getdata(test,clust)
  test0<-subset(testB,testB$smoke==0)
  test1<-subset(testB,testB$smoke==1)
  rq0<-rqss(y~qss(gestweek,lambda=.75),tau=quant,data=test0)
  rq1<-rqss(y~qss(gestweek,lambda=.75),tau=quant,data=test1)
  x1<-unique(test1$gestweek)
  x1<-x1[order(x1)]
  length(x1)
  qdiff<-rq0$coef[2:length(x1)]-rq1$coef[2:length(x1)]
  lo<-loess(qdiff~x1[-1],span=spn)
  lines(predict(lo),ylim=ylims,lwd=1,col=col1a)
}
quant<-.25;col1a<-rgb(1,20/255,60/255,alpha=0.2);
for(b in 1:B){
  clust<-cluster(data=test,clustername="rank_pregid",size=length(unique(test$rank_pregid)),method="srswr")
  testB<-getdata(test,clust)
  test0<-subset(testB,testB$smoke==0)
  test1<-subset(testB,testB$smoke==1)
  rq0<-rqss(y~qss(gestweek,lambda=.75),tau=quant,data=test0)
  rq1<-rqss(y~qss(gestweek,lambda=.75),tau=quant,data=test1)
  x1<-unique(test1$gestweek)
  x1<-x1[order(x1)]
  length(x1)
  qdiff<-rq0$coef[2:length(x1)]-rq1$coef[2:length(x1)]
  lo<-loess(qdiff~x1[-1],span=spn)
  lines(predict(lo),ylim=ylims,lwd=1,col=col1a)
}
quant<-.75;col1a<-rgb(0,191/255,1,alpha=0.2);
for(b in 1:B){
  clust<-cluster(data=test,clustername="rank_pregid",size=length(unique(test$rank_pregid)),method="srswr")
  testB<-getdata(test,clust)
  test0<-subset(testB,testB$smoke==0)
  test1<-subset(testB,testB$smoke==1)
  rq0<-rqss(y~qss(gestweek,lambda=.75),tau=quant,data=test0)
  rq1<-rqss(y~qss(gestweek,lambda=.75),tau=quant,data=test1)
  x1<-unique(test1$gestweek)
  x1<-x1[order(x1)]
  length(x1)
  qdiff<-rq0$coef[2:length(x1)]-rq1$coef[2:length(x1)]
  lo<-loess(qdiff~x1[-1],span=spn)
  lines(predict(lo),ylim=ylims,lwd=1,col=col1a)
}

quant<-.5;col1b<-"black"
test0<-subset(test,test$smoke==0)
test1<-subset(test,test$smoke==1)
rq0<-rqss(y~qss(gestweek,lambda=.75),tau=quant,data=test0)
rq1<-rqss(y~qss(gestweek,lambda=.75),tau=quant,data=test1)
x1<-unique(test1$gestweek)
x1<-x1[order(x1)]
length(x1)
qdiff<-rq0$coef[2:length(x1)]-rq1$coef[2:length(x1)]
lo<-loess(qdiff~x1[-1],span=spn)
lines(predict(lo),ylim=ylims,lwd=3,col=col1b)
quant<-.25;col1b<-"red"
test0<-subset(test,test$smoke==0)
test1<-subset(test,test$smoke==1)
rq0<-rqss(y~qss(gestweek,lambda=.75),tau=quant,data=test0)
rq1<-rqss(y~qss(gestweek,lambda=.75),tau=quant,data=test1)
x1<-unique(test1$gestweek)
x1<-x1[order(x1)]
length(x1)
qdiff<-rq0$coef[2:length(x1)]-rq1$coef[2:length(x1)]
lo<-loess(qdiff~x1[-1],span=spn)
lines(predict(lo),ylim=ylims,lwd=3,col=col1b)
quant<-.75;col1b<-"blue"
test0<-subset(test,test$smoke==0)
test1<-subset(test,test$smoke==1)
rq0<-rqss(y~qss(gestweek,lambda=.75),tau=quant,data=test0)
rq1<-rqss(y~qss(gestweek,lambda=.75),tau=quant,data=test1)
x1<-unique(test1$gestweek)
x1<-x1[order(x1)]
length(x1)
qdiff<-rq0$coef[2:length(x1)]-rq1$coef[2:length(x1)]
lo<-loess(qdiff~x1[-1],span=spn)
lines(predict(lo),ylim=ylims,lwd=3,col=col1b)
legend("topleft", inset=.05, title="Quantile",lty=c(1,1,1),lwd=c(3,3,3),col=c("red","black","blue"),
       c("0.25","0.50","0.75"), horiz=F,bty="n")
dev.off()
close.screen(all = TRUE) 
