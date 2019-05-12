
#Reading the data from the external source
Data=read.csv("F://MastersMaterial//DataMining//Project//Data_for_UCI_named.csv",header =TRUE)

set.seed(123)
attach(Data)

library(e1071)
library(ISLR)
library(class)


fix(Data)

#Sampling the data into Train and Test Set
SampleData=sample(1:nrow(Data),6000)
TrainSet=Data[SampleData,]
TestSet=Data[-SampleData,]

fix(TrainSet)
fix(TestSet)
dim(TrainSet)
dim(TestSet)




#Implementation by using Navies Bayes Algorithm
NB=naiveBayes(stabf~.,Data,TrainSet)
NB_Train.predict=predict(NB,TrainSet)
table(NB_Train.predict,TrainSet$stabf)

NavTrainerr=mean(NB_Train.predict!=TrainSet$stabf)*100
NavTrainacc=100-NavTrainerr

NB_Test.predict=predict(NB,TestSet)
table(NB_Test.predict,TestSet$stabf)
NaviTeserr=mean(NB_Test.predict!=TestSet$stabf)*100
NaviTestAcc=100-NaviTeserr

barplot(c(NavTrainacc,NavTrainerr,NaviTestAcc,NaviTeserr),ylim = c(0,100),main = "Navies Bayes Results ",
        ylab ="Percentages",col=c("lightGreen","red"),names.arg =c("Training Accuracy","Training Error","Test Accuracy","Test Error Rate"),legend=c("Accuracy","Error Rate"),args.legend = list(title="Percentages",x="topright",cex=0.7))



#Implementation by SVM Algorithm

#TrainSet[stabf]=as.factor(TrainSet[stabf])


#Linear kernel
Svm.model=svm(stabf~.,data = TrainSet,kernel="linear",cost=10,scale = FALSE)
summary(Svm.model)

svm.tune=tune(svm,stabf~.,data = TrainSet,kernel="linear",ranges=list(cost=c(0.1 ,1 ,10 ,100 ,1000)))
summary(svm.tune)
bestmod=svm.tune$best.model
summary(bestmod)


svm.train.predict=predict(bestmod,TrainSet)
table(svm.train.predict,TrainSet$stabf)
svmtracc=mean(svm.train.predict==TrainSet$stabf)*100
svmtrerr=100-svmtracc

svm.test.predict=predict(bestmod,TestSet)
table(svm.test.predict,TestSet$stabf)
svmteerr=mean(svm.test.predict!=TestSet$stabf)
svmteacc=100-svmteerr

barplot(c(svmtracc,svmtrerr,svmteacc,svmteerr),ylim = c(0,100),main = "SVM -Linear Kernel Output ",
        ylab ="Percentages",col=c("lightGreen","red"),names.arg =c("Training Accuracy","Training Error","Test Accuracy","Test Error Rate"),legend=c("Accuracy","Error Rate"),args.legend = list(title="Percentages",x="topright",cex=0.8))


  
#Radial Kernel


svm.model.kern= svm(stabf~.,data = TrainSet,kernel="radial",gamma=1,cost=1,scale = FALSE)
summary(svm.model.kern)

svm.tune.rad=tune(svm,stabf~.,data=TrainSet,, kernel="radial",gamma=c(0.5,1,2,3,4),ranges=list(cost=c(0.1 ,1 ,10 ,100 ,1000)))
summary(svm.tune.rad)
bestmodrad=svm.tune.rad$best.model

summary(svm.tune.rad$best.model)

svm.train.pred.rad=predict(bestmodrad,TrainSet)
table(svm.train.pred.rad,TrainSet$stabf)
svm.train.pred.acc=mean(svm.train.pred.rad==TrainSet$stabf)*100
svm.train.pred.err=100-svm.train.pred.acc


svm.test.pred.rad=predict(bestmodrad,TestSet)
table(svm.test.pred.rad,TestSet$stabf)
svm.test.pred.acc=mean(svm.test.pred.rad==TestSet$stabf)*100
svm.test.pred.err=100-svm.test.pred.acc

barplot(c(svm.train.pred.acc,svm.train.pred.err,svm.test.pred.acc,svm.test.pred.err),ylim = c(0,100),main = "SVM -Radial Kernel Output ",
        ylab ="Percentages",col=c("lightGreen","red"),names.arg =c("Training Accuracy","Training Error","Test Accuracy","Test Error Rate"),legend=c("Accuracy","Error Rate"),args.legend = list(title="Percentages",x="topright",cex=0.7))

#Polynominal kernel

svm.model.poly=svm(stabf~.,data = TrainSet,kernel="polynomial",gamma=1,cost=1,scale = FALSE)
summary(svm.model.poly)

svm.tune.poly=tune(svm,stabf~.,data=TrainSet,kernel="polynomial",gamma=c(0.5,1,2,3,4),ranges=list(cost=c(0.1 ,1 ,10 ,100 ,1000)))
summary(svm.tune.poly)
bestmod.poly=svm.tune.poly$best.model
summary(bestmod.poly)

svm.train.pred.poly=predict(bestmod.poly,TrainSet)
table(svm.train.pred.poly,TrainSet$stabf)
svm.train.pred.poly.acc=mean(svm.train.pred.poly==TrainSet$stabf)*100
svm.train.pred.poly.err=100-svm.train.pred.poly.acc


svm.test.pred.poly=predict(bestmod.poly,TestSet)
table(svm.test.pred.poly,TestSet$stabf)
svm.test.pred.poly.acc=mean(svm.test.pred.poly==TestSet$stabf)*100
svm.test.pred.poly.err=100-svm.test.pred.poly.acc

barplot(c(svm.train.pred.poly.acc,svm.train.pred.poly.err,svm.test.pred.poly.acc,svm.test.pred.poly.err),ylim = c(0,100),main = "SVM -Polynominal Kernel Output ",
        ylab ="Percentages",col=c("lightGreen","red"),names.arg =c("Training Accuracy","Training Error","Test Accuracy","Test Error Rate"),legend=c("Accuracy","Error Rate"),args.legend = list(title="Percentages",x="topright",cex=0.7))

#Decision Tree Holdout method  Implementation

library(tree)

sta.test=stabf[-SampleData]
sta.train=stabf[SampleData]

tree.dt=tree(stabf~.,Data,subset =SampleData)
summary(tree.dt)
plot(tree.dt)
text(tree.dt,pretty = 0)

tree.pred.train=predict(tree.dt,TrainSet,type = "class")
table(tree.pred.train,sta.train)
tree.pred.train.acc=mean(tree.pred.train==sta.train)*100
tree.pred.train.err=100-tree.pred.train.acc

tree.pred=predict(tree.dt,TestSet,type = "class")
table(tree.pred,sta.test)
tree.pred.test.acc=mean(tree.pred==sta.test)*100
tree.pred.test.err=100-tree.pred.test.acc

barplot(c(tree.pred.train.acc,tree.pred.test.err,tree.pred.test.acc,tree.pred.test.err),ylim = c(0,100),main = "Decision Tree -Holdout method ",
        ylab ="Percentages",col=c("lightGreen","red"),names.arg =c("Training Accuracy","Training Error","Test Accuracy","Test Error Rate"),legend=c("Accuracy","Error Rate"),args.legend = list(title="Percentages",x="topright",cex=0.8))

#Bagging  Algorithm implementation

library(randomForest)

tree.rand.bag=randomForest(stabf~.,Data,subset =SampleData,ntree=500,mtry=13)


tree.rand.train.pred=predict(tree.rand.bag,TrainSet,type="class")
table(tree.rand.train.pred,sta.train)
tree.rand.bag.acc=mean(tree.rand.train.pred==sta.train)*100
tree.rand.bag.err=100-tree.rand.bag.acc

tree.rand.test.pred.bag=predict(tree.rand.bag,TestSet,type="class")
table(tree.rand.test.pred.bag,sta.test)
tree.rand.test.bag.acc=mean(tree.rand.test.pred.bag==sta.test)*100
tree.rand.test.bag.err=100-tree.rand.test.bag.acc

barplot(c(tree.rand.bag.acc,tree.rand.bag.err,tree.rand.test.bag.acc,tree.rand.test.bag.err),ylim = c(0,100),main = "Bagging Method ",
        ylab ="Percentages",col=c("lightGreen","red"),names.arg =c("Training Accuracy","Training Error","Test Accuracy","Test Error Rate"),legend=c("Accuracy","Error Rate"),args.legend = list(title="Percentages",x="topright",cex=0.7))


#Random Forest 


tree.rand=randomForest(stabf~.,Data,subset =SampleData,ntree=100,mtry=2)
tree.rand.pred=predict(tree.rand,TestSet,type="class")
table(tree.rand.pred,sta.test)
tree.mtr2acc=mean(tree.rand.pred==sta.test)*100
tree.mtr2err=100-tree.mtr2acc

tree.randm3=randomForest(stabf~.,Data,subset =SampleData,ntree=100,mtry=3)

tree.randf.train.pred=predict(tree.randm3,TrainSet,type = "class")
table(tree.randf.train.pred,sta.train)
tree.randf.train.acc=mean(tree.randf.train.pred==sta.train)*100
tree.randf.train.err=100-tree.randf.train.acc

tree.rand.pred=predict(tree.randm3,TestSet,type="class")
table(tree.rand.pred,sta.test)
tree.randf.test.acc=mean(tree.rand.pred==sta.test)*100
tree.randf.test.err=100-tree.randf.test.acc

barplot(c(tree.randf.train.acc,tree.randf.train.err,tree.randf.test.acc,tree.randf.test.err),ylim = c(0,100),main = "Random Forest Method with ntry=3 ",
        ylab ="Percentages",col=c("lightGreen","red"),names.arg =c("Training Accuracy","Training Error","Test Accuracy","Test Error Rate"),legend=c("Accuracy","Error Rate"),args.legend = list(title="Percentages",x="topright",cex=0.8))




tree.rand=randomForest(stabf~.,Data,subset =SampleData,ntree=100,mtry=4)
tree.rand.pred=predict(tree.rand,TestSet,type="class")
table(tree.rand.pred,sta.test)

mean(tree.rand.pred==sta.test)


