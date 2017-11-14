require(forecast)
require(randomForest)
require(neuralnet)
require(caret)
require(RSNNS)
require(glmnet)
# > dim(smoothDat$Quality)
# [1] 811  13
last.1y = 758:811
last.3y = 657:811
last.5y = 561:811
all = 1:811
factors = c("Value","Growth","Momentum","Quality","Size","Vol","Yield","spx")

df = smoothDat$Vol
ind = last.3y
ret = df$PX_LAST
xreg = df[,-1]

train = 561:757
test = last.1y
df.train = df[train,]
df.test = df[test,]
ret.train = df$PX_LAST[train]
ret.test = df$PX_LAST[test]
xreg.train = df[train,-1]
xreg.test = df[test,-1]

### Time Series Regression
m.train.ar = auto.arima(ret.train,xreg=xreg.train,stationary = FALSE,ic="bic")
f.reg = forecast(m.train.ar,xreg = xreg.test)


### Penalized Regression (Tuning Parameter Chosen by CV)
getWeightedLasso = function(df,ind,f) {
  res = list()
  x = model.matrix(f,data=df)[ind,-1]
  y=df$PX_LAST[ind]
  hl = 52
  delta = 0.5^(1/hl)
  l = length(ind)
  weight = rep(NA,l)
  for (i in 1:l) {
    weight[i] = delta^(l-i)
  }
  grid = 2.716^seq(-5,5,length=10)
  #cv.out = cv.glmnet(x,y,alpha=1,weights=weight,lambda=grid)
  cv.out = cv.glmnet(x,y,alpha=1,lambda=grid)
  best.lam = cv.out$lambda.min
  #lasso = glmnet(x,y,alpha=1,lambda = grid,weights=weight)
  lasso = glmnet(x,y,alpha=1,lambda = grid)
  #plot(lasso,xvar="lambda")
  coef_ = predict(lasso,s=best.lam,type="coefficients")[1:9,]
  res = list(
    model = lasso,
    coef = coef_,
    bestLam = best.lam
  )
  res
}
m.train.lasso = getWeightedLasso(df,train,f=as.formula("PX_LAST~."))
f.lasso = predict(m.train.lasso$model,newx=as.matrix(xreg.test),s=m.train.lasso$bestLam,type="link")



########### Random Forest

m.train.rf = randomForest(ret.train~.,data=xreg.train)
plot(m.train.rf)
f.rf = predict(m.train.rf,newdata=xreg.test)



########## Regularized Neural Nets
f = as.formula(paste("PX_LAST~",paste(colnames(df)[-1],collapse = "+"),sep=""))

# nn1 = neuralnet(f,data=df[ind,],linear.output = TRUE,hidden=c(20,20,20)) #fitting process is kind of slow
# plot(nn1)



tuneGrid_ = expand.grid( size=6:15,decay = seq(0.05,0.15,length=10))
trControl_ = trainControl(method="repeatedcv",number=5,repeats=3,search="grid",verboseIter = FALSE,returnResamp = "all")
nn2 = caret::train(x=df[train,-1],y=df$PX_LAST[train],method='nnet',trControl=trControl_,tuneGrid=tuneGrid_,trace=FALSE,
                   metric="RMSE",linout=TRUE) # fitting is very slow due to repeated grid search
summary(nn2)
plot(nn2)
f.rnn = predict(nn2,newdata = xreg.test,type="raw")

######### Plotting
deparse(substitute(df))
plot(f.reg,main="Low Vol Factor Return Prediction",axes=FALSE)
lines(202:255,ret.test,col="red",lwd=3)
lines(202:255,f.lasso,col="orange",lwd=3)
lines(202:255,f.rf,col="yellow",lwd=3)
lines(202:255,f.rnn,col="green",lwd=3)
legend(x=0,y=1,"Time Series Regression",col="blue",pch=15)
legend(x=0,y=-0.35,"Lasso Regression",col="orange",pch=15)
legend(x=0,y=1-0.35,"Random Forest",col="yellow",pch=15)
legend(x=0,y=1-0.7,"Regularized Neural Net",col="green",pch=15)
legend(x=0,y=0,"Actual Value",col="red",pch=15)
axis(side=1,at=seq(1,255,by=4),labels=rownames(df)[seq(nrow(df)-254,nrow(df),by=4)])
axis(side=2,labels=TRUE)
################# Putting them together
f = as.formula(paste("PX_LAST~",paste(colnames(df)[-1],collapse = "+"),sep=""))



getModels = function(df,ind,f) {
  ret = df$PX_LAST
  xreg_ = df[,-1]
  tuneGrid_ = expand.grid( size=6:15,decay = seq(0.05,0.15,length=10))
  trControl_ = trainControl(method="repeatedcv",number=5,repeats=3,search="grid",verboseIter = FALSE,returnResamp = "all")
  res = list(
    armax = auto.arima(ret[ind],xreg=xreg_[ind,],stationary = FALSE,ic="bic"),
    lasso = getWeightedLasso(df,ind,f),
    rf = randomForest(f,data=df[ind,]),
    rnn = caret::train(x=df[ind,-1],y=df$PX_LAST[ind],method='nnet',trControl=trControl_,tuneGrid=tuneGrid_,trace=FALSE,
                       metric="RMSE",linout=TRUE)
  )
  res
}
pm1 = list(
  Quality = getModels(smoothDat$Quality,last.5y,f),
  Value = getModels(smoothDat$Value,last.5y,f),
  Growth = getModels(smoothDat$Growth,last.5y,f),
  Momentum = getModels(smoothDat$Momentum,last.5y,f),
  Size = getModels(smoothDat$Size,last.5y,f),
  Vol = getModels(smoothDat$Vol,last.5y,f),
  Yield = getModels(smoothDat$Yield,last.5y,f)
)

getPreds = function(fact="Quality") {
  exp1 = parse(text=paste("pm1$",fact,sep=""))
  mod = eval(exp1)
  exp2 = parse(text=paste("smoothDat2$",fact,sep=""))
  xreg = eval(exp2)
  xreg_ = xreg[nrow(xreg),-1]
  res = list(
    f.armax = forecast(mod$armax,xreg=xreg_),
    f.lasso = predict(mod$lasso$model,newx=as.matrix(xreg_),s=mod$lasso$bestLam,type="link"),
    f.rf = predict(mod$rf,newdata = xreg_),
    f.rnn = predict(mod$rnn,newdata = xreg_,type="raw")
  )
  res
}
scaledPreds = list(
  Quality = getPreds("Quality"),
  Value = getPreds("Value"),
  Growth = getPreds("Growth"),
  Momentum = getPreds("Momentum"),
  Size = getPreds("Size"),
  Vol = getPreds("Vol"),
  Yield = getPreds("Yield")
)


res = rep(0,times=4)
for (lst in scaledPreds) {
  tmp = c(lst$f.armax$mean[1],lst$f.lasso[1],lst$f.rf[1],lst$f.rnn[1])
  names(res) = rep("2017-10-13",times=4)
  res = rbind(res,tmp)
}
rownames(res) = c("0",names(scaledPreds))
colnames(res) = c("OLS","LASSO","Random Forest","Neural Nets")
res = res[-1,]
round(res,1)


################################ Recommendations

getMeanSd = function(fact) {
  exp = parse(text=paste("smoothDat3$",fact,"$PX_LAST",sep=""))
  mean = mean(eval(exp))
  sd = sd(eval(exp))
  list(
    mean = mean,
    sd = sd
  )
}
resScaled = res
resScaled["Quality",] = getMeanSd("Quality")$mean + getMeanSd("Quality")$sd * resScaled["Quality",]
resScaled["Value",] = getMeanSd("Value")$mean + getMeanSd("Value")$sd * resScaled["Value",]
resScaled["Growth",] = getMeanSd("Growth")$mean + getMeanSd("Growth")$sd * resScaled["Growth",]
resScaled["Momentum",] = getMeanSd("Momentum")$mean + getMeanSd("Momentum")$sd * resScaled["Momentum",]
resScaled["Size",] = getMeanSd("Size")$mean + getMeanSd("Size")$sd * resScaled["Size",]
resScaled["Vol",] = getMeanSd("Vol")$mean + getMeanSd("Vol")$sd * resScaled["Vol",]
resScaled["Yield",] = getMeanSd("Yield")$mean + getMeanSd("Yield")$sd * resScaled["Yield",]

round(resScaled,3)

vote = function(vec) {
  mean(vec[2:4])
}
rec = apply(res,1,vote)


write.csv(resScaled,"tmp.csv")
write.csv(rec,"tmp1.csv")
write.csv(res,"tmp2.csv")
