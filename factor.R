library(fansi)
library(tidyverse)
library(lattice)
library(survival)
library(Formula)
library(Hmisc)
library(DMwR2)
library(MASS)

data0<-read.csv('月回报率.csv')
Time<-read.csv('Time.csv')
Stock<-read.table('原始股票池0.txt')
data1<-data.frame()
data2 = data0 %>% group_split(Stkcd)
N=dim(Stock)[1];T=dim(Time)[1];
for (i in 1:N){
  for (t in 1:T){
    if (Time[t,1] %in% data2[[i]]$Trdmnt)
      data1[t,i]<-data2[[i]]$Mretwd[which(data2[[i]]$Trdmnt == Time[t,1])]
    else
      data1[t,i]<-NA
  }
}
write.csv(data1,file = "data.csv")
r1 <- colSums(is.na(data1))/nrow(data1) >= 0.2
data3<-data1[,!r1]
Stock[r1,1]
#删除缺失值大于0.2的列
data3<-knnImputation(data3,meth = 'weighAvg',scale = T)
sum(is.na(data3))

###计算因子个数
data00=NULL; LL=1; nn=NULL

Data=t(data3)              ##  including all features

pZ<-length(Data[,1]);     N=pZ             #dimension
n<-length(Data[1,]);      T=n              #sample size
rmax<-10
v<-rep(0, rmax); bpn<-(pZ+n)/(pZ*n); cpn<-min(pZ,n)
kfactor<-1:rmax
y00=pZ/(n-1)

hm1=hm2=hm3<-matrix(rep(0,N*rmax),N,rmax)
for(i in 1:rmax){ for(j in i:N){ hm1[j,i]<-1 } }		
for(i in 1:rmax){ for(j in (i+1):N){ hm2[j,i]<-1} }
for(i in 1:rmax){ for(j in (i+2):N){ hm3[j,i]<-1 } }

denote0=rep(0,13)
X=Data-matrix(rep(t(apply(Data,1,mean)),n),ncol=n)

## Method 1: the method of zheng: denote0[,13]
sn<-cov(t(X));hatRR=cov2cor(sn);lambdaZ=eigen(hatRR)$values;           
VectorZ=eigen(hatRR)$vectors
DD=NULL; lambdaLY=lambdaZ; p=pZ
pp=rmax+2; mz=rep(0,pp); dmz=mz; tmz=mz

for (kk in 1:pp){
  qu=3/4
  lambdaZ1=lambdaZ[-seq(max(0, 1),kk,1)]; z0=qu*lambdaZ[kk]+(1-qu)*lambdaZ[kk+1]
  ttt=c(lambdaZ1-lambdaZ[kk], z0-lambdaZ[kk]) 
  y0=(length(lambdaZ1))/(n-1)
  mz[kk]=-(1-y0)/z0+y0*mean(na.omit(1/ttt))
}
temp2018=(-1/mz)[-1]-1-sqrt(pZ/(n-1));temp1=seq(1,rmax,1);
temp2=cbind(temp1, temp2018[1:rmax])
k00new=max((temp2[,1][temp2[,2]>0]), 0)+1
denote0[13]=k00new 

## Method 2: PC1, PC2, PC3, IC1, IC2, IC3  
v<-rep(0,rmax)
kfactor<-1:rmax
bNT<-(N+T)/(N*T)
cNT<-min(N,T)
bev<-eigen((X%*%t(X))/T)$values
for(k in 1:rmax){
  v[k]<-sum(bev[(k+1):N])
}

PC1<-v-v[rmax]*bNT*log(bNT)*kfactor
PC2<-v+v[rmax]*bNT*log(cNT)*kfactor
PC3<-v+v[rmax]*log(cNT)/cNT*kfactor

IC1<-log(v)-bNT*log(bNT)*kfactor
IC2<-log(v)+bNT*log(cNT)*kfactor
IC3<-log(v)+log(cNT)/cNT*kfactor

PC1f<-which.min(PC1); denote0[1]=PC1f
PC2f<-which.min(PC2); denote0[2]=PC2f
PC3f<-which.min(PC3); denote0[3]=PC3f

IC1f<-which.min(IC1); denote0[4]=IC1f
IC2f<-which.min(IC2); denote0[5]=IC2f
IC3f<-which.min(IC3); denote0[6]=IC3f

## Method 3: the method of onatski
oev<-eigen((X%*%t(X))/T)$values
ow<-2^(2/3)/(2^(2/3)-1)
#	ormax<-1.55*min(N^(2/5),T^(2/5))	
delte1<-max(N^(-1/2),T^(-1/2))
delte2<-0
delte3<-max(N^(-2/3),T^(-2/3))
ou<-ow*oev[rmax+1]+(1-ow)*oev[2*rmax+1]
ON1f<-sum(ifelse(oev>(1+delte1)*ou,1,0)); denote0[7]=ON1f
ON2f<-sum(ifelse(oev>(1+delte2)*ou,1,0)); denote0[8]=ON2f
ON3f<-sum(ifelse(oev>(1+delte3)*ou,1,0)); denote0[9]=ON3f

## Method 4: the method of onatski2
nols<-4
noev<-eigen((X%*%t(X))/T)$values
oJ<-rep(1,(nols+1))
ed<-noev[1:rmax]-noev[2:(rmax+1)]
noj<-rmax+1
for(j in 1:4){
  noy<-noev[noj:(noj+nols)]
  nox<-(noj+seq(-1,(nols-1),1))^(2/3)
  nobeta<-sum((nox-mean(nox))*(noy-mean(noy)))/sum((nox-mean(nox))^2)
  nodelte<-2*abs(nobeta)
  noer<-ifelse(max(ed-nodelte)<0,0,max(which(ed>=nodelte)))
  noj<-noer+1
}
NONf<-noer; denote0[10]=NONf

## Method 5: the method of horenstein
hev<-eigen((X%*%t(X))/(T*N))$values
er1<-hev[1:rmax]
er2<-hev[2:(rmax+1)]
gr1<-hev%*%hm1
gr2<-hev%*%hm2
gr3<-hev%*%hm3
er<-er1/er2
gr<-log(gr1/gr2)/log(gr2/gr3)
ERf<-which.max(er); denote0[11]=ERf
GRf<-which.max(gr); denote0[12]=GRf

##  number of selected factors by $PC_3$, $IC_3$, $ON_2$, $ER$ and $GR$ and $ACT$ 
LS=c(2,3,6,8,11,12,13) 
print(denote0[LS])
print(denote0)

###先暂时选定r=6
r=6;

###因子初步线性回归
FACTORS<-read.csv('五因子.csv')[,-1]
X<-data3;
V<-eigen(data.matrix(X)%*%t(X)/N*T)$vectors;
phi<-eigen(data.matrix(X)%*%t(X)/N*T)$values;
F<-sqrt(T)*V[,1:r];
Lamda<-t(X)%*%F/T;
E<-matrix(0,N,T)
for (i in 1:N){
  for (t in 1:T){
    E[i,t]<-X[t,i]-t(Lamda[i,])%*%F[t,]
  }
}
# (1) Observable factor regression on r=6 selected factors
b<-c()
for (i in 1:dim(FACTORS)[2]){
  yy=FACTORS[,i]
  lmS=lm(yy~F)
  b<-c(b,summary(lmS)$r.squared)
}
b

###因子的替代性检验
##不相关性检验
phi1<-c();
for (i in 1:dim(FACTORS)[2]){
  model<-lm(FACTORS[,i]~F);
  pi<-0;
  for (s in 1:T){
    for (t in 1:T){
      pi<-pi+F[t,1:r]%*%t(F[s,1:r])*residuals(model)[t]*residuals(model)[s]/T;
    }
  }
  Beta=t(coefficients(model)[-1]);
  phi1<-c(phi1,abs(T*Beta%*%ginv(pi)%*%t(Beta)));
}
phi1#第6,7,10,12,17个因子可以排除

##精确替代性检验
Gamma<-array(0, c(r, r, T))
for (t in 1:T){
  for (i in 1:N){
    Gamma[,,t]<- Gamma[,,t]+E[i,t]^2*(Lamda[i,]%*%t(Lamda[i,]))/N}
}
Ej<-c();Mj<-c();
for (i in 1:(dim(FACTORS[,-c(6,7,10,12,17)])[2])){
  #tao(j)
  model1<-lm(FACTORS[,i]~-1+F)
  gamma0<-coefficients(model1)
  VV<-diag(phi[1:r])
  tao<-c();
  for (t in 1:T){
    tao<-c(tao,abs(residuals(model1)[t])/(t(gamma0)%*%solve(VV)%*%Gamma[,,t]%*%solve(VV)%*%gamma0/N)^0.5)
  }
  #E(j),M(j)
  Ej<-c(Ej,sum(tao>1.96)/T)
  Mj<-c(Mj,max(tao))
}
Mj
Ej

##近似替代性检验
NS<-c();R<-c();
for (i in 1:(dim(FACTORS[,-c(6,7,10,12,17)])[2])){
  model2<-lm(FACTORS[,i]~-1+F)
  gamma<-coefficients(model2)
  #Var1
  Var1<-cov(residuals(model2),residuals(model2))
  #Var2
  Var2<-cov(F%*%gamma,F%*%gamma)
  #Var3
  Var3<-cov(FACTORS[,i],FACTORS[,i])
  NS<-c(NS,Var1/Var2)
  R<-c(R,Var2/Var3)
}
NS
R

###多因子模型的建立(四因子);
rf<-read.csv('月度无风险利率.csv')
C<-data.frame(matrix(data=NA,nrow=4,ncol=N))
for (i in 1:N){
  model3<-lm((X[,i]-rf[,4])~-1+A)
  C[,i]<-coefficients(model3)
}
summary(model3)

###选股
D<-data.frame(matrix(nrow=N,ncol=2));D[,1]<-Stock[!r1,1];
aver<-apply(FACTORS[,1:4], 2, mean);
for (i in 1:N){
  D[i,2]<-t(aver)%*%C[,i]
}
write.csv(D,file='最终股票池筛选结果.csv')
