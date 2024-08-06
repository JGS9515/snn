path_datos_entrada<-"test.csv"#Poner aquí path datos entrada

spikes<-read.csv("spikes",header=FALSE)$V1

a<-read.csv(path_datos_entrada)$value

plot(a)
par(new=TRUE)

plot(spikes,col="red")


