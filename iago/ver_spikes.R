path_datos_entrada<-"nuevos datasets/callt2/preliminar/train_label_filled.csv"#Poner aquí path datos entrada

spikes<-read.csv("resultados/default/6/spikes.csv",header=FALSE)$V1

a<-read.csv(path_datos_entrada)$value

plot(a)
par(new=TRUE)

plot(spikes,col="red")


