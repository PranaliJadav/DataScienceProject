test<-read.csv('googleplaystore.csv')
attach(test)

library(leaps)


test2<-na.omit(test)
attach(test2)


full.data=cbind(Category, Rating, Reviews, Type, Price, Content.Rating)



data1<-leaps(full.data, Installs, method = "adjr2")
sort(data1$adjr2)
data2 = cbind(data1$which,data1$adjr2)

#Category, rating, Reviews, Price, Content.rating - 0.0151862151

#just cateogry, type, price, content.rating - 0.0132111202