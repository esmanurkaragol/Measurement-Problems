#RATİNG PRODUCTS
#average
#time based weighted average
# user based weighted average
# weighted rating
#Uygulama: Kullanıcı ve Zaman Ağırlıklı Kurs Puanı Hesaplama

import pandas as pd
import math
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.width", 500)
pd.set_option("display.expand_frame_repr", False)
pd.set_option("display.float_format", lambda x: "%5.f" %x)

#50+ saat, python A-Z: veri bilimi ve machine learning
# puan: 4.8(4.764925)
# toplam puan: 4611
# puan yüzdeleri: 75,20,4,1, <1
# yaklaşık sayısal karşılıkları: 3458, 922,184,46,6

df = pd.read_csv("C:/Users/esman/PycharmProjects/measurement_problems/datasets/course_reviews.csv")
df.head()
df.shape
#puanların dağılımı
df["Rating"].value_counts()

#soru soran kaç kişi var .örneğin  1 soru soran 276 kişi var...
df["Questions Asked"].value_counts()

# sorulan soru bazında verilen puan. örneğin 1 soru soran 276 kişi ort. kaç puan verdi?
df.groupby("Questions Asked").agg({"Questions Asked": "count",
                                   "Rating" : "mean"})
df.head()

#AVERAGE

#ortalama puan. sadece böyle bir ortalama hesabını göz önünde bulundurursan son zamanlardaki memnuniyet trendini kaçırmış olabilirsin.
#ürünlerle ilgili +,- trendleri geri dönüşleri göz önünde bulundurmalısın:
#ne yaparak bu güncel trendleri yakalayabiliriz? bunun için time based weighted average yani puan zamanlarına göre ağırlıklı ortalama almalıyız.
#böylece zamana göre bir ortalama hesaplamış oluruz
#Örneğin bir ürünün ilk 3 ayında yüksek puan alması son zamanlardaki yorumlarla aynı ağırlıkta değerlendirildiği için ben ürünün güncel tutumunu öğrenemeyeceğim.
df["Rating"].mean()

#time based weighted average
df.head()
df.info()
#TimeStamp: yorumların hangi gün ve saatde yazılıdğı bilgisini veren değişken.
#object olan değişkenleri zamana çevirmelisin
df["Timestamp"] = pd.to_datetime(df["Timestamp"])
df["Enrolled"] = pd.to_datetime(df["Enrolled"])
df.info()
# yapılan tüm yorumları gün cinsinden ifade etmemiz gerekiyor.
# örnek 3 gün önce bu yorum yapıldığının bilgisini görmeliyim. bunun için "timestamp" değişkenim var. bugünün tarihini giriyorum ve ona göre kaç gün önce ne zaman yapıldığını söylemesini istiyorum.
#bugünün tarihi için current_date oluşturulur. string değer veririz buna tarihe çevir dedik datatime ile.
current_date = pd.to_datetime("2021-02-10 0:0:0")

#bugunun tarihinden yorumların yapıldığı tariihi çıkar. bunu gün cinsinden söyle.
df["days"] = (current_date - df["Timestamp"]).dt.days
df.head()
#Son 30 günde yapılan yorumlar
df[df["days"] <= 30].count()
#30 günde yapılan yorumların ortalamasını sitiyorum. burada son 30 gün içeriisndeki yorumların sadece rating olanlarını seç diyorum. ve bunların ort. al
df.loc[df["days"] <=30, "Rating"].mean()

#30dan büyük, 90dan küçük eşit olanları seç ve ort. al buda benim 2.aralığım.
df.loc[(df["days"] > 30) & (df["days"] <=90), "Rating" ].mean()

#3.aralık da oluşturup yine bakıyorum.
df.loc[(df["days"] > 90) & (df["days"] <=180), "Rating" ].mean()
#3 aralık oluşturduğumda şunu gördüm ;
# 90 -180 gün önce ort.paun:4.75 iken //son 30-90 gün önce 4.76 //son 30 gün önce ise 4.77
#giderek puanımız artmış demek ki kurs son dönemlerde beğeniliyor.

#4.aralığıda oluştur.
df.loc[df["days"] >180, "Rating"].mean()

#şimdi bölmüş olduğun zaman aralıklarına bir ağırlık vereceksin.
df.loc[df["days"] <=30, "Rating"].mean() *28/100 + \
df.loc[(df["days"] > 30) & (df["days"] <=90), "Rating" ].mean() *26/100 + \
df.loc[(df["days"] > 90) & (df["days"] <=180), "Rating" ].mean() * 24/100 +\
df.loc[(df["days"] > 90) & (df["days"] <=180), "Rating" ].mean() * 22/100

#güncel olan yorumlara daha fazla ağırlık vermek istediğim için ona en fazla ağırlık verdim 28/100 dedim .

#her seferinde böyle ağırlık hesaplayıp yazmak gereksiz. daha sürdürülebilir bir şey yazalım.
def time_based_weighted_average(dataframe, w1=28, w2=26, w3=24, w4=22):
    return dataframe.loc[df["days"] < 30, "Rating"].mean() *w1 / 100 + \
        dataframe.loc[(dataframe["days"] > 30) & (dataframe["days"] <=90), "Rating"].mean() *w2 /100 + \
        dataframe.loc[(dataframe["days"] > 90) & (dataframe["days"] <= 180), "Rating"].mean() * w3 / 100 + \
        dataframe.loc[(dataframe["days"] > 180), "Rating"].mean() * w4 / 100
time_based_weighted_average(df)
#1.aralığa 30, 2.aralığa 26, 3.aralığa 22, 4.aralığa da 22 ağırlığını vermek istiyorsam. o zaman da diğer ağırlıkları yaz ve fonksiyona gönder.
#virgülden sonra yazılan her rakam önemli, her şeyiyle dikkat etmelisin.
time_based_weighted_average(df, 30, 26, 22, 22)

##KULLANICI TEMELLİ AĞIRLIKLI ORTALAMA (user based weighted average)
#Acaba tüm kullanıcıların verdiği puanlar aynı ağırlığa mı sahip olmalı?
#yani kursun tammaını izleyip puan verenle kursun %5 ini izleyen kullanıcının ağırlıkları aynı mı olmalı?
#kursun izlenme oranlarına göre farklı bir ağırlık vermelisin.

df.head()
#kursta ilerleme durumlarına (Progress) göre verilen puan arasında bir ilişki var.
df.groupby("Progress").agg({"Rating": "mean"})
# o zaman kursun izlenmesiyle oranlı olacak şekilde ağırlık hesaplansın.
#böylece ortalama hesaplama işi hassaslaşmış oluruz.
df.loc[df["Progress"] <= 10, "Rating"].mean() * 22 / 100 + \
    df.loc[(df["Progress"] > 10) & (df["Progress"] <= 45), "Rating"].mean() * 24 / 100 + \
    df.loc[(df["Progress"] > 45) & (df["Progress"] <= 75), "Rating"].mean() * 26 / 100 + \
    df.loc[(df["Progress"] > 75), "Rating"].mean() * 28 / 100

def user_based_weighted_average(dataframe, w1=22, w2=24, w3=26, w4=28):
    return dataframe.loc[dataframe["Progress"] <= 10, "Rating"].mean() *w1 / 100 + \
           dataframe.loc[(dataframe["Progress"] > 10) & (dataframe["Progress"] <= 45), "Rating"].mean() *w2 /100 + \
           dataframe.loc[(dataframe["Progress"] > 45) & (dataframe["Progress"] <= 75), "Rating"].mean() * w3 / 100 + \
           dataframe.loc[(dataframe["Progress"] > 75), "Rating"].mean() * w4 / 100
user_based_weighted_average(df,20,24,26,30)

#Ağırlıklı Derecelendirme (Weighted Rating)
#time_w:timedan gelecek olan puanın ağırlığı
#user_w: userdan gelecek olan puanın ağırlığı
def course_weighted_rating (dataframe, time_w=50, user_w=50):
    return time_based_weighted_average(dataframe) * time_w/100 + user_based_weighted_average(dataframe)*user_w/100
course_weighted_rating(df)
#user bneim için daha önemli daha yüksek verdim ağırlığı
course_weighted_rating(df, time_w=40, user_w=60)


