#Derecelendirmeye göre Sıralama (SORTİNG BY RATİNG)
#veride yer alan değişkenler:
#course_name: kursun adı
#instructor_name: eğitmenin adı
#purchase_count: satın alma sayıları
# rating:kursun ortalama puanı
#comment_count: kursun aldığı yorum sayısı
#5_point: 5 puan veren kişi sayısı (aynısı 4,3,2,1 puan içinde geçerli değişkenler vardır.)

#Kurs Sıralaması
import pandas as pd
import math
import scipy.stats as st
#standartlaşma için ihtiyacımız var.
from sklearn.preprocessing import MinMaxScaler
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.expand_frame_repr", False)
pd.set_option("display.float_format", lambda x: "%5.f" %x)

df = pd.read_csv("C:/Users/esman/PycharmProjects/measurement_problems/datasets/product_sorting.csv")
print(df.shape)
df.head(10)
#Sorting by Rating
#puana göre kursları sıralayabiliriz
df.sort_values("rating", ascending=False).head(20)
#yukarıdaki kod satırında puana göre kursları sıraladık ancak satın alma ve yorum sayısı gözden kaçtığı görülmektedir.
#bu nedenle bunlarıda dikkate alarak bir sıralama yapmalıyız. satın alma sayııs+yorum sayısı+ortalama hepsiini göz önünde bulundurarak sıralamalıyız.
#SORTİNG BY COMMENT COUNT OR PURCHASE COUNT
df.sort_values("purchase_count", ascending=False).head(20)

#yorum sayısına göre sırala
df.sort_values("comment_count", ascending = False).head(20)

#derecelendirme, satın alma, yoruma göre sıralama
#sorting by rating, comment and purchase
#şimdi purchase coount ve comment count büyük sayılar. rating ise 1-5 arasında kalıyor.
#rating max 5 olacağı için her türlü ezilecek çok fazla değerinı görmeyeceğiz.
#bu nedenle ben purchasec. ve comment c. 1-5 arasında olacak şekilde dönüşüm yapayım.

#purchase_count_scaled adında yeni bir değişken oluştuurup dönüşüm değerlerini burada tutacağım.
df["purchase_count_scaled"] = MinMaxScaler(feature_range=(1,5)). \
    fit(df[["purchase_count"]]). \
    transform(df[["purchase_count"]])

df.sort_values("purchase_count", ascending=False).head(20)
df.sort_values("commment_count", ascending=False).head(20)

df.describe().T

#"comment_count_scaled" YENİ değişken oluşturup buraya yeni dönüşüm değerlerini atıyorum.
df["comment_count_scaled"] = MinMaxScaler(feature_range=(1, 5)). \
    fit(df[["commment_count"]]). \
    transform(df[["commment_count"]])
#şimdi artık rating, purchase case ve comment count aynı aralıklarda (1-5 arasında standartlaştırdım) artık hepsini bir işleme sokabilirim.
(df["comment_count_scaled"] * 32 / 100 +
 df["purchase_count_scaled"] * 26 / 100 +
 df["rating"] * 42 / 100)


def weighted_sorting_score(dataframe, w1=32, w2=26, w3=42):
    return (dataframe["comment_count_scaled"] * w1 / 100 +
            dataframe["purchase_count_scaled"] * w2 / 100 +
            dataframe["rating"] * w3 / 100)

df["weighted_sorting_score"] = weighted_sorting_score(df)

df.sort_values("weighted_sorting_score", ascending=False).head(20)

#belirli bir key lere göre kurs isimlerini sırala
df[df["course_name"].str.contains("Veri Bilimi")].sort_values("weighted_sorting_score", ascending=False).head(20)


####################
# Bayesian Average Rating Score- istatiksel yöntem

# Sorting Products with 5 Star Rated(5 yıldızlı sistemlerde ürün sıralama)
# Sorting Products According to Distribution of 5 Star Rating (5 yıldızın dağılımına göre ürün sıralama)
#bayesian_average_rating: puan dağılımları üzerinden ağırlıklı bir şekilde olasılıksal ortalama hesaplar.
#bayesian average rating=  rating ilgili ortalama değer verecektir.
#burada hesaplanan rating, bir ürünün nihai ortalama puan veya skor olarak değerlendirilebilir.
#vahit hoca skor olarak değerlendiriyor çünkü var olan ratingden biraz aşağıda sonuç veriyor.
#elimizde geçmişte elde etmiş olduğumuz puanların dağılımları var. yani geçmiş bir bilgi, önsel bir bilgi var elimizde. bu önsel hesaplama işine odaklanarak gelecekle ilgili bir şeyler yapma yaklaşımıyla tekrar bir rating hesabı yapılır. bu olasılıksal bir rating hesabıdır.
#n : girilecek yıldızlara ait gözlemleme frekanslarını ifade etmektedir.
#confidence: hesaplanacak olan z tablo değerine ilişkin bir değer elde edebilmektir.
def bayesian_average_rating(n, confidence=0.95):
    if sum(n) == 0:
        return 0
    K = len(n)
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    N = sum(n)
    first_part = 0.0
    second_part = 0.0
    for k, n_k in enumerate(n):
        first_part += (k + 1) * (n[k] + 1) / (N + K)
        second_part += (k + 1) * (k + 1) * (n[k] + 1) / (N + K)
    score = first_part - z * math.sqrt((second_part - first_part * first_part) / (N + K + 1))
    return score


df.head()
#tek odağımız ürünlerin puanları ise bu kullanılabilir. ancak birden fazla özelliği göz önünde bulundurmaz.
#apply ile df üzerinde gez. ne yapması gerektiğini de fonksiyon ile tanımla. bu fonkdiyonuda lambda ile tanımlayabilirsin.
df["bar_score"] = df.apply(lambda x: bayesian_average_rating(x[["1_point",
                                                                "2_point",
                                                                "3_point",
                                                                "4_point",
                                                                "5_point"]]), axis=1)
#önceden WSS ile sıralamıştık ratingleri bu buna bakalım.
df.sort_values("weighted_sorting_score", ascending=False).head(20)
#şimdi de bar_score ile hesapladığımzı ratinglere bakalım.
df.sort_values("bar_score", ascending=False).head(20)
#bar score bize sadece ratinglere odaklanarak bir sıralama yaptı. bu yüzden iyi gibi ama yine social prooflar gözden kaçtı. yorum sayıları, satın alma sayıları...
#eğer tek odak sadece verilen puanlar olacaksa o zaman bar_score kullanabiliriz.

df[df["course_name"].index.isin([5, 1])].sort_values("bar_score", ascending=False)


####################
# Hybrid Sorting: BAR Score + Diğer Faktorler (sorting by rating, comment, purchase)


# Rating Products
# - Average
# - Time-Based Weighted Average
# - User-Based Weighted Average
# - Weighted Rating
# - Bayesian Average Rating Score

# Sorting Products
# - Sorting by Rating
# - Sorting by Comment Count or Purchase Count
# - Sorting by Rating, Comment and Purchase
# - Sorting by Bayesian Average Rating Score (Sorting Products with 5 Star Rated)
# - Hybrid Sorting: BAR Score + Diğer Faktorler (Sorting by Rating, Comment and Purchase)


def hybrid_sorting_score(dataframe, bar_w=60, wss_w=40):
    bar_score = dataframe.apply(lambda x: bayesian_average_rating(x[["1_point",
                                                                     "2_point",
                                                                     "3_point",
                                                                     "4_point",
                                                                     "5_point"]]), axis=1)
    wss_score = weighted_sorting_score(dataframe)
#bar_score ve wss_score belirli bir ağırlık vererek topluyoruz. böylece daha hassas sonuçlar elde etmiş olurum. böylece hem rating hem social proof göz önünde bulundurulur.
    return bar_score*bar_w/100 + wss_score*wss_w/100


df["hybrid_sorting_score"] = hybrid_sorting_score(df)

df.sort_values("hybrid_sorting_score", ascending=False).head(20)

df[df["course_name"].str.contains("Veri Bilimi")].sort_values("hybrid_sorting_score", ascending=False).head(20)


############################################
# Uygulama: IMDB Movie Scoring & Sorting

import pandas as pd
import math
import scipy.stats as st
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

df = pd.read_csv("datasets/movies_metadata.csv",
                 low_memory=False)  # DtypeWarning kapamak icin

df = df[["title", "vote_average", "vote_count"]]

df.head()
df.shape

########################
# Vote Average'a Göre Sıralama
########################

df.sort_values("vote_average", ascending=False).head(20)
#vote_count(oy sayısı) describe atıp çeyrek değerler veriyorum. çeyrek değerlere göre filmlerin aldıkları oy sayıları gelecektir.
df["vote_count"].describe([0.10, 0.25, 0.50, 0.70, 0.80, 0.90, 0.95, 0.99]).T
#sıralama yaparken 400 den kçk olması önkoşulunu koyuyorum.
df[df["vote_count"] > 400].sort_values("vote_average", ascending=False).head(20)

from sklearn.preprocessing import MinMaxScaler
#vote_countu 1-10 arasında standartlaştıralım.
#fit ile değişiklik bilgisi uygulanır. tranform ile dönüştürüyoruz.
df["vote_count_score"] = MinMaxScaler(feature_range=(1, 10)). \
    fit(df[["vote_count"]]). \
    transform(df[["vote_count"]])

########################
# ikisi de (vote_average * vote_count) 1-10 arasında o yuzden çarpabiliriz.
df["average_count_score"] = df["vote_average"] * df["vote_count_score"]

df.sort_values("average_count_score", ascending=False).head(20)


# IMDB Weighted Rating
#2015 yılına kadar bu yöntemle filmleri sıralıyorlardı.
# weighted_rating = (v/(v+M) * r) + (M/(v+M) * C)

# r = vote average - ilgili filmin puanı
# v = vote count - oy sayısı
# m=gereken oy sayısı
# R=ilgili filmin puanı
# M = minimum votes required to be listed in the Top 250 - gereken min. oy sayısı
# C = the mean vote across the whole report (currently 7.0) - genel bütün kitlenin ortalaması

# Film 1:
# r = 8 / film1 in puanı
# M = 500 / gereken min. oy sayısı
# v = 1000/ filmin aldığı oy sayısı

#formülde yerine yerleştir ->  (1000 / (1000+500))*8 = 5.33


# Film 2:
# r = 8 /film1 ile aynı
# M = 500/ film1 ile aynı
# v = 3000/ film 1 den fazla oy almış

# formülde yerine yerleştir ->  (3000 / (3000+500))*8 = 6.85
#film1 ile 2 ye bak. aynı puan, aynı min. oy sayısı ama film 2 daha fazla oy almış. bunu matematiksel olarak göstermiş olduk.



#puanı 9.5 olan başka bir film düşün. şimdi burada
# daha fazla puana sahip olmasına rağmen kendisinden daha fazla oy alan film 2 nin önüne geçemedi. gereken min. oy sayısının önemi çok net görülmekte.
#dersin başındaki 2 mouse örneğine geldik yine.
# (1000 / (1000+500))*9.5 =6.33

# Film 1:
# r = 8
# M = 500
# v = 1000

# Birinci bölüm / formülün sağ tarafını yapalım:
# (1000 / (1000+500))*8 = 5.33

# İkinci bölüm/formülün sol tarafını hesaplayalım:
# 500/(1000+500) * 7 = 2.33

# weighted_rating = (v/(v+M) * r) + (M/(v+M) * C)
#îmdb puanı hesaplama = sağ taraf + sol taraf
# Toplam = 5.33 + 2.33 = 7.66



# Film 2 içinde film 1 de ki işlemlerin aynısını yapalım:
# r = 8
# M = 500
# v = 3000

# Birinci bölüm:
# (3000 / (3000+500))*8 = 6.85

# İkinci bölüm:
# 500/(3000+500) * 7 = 1

# Toplam = 7.85



M = 2500
C = df['vote_average'].mean()

def weighted_rating(r, v, M, C):
    return (v / (v + M) * r) + (M / (v + M) * C)

df.sort_values("average_count_score", ascending=False).head(10)

weighted_rating(7.40000, 11444.00000, M, C)

weighted_rating(8.10000, 14075.00000, M, C)

weighted_rating(8.50000, 8358.00000, M, C)
#bizim yaptığımız
df["weighted_rating"] = weighted_rating(df["vote_average"],
                                        df["vote_count"], M, C)
#imdb nin yöntemi
df.sort_values("weighted_rating", ascending=False).head(10)

####################
# Bayesian Average Rating Score
#imdbye göre ilk 5 e bakalım.
# 12481                                    The Dark Knight
# 314                             The Shawshank Redemption
# 2843                                          Fight Club
# 15480                                          Inception
# 292                                         Pulp Fiction


#bar skorunu kullanarak deneyelim
def bayesian_average_rating(n, confidence=0.95):
    if sum(n) == 0:
        return 0
    K = len(n)
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    N = sum(n)
    first_part = 0.0
    second_part = 0.0
    for k, n_k in enumerate(n):
        first_part += (k + 1) * (n[k] + 1) / (N + K)
        second_part += (k + 1) * (k + 1) * (n[k] + 1) / (N + K)
    score = first_part - z * math.sqrt((second_part - first_part * first_part) / (N + K + 1))
    return score
#filmler kaç tane 1 yıldız 2 yıldız, 3 4 5....10  yıldız aldıysa bu değerleri parametre olarak gir.
#esaretin bedeli filmi için değerler(normalde (sitedeki güncel puan) 9.3 imiş, biizm hesapladığımız ise 9.14)
bayesian_average_rating([34733, 4355, 4704, 6561, 13515, 26183, 87368, 273082, 600260, 1295351])
#baba filmi için değerler (normalde (sitedeki güncel puan) 9.2 imiş, biizm hesapladığımız ise 8,94)
bayesian_average_rating([37128, 5879, 6268, 8419, 16603, 30016, 78538, 199430, 402518, 837905])

#bu hesabı tüm filmlere uygulamak istediğini düşün
#bu veri setinde yıldız sayılarından kaç kişi verdiyse bunun bilgisi vardır. (imdb_ratings.csv)
df = pd.read_csv("datasets/imdb_ratings.csv")
df = df.iloc[0:, 1:]


df["bar_score"] = df.apply(lambda x: bayesian_average_rating(x[["one", "two", "three", "four", "five",
                                                                "six", "seven", "eight", "nine", "ten"]]), axis=1)
df.sort_values("bar_score", ascending=False).head(20)


# Weighted Average Ratings
# IMDb publishes weighted vote averages rather than raw data averages.
# The simplest way to explain it is that although we accept and consider all votes received by users,
# not all votes have the same impact (or ‘weight’) on the final rating.

# When unusual voting activity is detected,
# an alternate weighting calculation may be applied in order to preserve the reliability of our system.
# To ensure that our rating mechanism remains effective,
# we do not disclose the exact method used to generate the rating.

# See also the complete FAQ for IMDb ratings.


#TAVSİYE SİSTEMİ GELİŞTİRMİŞ OLDUK. FİLM ANALİZLERİ GERÇEKLEŞTİRMEK İÇİNDE BUNLARI KULLANABİLİRİZ.