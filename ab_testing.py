#TEMEL İSTATİKSEL KAVRAMLAR

# Temel İstatistik Kavramları


import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# !pip install statsmodels
import statsmodels.stats.api as sms
from scipy.stats import ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu, \
    pearsonr, spearmanr, kendalltau, f_oneway, kruskal
from statsmodels.stats.proportion import proportions_ztest

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 10)
pd.set_option('display.float_format', lambda x: '%.5f' % x)


############################
# Sampling (Örnekleme)
#popülasyon ile örneklem arasındaki ilişkinin veri dünyasına ne şekilde yansıdığını görmeye çalışıyoruz.

#0-80 arasında 10000 tane sayı seç dedim ve bunlardan bir popülasyon oluşturdum.
populasyon = np.random.randint(0, 80, 10000)
populasyon.mean()
#ama bu 10.000 kişinin hepsini gezmek zor.
# o yüzden sen öyle bir grup seç ki bu grup o 10 bin kişiyi iyi bir şekilde temsil etsin bende o grup üzerinden çalışayım.
# böylece zaman ,para,iş yükü olarak daha az harcamış olursun. daha avantajlı. belirli bir hata payı tebikide olacaktır. ancak bu kabul edilebilirdir.
np.random.seed(115)
#10 bin tane vardı ben sadece 100 kişiyi seçmek istiyorum.
orneklem = np.random.choice(a=populasyon, size=100)
orneklem.mean()

#diyelim ki 10 tane örneklem çekmek istiyorum
np.random.seed(10)
orneklem1 = np.random.choice(a=populasyon, size=100)
orneklem2 = np.random.choice(a=populasyon, size=100)
orneklem3 = np.random.choice(a=populasyon, size=100)
orneklem4 = np.random.choice(a=populasyon, size=100)
orneklem5 = np.random.choice(a=populasyon, size=100)
orneklem6 = np.random.choice(a=populasyon, size=100)
orneklem7 = np.random.choice(a=populasyon, size=100)
orneklem8 = np.random.choice(a=populasyon, size=100)
orneklem9 = np.random.choice(a=populasyon, size=100)
orneklem10 = np.random.choice(a=populasyon, size=100)

(orneklem1.mean() + orneklem2.mean() + orneklem3.mean() + orneklem4.mean() + orneklem5.mean()
 + orneklem6.mean() + orneklem7.mean() + orneklem8.mean() + orneklem9.mean() + orneklem10.mean()) / 10

#ana kitle içeriisnden birden fazla örneklem seçip ortalamsaını aldığında daha da hassaslaşacaktır yaptığın hesaplama.

############################
# Descriptive Statistics (Betimsel İstatistikler) - KEŞİFÇİ VERİ ANALİZİ
#Tamamlayıcı - Açıklayıcı İstatistikler de denir.
#elimizdeki veri setini betimlemeye çalışacağız.

df = sns.load_dataset("tips")
#describe verideki sayısal değişkenleri seçerek bunları gösterir.
df.describe().T
#desribe aldığına bize çeyrek değerlerinide gösterir. ve çeyrek değerler ilgili değişkenlerin dağılımı hakkında bilgi vermektedir.
#eğer elimizdeki değişkeni çarpık ise yani içeriisnde aykırı değerler varsa ortalama değil medyan kullanmalıyız.
#eğer ortalam ile medyan birbirine çok yakın ise o zaman sorun yok.
# ama arada ciddi farklar varsa hem ortalama hem medyan hem de standart sapmasını söylemek mantıklı olacaktır.


############################
# Confidence Intervals (Güven Aralıkları)

# Tips Veri Setindeki Sayısal Değişkenler için Güven Aralığı Hesabı
#tips veri seti, içeriisnde resteron hesap ödemelerine ilişkin değişkenleri barındırır.
df = sns.load_dataset("tips")
df.describe().T

df.head()
#ilgili değişken için güven aralığını hesaplayalım.
#½95 güven ile ödenilen hesap ortalaması istatistiki olarak 18,66 ile 20.95 arasındadır. ½5de hata payı vardır.
# buradan yola çıkarak çalışanlarımıa ödeyeceğim maaşı, sergileyeceğim tutumu değiştirebilirim.
sms.DescrStatsW(df["total_bill"]).tconfint_mean()
#gelen bahşişlerin ortalama aralığı
sms.DescrStatsW(df["tip"]).tconfint_mean()

# Titanic Veri Setindeki Sayısal Değişkenler için Güven Aralığı Hesabı
df = sns.load_dataset("titanic")
df.describe().T
sms.DescrStatsW(df["age"].dropna()).tconfint_mean()

sms.DescrStatsW(df["fare"].dropna()).tconfint_mean()


######################################################
# Correlation (Korelasyon)
#yine tips veri setini kullanacağız.
# Bahşiş veri seti:
# total_bill: yemeğin toplam fiyatı (dikkat et bu fiyat içinde bahşiş ve vergi dahil)
# tip: bahşiş
# sex: ücreti ödeyen kişinin cinsiyeti (0=male, 1=female)
# smoker: grupta sigara içen var mı? (0=No, 1=Yes)
# day: gün (3=Thur, 4=Fri, 5=Sat, 6=Sun)/yemeğin yendiği günleri ifade etmektedir.
# time: ne zaman? (0=Day, 1=Night)/ ne zaman yemek yenmiş sabah mı akşam mı?
# size: grupta kaç kişi var?

df = sns.load_dataset('tips')
df.head()
#bahşişler ile ödenen hesap arasında bir korelasyon var mı?
# normalde ödenen hesap artttıkça bahşişin artmasını beklerim.

#total_bill(yemeğin toplam fiyatı) içinde bahşiş ve vergi dahildi o zaman bu bahşişi çıkar önce.
df["total_bill"] = df["total_bill"] - df["tip"]

#çıkan grafiği incelediğimde pozitif yönte orta şiddette bir ilişkilerinin olduğunu söyleyebilirim.
df.plot.scatter("tip", "total_bill")
plt.show()


#corr methodu iki değişken arasındaki korelasyonu vermektedir.
#tip ile total bill arasındaki korelasyonu verdiğimizde 0.57 çıkmaktadır. yani orta şiddetde pozitif korelasyon olduğu doğrulandı.
#yani ödenecek hesap tutarı arttıkça verilen bahşişde artacaktır.
df["tip"].corr(df["total_bill"])

######################################################
# AB Testing (Bağımsız İki Örneklem T Testi)
#gruplar bağımlı olabilir, daha fazla grup olabilir falan bunların hepsi için farklı testler kullanılır.
#ürün özelliklerini denemek için AB testleri kullanılır;
# -2 grubun ortalaması kıyaslanıyor olabilir.
# - Dönüşüm oranlarını kıyaslıyor olabiliriz.
# - ilgilendiğimiz herhangi bir olaya yönelik oranlarıda kıyaslıyor olabiliriz.
#yani daha çok özellik


# 1. Hipotezleri Kur (neyi sınamak istiyorsan ona yönelik hipotez kur)
# 2. Varsayım Kontrolü
#   - 1. Normallik Varsayımı
#   - 2. Varyans Homojenliği
# 3. Hipotezin Uygulanması
#   - 1. Varsayımlar sağlanıyorsa bağımsız iki örneklem t testi (parametrik test)
#   - 2. Varsayımlar sağlanmıyorsa mannwhitneyu testi (non-parametrik test)
# 4. p-value değerine göre sonuçları yorumla
# Not:
# - Normallik sağlanmıyorsa direk 2 numara. Varyans homojenliği sağlanmıyorsa 1 numaraya arguman girilir.
# - Normallik incelemesi öncesi aykırı değer incelemesi ve düzeltmesi yapmak faydalı olabilir.


############################
# Uygulama 1: Sigara İçenler ile İçmeyenlerin Hesap Ortalamaları Arasında İstatistiki Olarak Anlamlı bir Fark var mıdır?
#masada sigara içen birisi olduğunda daha fazla bahşiş bırakıyor mu acaba?
df = sns.load_dataset("tips")
df.head()
#sigara içen ve içmeyenleirn ortalamasına bakalım, farkın olduğunu göreceksin
df.groupby("smoker").agg({"total_bill": "mean"})
#bu fark şans eseri mi ortaya çıktı şimdi bunu incelemeye başla.


# 1. Hipotezi Kur
# H0: M1 = M2
# H1: M1 != M2

# 2. Varsayım Kontrolü
# 2.1.Normallik Varsayımı(bunu test etmek için shapiro testini kullanıcam)
# 2.2.Varyans Homojenliği (levene testi kullanılır)

# 2.1.Normallik Varsayımı:
# H0: Normal dağılım varsayımı sağlanmaktadır.
# H1:..sağlanmamaktadır.

#shapiro methodu= bir değişkenin dağılımının normal olup olmadığını test eder.
test_stat, pvalue = shapiro(df.loc[df["smoker"] == "Yes", "total_bill"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# p-value < ise 0.05'ten HO RED.
# p-value < değilse 0.05 H0 REDDEDILEMEZ.


test_stat, pvalue = shapiro(df.loc[df["smoker"] == "No", "total_bill"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))


############################
# Varyans Homojenligi Varsayımı (levene testi kullanılır)
#levene testine 2 grup gönderilir ve bu 2 gruba göre bana varyans homejenliği varsayımının sağlanıp sağlanmadığını söyler.

# H0: Varyanslar Homojendir
# H1: Varyanslar Homojen Değildir

test_stat, pvalue = levene(df.loc[df["smoker"] == "Yes", "total_bill"],
                           df.loc[df["smoker"] == "No", "total_bill"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# p-value < ise 0.05 'ten HO RED.
# p-value < değilse 0.05 H0 REDDEDILEMEZ.


# 3 ve 4. Hipotezin Uygulanması
############################

# 1. Varsayımlar sağlanıyorsa bağımsız iki örneklem t testi (parametrik test)
# 2. Varsayımlar sağlanmıyorsa mannwhitneyu testi (non-parametrik test)

############################
# 1.1 Varsayımlar sağlanıyorsa bağımsız iki örneklem t testi (parametrik test)
#Ttest methodu şu d;urumlarda kullanılır
# eğer normallik varsayımı sağlanıyorsa beni kullanabilirsin der,
# normallik varsayımı sağlanıyor ve varyans homejenliği varsayımı sağlanıyorsa da kullanabilirisin,
# NV sağlanıyor VHV sağlanmıyorsa da kullanabilirsin. ama bu durumda equaal_var=False demelisin.

test_stat, pvalue = ttest_ind(df.loc[df["smoker"] == "Yes", "total_bill"],
                              df.loc[df["smoker"] == "No", "total_bill"],
                              equal_var=True)
#0.18 çıktı anlamlı bir farklılık yok ççünkü h0 reddedemedik.
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# p-value < ise 0.05 'ten HO RED.
# p-value < değilse 0.05 H0 REDDEDILEMEZ.

############################
# 1.2 Varsayımlar sağlanmıyorsa mannwhitneyu testi (non-parametrik test)


test_stat, pvalue = mannwhitneyu(df.loc[df["smoker"] == "Yes", "total_bill"],
                                 df.loc[df["smoker"] == "No", "total_bill"])

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))



############################
# Uygulama 2: Titanic Kadın ve Erkek Yolcuların Yaş Ortalamaları Arasında İstatistiksel Olarak Anl. Fark. var mıdır?
############################

df = sns.load_dataset("titanic")
df.head()
#farkın var olduğunu görüyorum. acaba bu fark şans eseri mi ortaya çıktı? bunun almak için ist. test yapmalıyım.
df.groupby("sex").agg({"age": "mean"})


# 1. Hipotezleri kur:
# H0: M1  = M2 (Kadın ve Erkek Yolcuların Yaş Ortalamaları Arasında İstatistiksel Olarak Anl. Fark. Yoktur)
# H1: M1! = M2 (... vardır)


# 2. Varsayımları İncele

# Normallik varsayımı
# H0: Normal dağılım varsayımı sağlanmaktadır. (shapiro testi)
# H1:..sağlanmamaktadır


test_stat, pvalue = shapiro(df.loc[df["sex"] == "female", "age"].dropna())
#p value= 0.0071 çıktı h0 reddedilir.
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

test_stat, pvalue = shapiro(df.loc[df["sex"] == "male", "age"].dropna())
#p value = 0.0000 h0 reddedilir.
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

## yukarıda yaptık gördük iksinde de ho reddettik. yanı normallik varsayım sağlanmıyor.
# şimdi direkt normalde non parametrik hesaplamaya geçmeliyiz. ama hatırlama amaçlı varyans homojenliğinide inceleyelim.
# Varyans homojenliği (levene methodu)
# H0: Varyanslar Homojendir
# H1: Varyanslar Homojen Değildir

test_stat, pvalue = levene(df.loc[df["sex"] == "female", "age"].dropna(),
                           df.loc[df["sex"] == "male", "age"].dropna())
#p value=0.971 o zaman h0 reddedemen varyansalar homojendir. ama varsayım zaten sağlanmadı ben bunu öylesine inceledim.
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# Varsayımlar sağlanmadığı için nonparametrik
#mannwhitneyu testi : non-parametrik 2 örneklem karşılaştırma testini kullanacağız.

test_stat, pvalue = mannwhitneyu(df.loc[df["sex"] == "female", "age"].dropna(),
                                 df.loc[df["sex"] == "male", "age"].dropna())
#p value= 0.0261. ho reddedilir.
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))














############################
# Uygulama 3: Diyabet Hastası Olan ve Olmayanların Yaşları Ort. Arasında İst. Ol. Anl. Fark var mıdır?
############################

df = pd.read_csv("datasets/diabetes.csv")
df.head()
#Outcome=1 ise diyabet
#outcome=0 ise diyabet değil.

#diyabet olmayanların yaş ortalaması 31,9; diyabet olanların yaş ortalaması 37,06. fark olduğunu görüyoruz ama test et.
df.groupby("Outcome").agg({"Age": "mean"})

# 1. Hipotezleri kur
# H0: M1 = M2
# Diyabet Hastası Olan ve Olmayanların Yaşları Ort. Arasında İst. Ol. Anl. Fark Yoktur
# H1: M1 != M2
# .... vardır.

# 2. Varsayımları İncele

# Normallik Varsayımı (H0: Normal dağılım varsayımı sağlanmaktadır.)
test_stat, pvalue = shapiro(df.loc[df["Outcome"] == 1, "Age"].dropna())
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

test_stat, pvalue = shapiro(df.loc[df["Outcome"] == 0, "Age"].dropna())
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
#yukarıdaki kod çalıştığında ho reddedilir çünkü p value=0.000
# Normallik varsayımı sağlanmadığı için nonparametrik.//medyanların karşılaştıırlması

# Hipotez (H0: M1 = M2)
test_stat, pvalue = mannwhitneyu(df.loc[df["Outcome"] == 1, "Age"].dropna(),
                                 df.loc[df["Outcome"] == 0, "Age"].dropna())
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
#buraya kadar yazdığım kdolarla şunu anlıyorum ki aralarında anlamlı bir farklılık vardır. diyabet olanların yaş ortalaması daha yüksektir.








###################################################
# İş Problemi: Kursun Büyük Çoğunluğunu İzleyenler ile İzlemeyenlerin Puanları Birbirinden Farklı mı?

# H0: M1 = M2 (... iki grup ortalamaları arasında ist ol.anl.fark yoktur.)
# H1: M1 != M2 (...vardır)

df = pd.read_csv("datasets/course_reviews.csv")
df.head()
#veriye göre özelleştirmeler yapabilirim. mesela kursu az izleyenlere mail at ve hadi a< daha gayret gibi özel içerik hazırla.
#eğer kursda ilerlemesi iyiyse süpersin , teşekkürler vs gibi mailler gönderebilirim.

df[(df["Progress"] > 75)]["Rating"].mean()

df[(df["Progress"] < 25)]["Rating"].mean()

#1.grup için normallik varsayımını inceleyelim.
test_stat, pvalue = shapiro(df[(df["Progress"] > 75)]["Rating"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

#2.grup için normallik varsayımını inceleyelim.
test_stat, pvalue = shapiro(df[(df["Progress"] < 25)]["Rating"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

#non-parametrik kullanmalısın.
test_stat, pvalue = mannwhitneyu(df[(df["Progress"] > 75)]["Rating"],
                                 df[(df["Progress"] < 25)]["Rating"])

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

#buradan sonuçla istatiski olarak gerçek bir farkın olduğu çıkarımını yapabiliriz.


######################################################
# AB Testing (İki Örneklem Oran Testi)

# H0: p1 = p2
# Yeni Tasarımın Dönüşüm Oranı ile Eski Tasarımın Dönüşüm Oranı Arasında İst. Ol. Anlamlı Farklılık Yoktur.
# H1: p1 != p2
# ... vardır

#Tasarım 1 de 1000 kişinin çarptiğını ve 300 kişinin kaydolduğunu varsıyorum.
# 2.tasarımda ise 1100 kişi çarptı 250 kişi kaydolduğunu varsayıyorum.
#başarı sayıları ve gözlem sayıları ayrı bır arrayde tanımlanır.

basari_sayisi = np.array([300, 250])
gozlem_sayilari = np.array([1000, 1100])
#ho reddedildi demek ki anlamlı bir farklılık var.
proportions_ztest(count=basari_sayisi, nobs=gozlem_sayilari)


basari_sayisi / gozlem_sayilari


############################
# Uygulama: Kadın ve Erkeklerin Hayatta Kalma Oranları Arasında İst. Olarak An. Farklılık var mıdır?


# H0: p1 = p2 veya p1-p2=0 aynı şey
# Kadın ve Erkeklerin Hayatta Kalma Oranları Arasında İst. Olarak An. Fark yoktur

# H1: p1 != p2
# .. vardır

df = sns.load_dataset("titanic")
df.head()
#kadınların hayatta kalma oranlarının ortalaması = 0.74
df.loc[df["sex"] == "female", "survived"].mean()
#erkeklerin hayatta kalma oranlarının ortalaması = 0.18
df.loc[df["sex"] == "male", "survived"].mean()
#0.74 ve 0.18 arada fark bariz var gözüküyor.

#kadınlar için başarı sayısı
female_succ_count = df.loc[df["sex"] == "female", "survived"].sum()
#erekekler için başarı sayısı
male_succ_count = df.loc[df["sex"] == "male", "survived"].sum()
#başarı sayısı ve gözlem sayısını hesapla
#1.bölüm count, başarı sayıları girilir.
#2.bölüm nobs, toplam gözlem sayıları girilir.
test_stat, pvalue = proportions_ztest(count=[female_succ_count, male_succ_count],
                                      nobs=[df.loc[df["sex"] == "female", "survived"].shape[0], #kadınlar için toplam gözlem sayısı
                                            df.loc[df["sex"] == "male", "survived"].shape[0]])  #erkekler için toplam gözlem sayısı
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
#Analmalı fark olduğunu ispatlamış olduk.





######################################################
# ANOVA (Analysis of Variance)


# İkiden fazla grup ortalamasını karşılaştırmak için kullanılır.
#garsonlar haftanın günlerinde ödenen hesaplarda bir farklılık olduğunu düşünüyorlar.
# bu farklılık şans eseri mi ortaya çıktı. bunun için istatiksel yöntemler kullanarak incele.
df = sns.load_dataset("tips")
df.head()
# 2 gün hafta içi, 2 gün de hafta sonu olamk üzere elimde veriler var. bunların ortalamasını alıyorum.
df.groupby("day")["total_bill"].mean()


# 1. Hipotezleri kur
#totalde 4 günün verisi var o yüzdem m1,m2,m3,m4
# HO: m1 = m2 = m3 = m4
# Grup ortalamaları arasında fark yoktur.

# H1: .. fark vardır

# 2. Varsayım kontrolü

# Normallik varsayımı
# Varyans homojenliği varsayımı

# Varsayım sağlanıyorsa ->one way anova (tek yönlü anova testi)
# Varsayım sağlanmıyorsa -> kruskal (2 den fazla grup karşılaştırmaları için kullanılan non-parametrik testtir.)

# H0: Normal dağılım varsayımı sağlanmaktadır.
#normallik varsayımını test etmek için ;
#bir kategorik değişkenin sınıflarını üzerinde gezebilececeğimiz litaratif bir nesneye(listeye) çevirdik.
for group in list(df["day"].unique()):
    pvalue = shapiro(df.loc[df["day"] == group, "total_bill"])[1]
    print(group, 'p-value: %.4f' % pvalue)
#Sat p-value: 0.0000  -> 0.05 den küçük old.için H0 REDDEDİLİR.
#Thur p-value: 0.0000 -> 0.05 den küçük old.için H0 REDDEDİLİR.
#Fri p-value: 0.0409  -> 0.05 den küçük old.için H0 REDDEDİLİR.

# H0: Varyans homojenliği varsayımı sağlanmaktadır. burada da levene testi kullanılır.
#p-value = 0.5741 çıktı o zaman varyans homojenliği sağlandı reddedemeyz.
# ama zaten normallik parametresinden red yediğimiz için her türlü non-paremetriğe düştük. yanı bunu öylesine hesapladık.
test_stat, pvalue = levene(df.loc[df["day"] == "Sun", "total_bill"],
                           df.loc[df["day"] == "Sat", "total_bill"],
                           df.loc[df["day"] == "Thur", "total_bill"],
                           df.loc[df["day"] == "Fri", "total_bill"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))


# 3. Hipotez testi ve p-value yorumu

# Hiç biri varsayımı sağlamıyordu.
#ortalamalar ve medyanlar arasında bir farklılık var ama bu anlamlı mı?
df.groupby("day").agg({"total_bill": ["mean", "median"]})


# HO: Grup ortalamaları arasında ist ol anl fark yoktur
#varsayımın sağlandığını düşünelim. o zaman parametrik anova test kullanacağız.
# parametrik anova testi: pvalue=0.04245383328952047 çıktı HO REDDEDİLİR.Analmlaı farklılık vardır deriz.
f_oneway(df.loc[df["day"] == "Thur", "total_bill"],
         df.loc[df["day"] == "Fri", "total_bill"],
         df.loc[df["day"] == "Sat", "total_bill"],
         df.loc[df["day"] == "Sun", "total_bill"])

#burada bizim varsayımlarımız sağlanmamıştı. nonparemetrik test kullanmalıyız.
# Nonparametrik anova testi: pvalue=0.01543300820104127 çıktı. HO reddedilir. Anlamlı farklılık var.
kruskal(df.loc[df["day"] == "Thur", "total_bill"],
        df.loc[df["day"] == "Fri", "total_bill"],
        df.loc[df["day"] == "Sat", "total_bill"],
        df.loc[df["day"] == "Sun", "total_bill"])
#okey anlamlı fark var tamam ama kimden kaynaklanıyor bu durum.
#biz bu karşılaştırmayı yapmak için "tukey testini" kullanacağız.
#multicomp = çoklu karşılaştırma
#total_bill = ödenen hesap
#df["days"] = istediğimiz grup değişkeni girilir.

from statsmodels.stats.multicomp import MultiComparison
comparison = MultiComparison(df['total_bill'], df['day'])
#yaygınca kbaul edilen kıyaslama noktası 0.05dir. 0.05 den küçükse reddederiz.
tukey = comparison.tukeyhsd(0.05)
print(tukey.summary())
#biraz verileri kurcaladığımzıda farklılığın asıl kaynağının hafta sonu olduğu anlaşılmaktadır????