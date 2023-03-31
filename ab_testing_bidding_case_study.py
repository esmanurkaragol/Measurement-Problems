#############################GÖREV1 : veriyi hazırlama ve analiz etme
#adım1
import pandas as pd
from scipy.stats import ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu, \
    pearsonr, spearmanr, kendalltau, f_oneway, kruskal

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.width", 500)
pd.set_option("display.expand_frame_repr", False)
pd.set_option("display.float_format", lambda x: "%5.f" %x)

dataframe_control = pd.read_excel("datasets/ab_testing.xlsx", sheet_name="Control Group")
dataframe_test = pd.read_excel("datasets/ab_testing.xlsx", sheet_name= "Test Group")

df_control = dataframe_control.copy()
df_test = dataframe_test.copy()
#ADIM2
#DRY
def check_df (df, head=5):
    print("#############  SHAPE  ###############")
    print(df.shape)
    print("#############  TYPES  ###############")
    print(df.dtypes)
    print("#############  HEAD  ###############")
    print(df.head(10))
    print("#############  TAİL   ###############")
    print(df.tail(10))
    print("#############  NA  ###############")
    print(df.isnull().sum())
    print("#############  QUANTİLE  ###############")
    print(df.quantile([0,0.05,0.50,0.95,0.99,1]).T)

check_df(df_control)
check_df(df_test)

#ADIM3
df_control["group"] = "control"
df_test["group"] = "test"
#axis= 0: satır bazlı birleştirme (alt alta)
#axis=1: sütun bazlı birleştirme
#ignore_index=True: indexi 0 dan başlatarak birleştirir.
#ignore_index= False: idexi kaldığı yerden birleşmeye devam eder

df = pd.concat([df_control, df_test], axis=0, ignore_index=True)
df.head(5)
df.tail(5)

############################ GÖREV 2
# average bidding ile maximum bidding arasında İstatistiki Olarak Anlamlı bir Fark var mı?

#adım1 (hipotez kur)
# H0: M1 = M2
# H1: M1 != M2

# adım 2
df.groupby("group").agg({"Purchase": "mean"})

######################### GÖREV 3

#3.1. adım1 (varsayım kontrolü)

#3.1.1. Normallik Varsayımı (shapiro test):
# H0: Normal dağılım varsayımı sağlanmaktadır. p>0.05 HO reddedilmez.
# H1:..sağlanmamaktadır. p<0.05 HO RED.
test_stat, pvalue = shapiro(df.loc[df["group"] == "control", "Purchase"].dropna())
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
#p-value = 0.5891 --> HO REDDEDİLMEZ --> Normallik Varsayımı sağlandı --> VH de incele

test_stat, pvalue = shapiro(df.loc[df["group"] == "test", "Purchase"].dropna())
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
#p-value = 0.1541 --> HO REDDEDİLMEZ --> Normallik Varsayımı sağlandı --> VH de incele

#3.1.2. Varyans Homojenliği (levene test):
# H0: Varyanslar Homojendir.  p>0.05 HO reddedilmez.
# H1: Varyanslar Homojen Değildir.  p<0.05 HO reddedilir.
test_stat, pvalue = levene(df.loc[df["group"] == "control", "Purchase"].dropna(),
                           df.loc[df["group"] == "test", "Purchase"].dropna())
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
#p-value = 0.1083 --> HO REDDEDİLEMEZ --> Varyans Homojenliği Sağlandı.


#3.2. adım2 (uygun test seçimi)

# Parameetrik Test Olan BAĞIMSIZ İKİ ÖRNEKLEM T-TEST kullanmalıyız.
#H0: iki grup arasında istatistiki olarak anlamlı fark yoktur. --> p> 0.05
#H1: iki grup arasında istatistiki olarak anlamlı fark vardır. --> p< 0.05
test_stat, pvalue = ttest_ind(df.loc[df["group"] == "control", "Purchase"].dropna(),
                              df.loc[df["group"] == "test", "Purchase"].dropna(),
                              equal_var=True)
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

#3.3. adım3
#p-value = 0.3493 --> ho reddedilemez --> ANLAMLI FARK YOKTUR


################################ GÖREV 4
#adım1;
#adım2: istatistiki olarak incelemeler yaptığımızda anlamlı bir farkın olmadığı anlaşıldı.
#bu durumda ilk başta ortalamalarına baktığımzıda fark var gibi gözükse de aslında bunun şansa bağlı olarak ortaya çıktığını görüyoruz.
#facebook' un yeni teklif türü satın alma bazında (purchase) incelediğimizde eskisine göre daha fazla dönüşüm getirmemektedir.
