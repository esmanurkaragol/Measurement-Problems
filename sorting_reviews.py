#SORTİNG REVİEWS
import pandas as pd
import math
import scipy.stats as st

pd.set_option("display.max_columns")
pd.set_option("display.expand_frame_repr", False)
pd.set_option("display.float_format", lambda x: "%.5f" %x)

#ÜST ALT FARKI SKORU
#up-down diff score = (up ratings) - (down ratings)
#ikili puanlandırma var (like-dislike)

# Review 1: 600 up(like) 400 down(dislike) total 1000
# Review 2: 5500 up 4500 down total 10000

#yorumların farkını dönderecek bir fonksiyon yazalım.
def score_up_down_diff(up, down):
    return up - down

# Review 1 Score= 200
score_up_down_diff(600, 400)

# Review 2 Score=1000
score_up_down_diff(5500, 4500)

#bu sonuçlara göre hangi yanıtı daha yukarıda tutardınız?
#şimdi farklılıklara bakınca 2.olan 1000, 1.olan 200 . bu durumda 2.olan kazanıyor gibi gözükmektedir.
#ama yüzdelikten dolayı 1.kazanıyor. bu durumda böyle bir fark yöntemi pek doğru olmaz.
#yanlış skorlama yöntemidir. bunu kullanmak doğru değil. hassas ölçüm yapmaz.

###############################################
# Score = Average rating = (up ratings) / (all ratings) bu förmülü uyguladığımızda;
# elimizde likeların oranı oalcaktır. bu sayede gözlem yürütebiliriz.
#faydalılık(beğenilme) oranı olarakda düşünebilirsin bu skoru.

def score_average_rating(up, down):
    if up + down == 0:
        return 0
    return up / (up + down)

score_average_rating(600, 400)
score_average_rating(5500, 4500)

# Review 1: 2 up (2 like), 0 down(dislike), total 2 (toplamda 2 yorum)
# Review 2: 100 up, 1 down, total 101

score_average_rating(2, 0)
score_average_rating(100, 1)
#yukarıda yazmış olduğumuz "Score = Average rating = (up ratings) / (all ratings)" formülü okey ama bu seferde frekans bilgisini kaçırdı.
#sayı yüksekliğini, frekans yüksekliğini göz önünde bulunduramadı. bu nedenle bunu yapmak doğru değil.

# WİLSON LOWER BOUND SCORE
#bize herhangi bir 2 ürün, 2 yorumu... skorlama imkanı verir.
#bir güven aralığı hesaplar bu güven aralığının alt sınırınıda wlb skor olarak kabul eder.
#2 sonucu olan olayın nasıl gerçekleşebileceği olasığılığını hesaplar.
#hem oran hem de frekans bilgisi eş zamanlı göz önünde bulunduracak şekilde bir sıralama skoru elde etmemizi sağlar.
# up =600, down =400 bu durumda (up ratings) / (all ratings)= 600/400=0.6
#biz bu 0.6 için bir güven aralığı hesapladığımzıda örnek olarak 0.5 - 0.7 arasında bazı aralıklar belirleriz.bu aralıklarda da ;
# istatiksel olarak 100 kullanıcıdan 95 i (%95 güven ile bu sabittir) bu yorumla ilgili bir etkileşim sağladığında %5 yanılma payım olmakla birlikte bu yorumun up oranı ½0.5-½0.7 arasında olacaktır yorumunu yapabiliyorum.

#0.5 ile 0.7 arasında dedik ya hani tek skor belirlemem gerektiği için alt skor olan 0.5 i kabul ediyorum

# WİLSON LOWER BOUND SCORE= 0.5


#şimdi bu olayı tüm gözlem birimlerine uygula bakalım.
def wilson_lower_bound(up, down, confidence=0.95):
    """
    Wilson Lower Bound Score hesapla

    - Bernoulli parametresi p için hesaplanacak güven aralığının alt sınırı WLB skoru olarak kabul edilir.
    - Hesaplanacak skor ürün sıralaması için kullanılır.
    - Not:
    Eğer skorlar 1-5 arasıdaysa 1-3 negatif, 4-5 pozitif olarak işaretlenir ve bernoulli'ye uygun hale getirilebilir.
    Bu beraberinde bazı problemleri de getirir. Bu sebeple bayesian average rating yapmak gerekir.

    Parameters
    ----------
    up: int
        up count
    down: int
        down count
    confidence: float
        confidence

    Returns
    -------
    wilson score: float

    """
    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * up / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)


wilson_lower_bound(600, 400)    #wlb=0.56
wilson_lower_bound(5500, 4500)  #wlb=0.54
#bu durumda wlb=0.56 olan daha iyi diyebilrim.oransal olarak daha iyi.
#wlb=0.54 olan değerlere bakınca çok daha yüksek bu nedenle daha güvenilirdir.

wilson_lower_bound(2, 0)    #wlb=0.34
wilson_lower_bound(100, 1)  #wlb=0.94



###################################################
# Case Study

up = [15, 70, 14, 4, 2, 5, 8, 37, 21, 52, 28, 147, 61, 30, 23, 40, 37, 61, 54, 18, 12, 68]
down = [0, 2, 2, 2, 15, 2, 6, 5, 23, 8, 12, 2, 1, 1, 5, 1, 2, 6, 2, 0, 2, 2]
comments = pd.DataFrame({"up": up, "down": down})



# score_pos_neg_diff
comments["score_pos_neg_diff"] = comments.apply(lambda x: score_up_down_diff(x["up"],
                                                                             x["down"]), axis=1)

# score_average_rating
comments["score_average_rating"] = comments.apply(lambda x: score_average_rating(x["up"], x["down"]), axis=1)

# wilson_lower_bound
comments["wilson_lower_bound"] = comments.apply(lambda x: wilson_lower_bound(x["up"], x["down"]), axis=1)


#sıralama aslında wlb ye göre yapılmalı.
comments.sort_values("wilson_lower_bound", ascending=False)
comments.sort_values






