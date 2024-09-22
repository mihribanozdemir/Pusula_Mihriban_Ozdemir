Mihriban Özdemir
Pusula Talent Academy Case Study Çalışması 
Bu projede, bir sağlık veri seti üzerinde Exploratory Data Analysis (EDA) ve veri ön işleme işlemleri yapılmıştır. Veri seti, çeşitli demografik bilgiler (cinsiyet, yaş, kilo, boy, vs.), yan etki başlangıç tarihleri ve kronik hastalık bilgilerini içermektedir. Projede, hem kategorik hem de sayısal veriler üzerinde işlemler uygulanmış ve gerekli grafiksel görselleştirmeler yapılmıştır.

Gerekli Kurulumlar
Projeyi çalıştırabilmek için aşağıdaki paketlerin yüklü olması gerekmektedir:

Pandas
NumPy
Matplotlib
Seaborn
Sklearn
Missingno (Eksik veri görselleştirme için)
Bu paketleri yüklemek için aşağıdaki komutları kullanabilirsiniz:

pip install pandas numpy matplotlib seaborn scikit-learn missingno

Proje İçeriği ve Aşamaları
1. Veri Yükleme ve İlk İnceleme
Veri seti yüklendikten sonra, verinin genel yapısını anlamak amacıyla sütunlar, veri türleri ve eksik veri sayıları incelenmiştir.

2. Kategorik Değişkenlerin Analizi
Kategorik değişkenlerin frekans dağılımı, görselleştirilerek analiz edilmiştir. sns.countplot() kullanarak cinsiyet, il ve kan grubu gibi kategorik değişkenlerin dağılımı grafiklerle sunulmuştur.

3. Sayısal Değişkenlerin Analizi
Sayısal değişkenler (kilo, boy, yaş, vb.) describe() metodu ile analiz edilmiş ve histogramlar ile görselleştirilmiştir. Bu sayısal değişkenlerin merkezi eğilim (mean, median) ve dağılım bilgileri incelenmiştir.

4. Eksik Verilerin Analizi ve Doldurulması
Veri setindeki eksik değerler missingno ile görselleştirilmiş ve hangi sütunlarda ne kadar eksik veri bulunduğu tespit edilmiştir.

Eksik veriler kategorik değişkenler için "Bilinmiyor", sayısal değişkenler için ise medyan ile doldurulmuştur.

5. Korelasyon Analizi
Veri setindeki sayısal değişkenler arasında korelasyon olup olmadığını anlamak amacıyla korelasyon matrisi oluşturulmuştur.

6. Outlier (Aykırı Değer) Tespiti
Sayısal değişkenlerde aykırı değer olup olmadığı IQR (Interquartile Range) yöntemi ile incelenmiştir. Aykırı değerler tespit edildiğinde bunlar veri setinden çıkarılabilir veya daha ileri analizler için işaretlenebilir.

7. Veri Dönüşümleri ve Encoding
Kategorik değişkenler üzerinde iki farklı dönüşüm uygulanmıştır:

İkili sınıfa sahip kategoriler için Label Encoding
Çoklu sınıfa sahip kategoriler için One-Hot Encoding

8. Pivot Table ve Gruplama
Veriler üzerinde çeşitli pivot tablolar oluşturularak, ilaç kullanımı, yan etki süresi ve demografik özellikler arasındaki ilişki incelenmiştir.

Sonuçlar
Veri analizi ve ön işleme süreçleri başarıyla tamamlanmış olup, elde edilen bulgular ve görselleştirmeler raporlanmıştır. Aykırı değerlerin kontrolü, eksik verilerin doldurulması, kategorik değişkenlerin encoding işlemleri ve korelasyon analizleri yapılmıştır.

