# Maskeni_KontrolEt

Uygulanan  proje : Yüz de bir maske olduğuna doğrulamak için görüntülerin işlenmesi.
Projenin ilk amacı Hastanelerde , okullarda ve Alışveriş Merkezinde maskenin varlığını kontrol etme ,mekanizmasını hızlandırmaya yardımcı olmaktir.
Bu Proje önceliklerinden biri corona hastalik nedeniyle zor zamanlardan geçiyoruz ve önleme en iyi çözüm.
 bu adımların her birinde kullanılan ilkeleri ve daha fazla yaygın olarak bulunan insan algılama ve yüz algılama algoritmalarının kullanımını özetlemektedir.  Algoritmanın test video dizilerindeki performansının analizi, maskeli yüz algılama performansında daha fazla iyileştirme için faydalı bilgiler verir.Yaklaşımdaki sınırlamalar, tekniğin performansının iyileştirilmesi için daha fazla analiz yapılmasını gerektirir.
 Maskeli yüzleri algılamak için üç dikkatli evrişimli sinir ağından oluşan kademeli bir temele dayanan yeni bir CNN çerçevesi öneriyorlar.  Ayrıca maskeli yüz eğitim örneklerinin azlığı nedeniyle.  tasarlanmış CNN modellerimize ince ayar yapmak için "maskeli veri kümesi" adlı yeni bir veri kümesi.  Değerlendirme,maskeli test setinde maskeli yüz algılama algoritmasını önerdi ve performans elde etti.  
İki tane farkli yöntem kullandım .
Birincisi uygulamaya bir fotoğraf eklediğmiz zaman fotoğraftaki yüz maskelimi maskesizmi tanımlar. Maskeli ise yüz üzerinde yeşil kare çizilir , Maskesiz ise yüz üzerinde kırmızı kare çizilir.
İkinci yöntem Kamera olarak yaptık , kameranın karşısında kişi maskelimi maskesizmi aynı şekilde tanımlar, Yine  Maskeli ise yüz üzerinde yeşil kare çizilir ,Maskesiz ise yüz üzerinde kırmızı kare çizilir.
Bizim veri setimiz mesela 800 maskeli, 800 maskesiz, ondan sonra görütü işlemi yapmak için birkaç kütüphane kullandık (Numpy ,Tensorflow ,Keras, Glob ,Matplotlib  ,Opencv) .
Ondan sonra yüzü kesicez sadece yüz tanılmlansın diye Sonrada test yapar (Maskeli mi , Maskesiz mi).
Classfication yaptığmız zaman iki Sınıfa ayırılır: Birincisi Maskeli, İkincisi Maskesiz.
Yüzümüz Maskeli ise : Verimiz (True) olarak gösterilecektir.
Yüzümüz Maskesiz ise : Verimiz (False) olarak gösterilecektir.

![image](https://user-images.githubusercontent.com/47948105/194058503-3950e7c1-fa40-4af5-a7f1-6637d830c265.png)

![image](https://user-images.githubusercontent.com/47948105/194058732-35adb6df-ccb0-4fc9-b732-a53cb90e6d50.png)

![image](https://user-images.githubusercontent.com/47948105/194058544-85d0ef35-f19a-44e1-9d46-2da28e2e6a49.png)

![image](https://user-images.githubusercontent.com/47948105/194058620-9c7fd246-f3d4-4078-8af9-a208add9f016.png)

![image](https://user-images.githubusercontent.com/47948105/194058641-4a9ad773-6186-4339-b455-63682ade67a7.png)

![image](https://user-images.githubusercontent.com/47948105/194058679-2131eda3-06ec-4f6b-bf62-7fb0c55cd0e4.png)
