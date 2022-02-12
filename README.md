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
  
 
