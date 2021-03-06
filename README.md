# Audio Authenticity Identification by Background Noise Comparison
НИР **"Сравнение условий получения аудиозаписей на основе анализа фоновых шумов"**  

Общая постановка задачи:  
*В рамках криминалистической экспертизы решается задача установления подлиности частей аудиозаписи на основе анализа фоновых шумов. Фоновый шум аудиозаписи должен иметь одинаковые параметры по всей фонограмме. Если фоновый шум какого-то участка аудиозаписи отличается от остальных, участок считается вставкой.*

Постановка задачи на данный момент:  
*Имеются две аудиозаписи фоновых шумов, необходимо установить, идентичны ли они.*

Допущения, упрощающие нам жизнь:
* Работаем с аудиозаписями, которые представляют из себя *только* фоновый шум
* Две аудиозаписи считаются записанными в одинаковых условиях, если они нарезаны из одного длинного фрагмента
 
Примерный план работ  по практике на ближайшее время:
- [x] Набрать начальный датасет фоновых шумов в разных локациях на разные устройства.
- [x] Исследовать статистические методы:
  - [x] Реализовать простейшие статистические методы сравнения двух фрагментов фоновых шумов:
    - [x] Критерий хи-квадрат Пирсона, усредненный по частотным срезам,
    - [x] Критерий Стьюдента, усредненный по частотным срезам,
    - [x] Критерий Манна-Уитни-Вилкоксона, усредненный по частотным срезам,
    - [x] Среднее по трем статистическим критериям, усредненный по частотным срезам.
  - [x] Обучить по классификатору на каждый исследованный критерий.
- [x] Исследовать методы машинного обучения на признаковом пространстве, построенном при помощи статистических критериев:    
  - [x] Для начала визуализировать результаты работы статметодов - набор трехмерных точек в осях (U пирсон, U стьюдент, U манн).
  - [x] Обучить классификатор в построенном признаковом пространстве.
- [x] Исследовать сиамские сети (с VGG16 без последних двух слоев в качестве основы для подсетей):
  - [x] Сравнительное исследование методов сравнения пар векторов-дескрипторов:
    - [x] Сравнительное исследование различных линейных метрик близости векторов-дескрипторов:
       - [x] Евклидово расстояние,
       - [x] Расстояние Чебышева,
       - [x] Косинусное сходство.
     - [x] Сравнительное исследование различных методов классификации пар векторов-дескрипторов:
       - [x] MLP,
       - [x] Random Forest,
       - [x] Gradient Boosting.       
     - [ ] Сравнение результатов с классическими для обработки аудиосигналов методами:
       - [ ] Мера Кульбака-Лейблера (отложено),
       - [ ] Мера Итакуры-Сайто (отложено),
       - [x] Фингерпринтинг.
  - [ ] Cравнительное исследование различных моделей подсетей. (отложено)   
  - [ ] Использование различных входных данных для модели:
    - [x] Спектрограмма,
    - [x] Мел-спектрограмма,
    - [ ] Сравнительное исследование влияния окна Фурье-преобразования на качество работы сети. (отложено)
- [x] Работа с датасетом:
  - [x] Преобразовать к формату, с которым работает сеть.
  - [x] Можно записывать самостоятельно на различные устройства, в различных помещеняих, в различное время суток.
  - [ ] Можно воспользоваться открытыми ресурсами с аудиозаписями (см. линк в полезных материалах) (опционально).
  
Примерный план работ по теории на ближайшее время:  
- [x] Освежить в памяти матстатистику  
- [x] Освежить в памяти теорию цифровой обработки сигналов  
- [x] Читать о машинном обучении в целом  
- [x] Освежить в памяти меры близости n-мерных векторов
- [x] Познакомиться с сиамскими сетями

Примерный план работ по презентации исследований:
- [x] Выступить на [АрМНТК](https://miem.hse.ru/armntk/)
- [x] Выступить на [LXII МНК](https://ssau.ru/events/1127-lxxi-molodezhnaya-nauchnaya-konferentsiya-posvyashchennaya-60-letiyu-poleta-v-kosmos-yua-gagarina)
- [ ] Выступить на [ПИТ-2021](https://ssau.ru/events/1137-mezhdunarodnaya-nauchno-tekhnicheskaya-konferentsiya-perspektivnye-informatsionnye-tekhnologii-pit-2021)
- [ ] Защитить диплом 
- [ ] Опубликоваться в [каком-нибудь](https://www.springer.com/journal/12005) [журнале](http://www.computeroptics.ru/) (опционально)
- [ ] Выступить на [ИТНТ-2022](http://itnt-conf.org/index.php)

Полезные библиотеки:
  * Обработка аудио:
    * [pysndfx](https://pypi.org/project/pysndfx/) - обертка над [sox](http://sox.sourceforge.net/), инструкция по установке sox [тут](https://stackoverflow.com/questions/17667491/how-to-use-sox-in-windows)
    * [librosa](https://librosa.org/doc/latest/index.html) - удобно строить спектрограммы
    * [soundfile](https://pysoundfile.readthedocs.io/en/latest/) - удобная работа с файлами
    * [scipy.signal](https://docs.scipy.org/doc/scipy/reference/signal.html#module-scipy.signal) - удобное конструирование ЛИС-систем
    * [numpy.fft](https://numpy.org/doc/stable/reference/routines.fft.html) - удобная работа с преобразованием Фурье
  * Визуализация:
    * [pandas](https://pandas.pydata.org/docs/)
    * [matplotlib](https://matplotlib.org/3.3.1/contents.html)
  * Машинное обучение:
    * [scikit-learn](https://scikit-learn.org/stable/)
    * [keras](https://keras.io/)
    * [tensorflow](https://www.tensorflow.org/)

Полезные материалы:
 * [Введение в сиамские сети](https://towardsdatascience.com/a-friendly-introduction-to-siamese-networks-85ab17522942)
 * [Введение в сиамские сети](https://towardsdatascience.com/one-shot-learning-with-siamese-networks-using-keras-17f34e75bb3d)
 * [Сравнение изображений сиамскими сетями](https://www.pyimagesearch.com/2020/12/07/comparing-images-for-similarity-using-siamese-networks-keras-and-tensorflow/)
 * [Сравнение изображений сиамскими сетями](https://www.researchgate.net/publication/320055318_Image_similarity_using_Deep_CNN_and_Curriculum_Learning)
 * [Сравнение аудиозаписей сиамскими сетями](https://towardsdatascience.com/calculating-audio-song-similarity-using-siamese-neural-networks-62730e8f3e3d)
 * [Аудиозаписи шумов в свободном доступе](https://annotator.freesound.org/fsd/explore/%252Fm%252F093_4n/)
