# Audio Authenticity Identification by Background Noise Comparison
НИР **"Сравнение условий получения аудиозаписей на основе анализа фоновых шумов"**  

Общая постановка задачи:  
*В рамках криминалистической экспертизы решается задача установления подлиности частей аудиозаписи на основе анализа фоновых шумов. Фоновый шум аудиозаписи должен иметь одинаковые параметры по всей фонограмме. Если фоновый шум какого-то участка аудиозаписи отличается от остальных, участок считается вставкой.*

Постановка задачи на данный момент:  
*Имеются две аудиозаписи фоновых шумов, необходимо установить, идентичны ли их фоновые шумы.*
 
Примерный план работ  по коду на ближайшее время:
- [ ] Набрать начальный датасет фоновых шумов в разных локациях на разные устройства.
- [x] Исследовать статистические методы:
  - [x] Реализовать простейшие статистические методы сравнения двух фрагментов фоновых шумов:
    - [x] Критерий хи-квадрат Пирсона,
    - [x] Критерий Стьюдента,
    - [x] Критерий Манна-Уитни-Вилкоксона.
  - [x] Определить пороги значений величины расхождения критериев, при которых можно считать, что шумы различны (для начала можно эвристически). 
  - [x] Реализовать метод на основе голосования по трем статистическим критериям:
    - [x] Каждый критерий равноправен,
    - [x] Взвешенное голосование (для начала веса можно задать эвристически).
- [x] Начать подключать примитивное машинное обучение:    
  - [x] Для начала визуализировать результаты работы статметодов - набор трехмерных точек в осях (U пирсон, U стьюдент, U манн).
  - [x] Реализовать логистическую регрессию на визуализированном наборе признаков.
- [ ] Начать подключать машинное обучение покруче (*тут пока туман войны, но точно будем возиться со спектрограммами и преобразованием Фурье*).

Примерный план работ по теории на ближайшее время:  
- [ ] Освежить в памяти матстатистику  
- [ ] Освежить в памяти теорию цифровой обработки сигналов  
- [ ] Читать о машинном обучении в целом  
- [ ] Искать статьи/репозитории по тематике исследования

Какие библиотеки могут оказаться полезными:
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
