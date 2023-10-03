import cv2 as cv
import numpy as np
import sys

from matplotlib import pyplot as plt
from matplotlib.widgets import Button as btn
from scipy.spatial.distance import cosine
from sklearn.decomposition import PCA
from PyQt5 import QtCore, QtGui, QtWidgets

# -----------------------------------------------------------------------------

# класс отвечающий за интерфейс программы
# где вводятся начальные параметры
class Ui_MainWindow(object):
    # конструктор, где создаются объекты и 
    # задаются размеры и т. п.
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(432, 292)
        MainWindow.setMinimumSize(QtCore.QSize(432, 292))
        MainWindow.setMaximumSize(QtCore.QSize(432, 292))
        MainWindow.setStyleSheet("background-color: lightgray")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(10, 10, 401, 31))
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(10, 50, 241, 31))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(10, 120, 381, 31))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.lineEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit.setGeometry(QtCore.QRect(250, 50, 151, 31))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.lineEdit.setFont(font)
        self.lineEdit.setText("")
        self.lineEdit.setObjectName("lineEdit")
        self.lineEdit_2 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_2.setGeometry(QtCore.QRect(250, 150, 151, 31))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.lineEdit_2.setFont(font)
        self.lineEdit_2.setText("")
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(140, 230, 151, 51))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.pushButton.setFont(font)
        self.pushButton.setObjectName("pushButton")
        # присоединение кнопки к методу button_click
        self.pushButton.clicked.connect(self.button_click)
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(10, 150, 241, 31))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(10, 80, 131, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setGeometry(QtCore.QRect(0, 110, 441, 16))
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.line_2 = QtWidgets.QFrame(self.centralwidget)
        self.line_2.setGeometry(QtCore.QRect(0, 210, 441, 16))
        self.line_2.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(10, 180, 131, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_6.setFont(font)
        self.label_6.setObjectName("label_6")
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        
    # конструктор для задания содержимого,
    # созданного в предыдущем конструкторе
    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Программа"))
        self.label.setText(_translate("MainWindow", "Введите следующие параметры:"))
        self.label_2.setText(_translate("MainWindow", "Количество классов = "))
        self.label_3.setText(_translate("MainWindow", "Количество тестовых изображений"))
        self.pushButton.setText(_translate("MainWindow", "Принять"))
        self.label_4.setText(_translate("MainWindow", "для каждого класса ="))
        self.label_5.setText(_translate("MainWindow", "(не больше 38)"))
        self.label_6.setText(_translate("MainWindow", "(не больше 63)"))
    
    # создание действие по нажатию кнопки
    # которое берет значение введенные пользователем
    # из полей в интерфейсе, и передает в класс для вычислений
    def button_click(self):
        Detection(int(self.lineEdit_2.text()), int(self.lineEdit.text())).show_result()

# -----------------------------------------------------------------------------

# класс изображений, которые загружаются из
# базы данных YaleB. Отсюда изображение возвращается
# либо в стандартном виде, либо в полутоново (сером)
class Image():
    # конструктор класса, куда подается
    # класс и индекс изображения,
    # а также тип возвращаемого изображения
    # т. е. либо обычное, либо серое
    def __init__(self, i, j, tp):
        self.tp = tp # тип изображение
        self.image = cv.imread(f'yaleB/yaleB{i}/{j}.pgm') # обычное изображение
        self.image_gray = cv.imread(f'yaleB/yaleB{i}/{j}.pgm', 0) # полутоновое
        
    # функция дающая доступ к изображение
    # т. е. по заданному тип изображения из
    # любого места программы возвращает
    # изображение
    def get_image(self):
        if self.tp == 'normal':
            return self.image # возвращение обычного изображение
        
        elif self.tp == 'gray':
            return self.image_gray # возвращение полутонового

# -----------------------------------------------------------------------------

# основная часть программы, где происходят все
# вычисления, сравнения, и т. д.
class Detection():
    # конструктор, который принимает такие
    # входные параметры, как кол-во классов (n)
    # и кол-во тестовых изображение (m), после чего
    # генерирует на их основе массивы данных и
    # считывая изображения из БД заносит их в эти
    # массивы. т. е. для эталонов один масив, а для
    # тестового другой
    def __init__(self, n, m):
        self.n = n # кол-во классов
        self.m = m # кол-во тестовых
        self.standards = [] # массив эталонов
        self.tests = [[] for i in range(self.m)] # массив тестовых
        self.get_dataset() # вызов метода для заполнения вышеуказанных массивов
    
    # получение эталонов и тестовых изображений
    def get_dataset(self):
        # получение эталонов
        for i in range(self.m):
            # обращение к классу изображений, для получения изображения
            image = Image(i + 1, 1, 'gray').get_image()
            gamma = self.adjust_gamma(image) # гамма-коррекция изображения
            log = self.log_transform(image) # логарифмирование изображения
            result = np.mean([gamma, log], axis=0) # слияние вышеполученных результатов
            # загрузка полученного изображения в массив эталонов
            self.standards.append(result)
        
        # все то же самое, что сверху,
        # только идет получение тестовых изображений
        for i in range(self.m):
            for j in range(1, self.n + 1):
                image = Image(i + 1, 1, 'gray').get_image()
                gamma = self.adjust_gamma(image)
                log = self.log_transform(image)
                result = np.mean([gamma, log], axis=0)
                self.tests[i].append(result)
    
    # метод, реализующей коррекцию гаммы.
    # он принимает такие входные данные, как
    # само изображение, а также коэффициент гаммы
    def adjust_gamma(self, image, gamma=2):
        invGamma = 1.0 / gamma
        table = [((i / 255) ** invGamma) * 255 for i in range(256)]
        table = np.array(table, np.uint8)
        # это все просто изменение гаммы для полученного изображения
        # и вывод полученного изображения
        return cv.LUT(image, table)
    
    # метод для логарифмирования, который
    # на вход получает изображение
    def log_transform(self, image):
        log = np.uint8(np.log1p(image))
        cv.normalize(log, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
        # это все просто вычисления и вывод
        return log
    
    # метод отвечающий за 2D DCT, а также
    # за вычисление сходства двух изображений в процентах.
    # сходство вычисляется через косинусное расстояние
    # после переводится в проценты.
    # метод принимает два изображение на вход
    def dct(self, image_1, image_2):
        dct1 = cv.dct(np.float32(image_1))
        dct2 = cv.dct(np.float32(image_2))
        sim = 1 - cosine(dct1.flatten(), dct2.flatten())
        # тут берется DCT от каждого из двух полученных
        # изображений, после чего между полученными данными
        # вычисляется косинусное расстояние и
        # выводится в процентах
        return sim * 100
    
    # метод отвечающий за PCA
    # принимает на вход одно изображение
    def pca(self, image):
        # задается количество компонентво
        # которое будет браться из изображения.
        # их кол-во не должно превышать разрешение
        # изображения. если кратко, то метод берет
        # гланвые компоненты (основную информацию)
        # из изображения и выводит их
        comp = PCA(n_components=168)
        image_pca = comp.fit_transform(image)

        return image_pca
    
    # метод для сравнение каждого
    # тестового изображения с эталоно,
    # после чего получается массив с результатами
    # где к каждому тестовому изображению
    # соотносится эталон (класс), с которым
    # он больше всего имеет сходство
    def compare_images(self):
        # массив результатов
        self.result = [[] for i in range(self.m)]
        
        # цикл для сравнения
        for i in range(self.m):
            for j in range(len(self.tests[0])):
                test = self.tests[i][j]
                compared = []
                
                for k in range(self.m):
                    # идет вычисление сначала PCA,
                    # потом DCT и косинусное расстояние
                    dct = self.dct(self.pca(self.standards[k]), 
                                   self.pca(test))
                   
                    compared.append(dct)
                
                # результат заносится в массив
                self.result[i].append([j, np.argmax(compared)])
    
    # метод вычисления точности
    # распознования. т. е. правильно ли
    # программа распазнала класс тестового
    # изображения. просто берется тестовое изображение
    # из полученных результатов, после смотрется какой
    # класс оно имело, и какой класс получило после 
    # распознования. если классы совпадаются, значит
    # распозналось верно
    def get_accuracy(self):
        # массив для записи точности
        self.accuracy = [[] for i in range(self.m)]
        
        trues = 0 # кол-во правильно распознанных
        alls = 0 # кол-во всех изображений
        
        # цикл для подсчета точности
        for i in range(self.m):
            for j in range(len(self.tests[0])):
                alls += 1
    
                if self.result[i][j][1] == i:
                    trues += 1
                    
                # запись результата
                self.accuracy[i].append((trues / alls) * 100)
    
    # метод для вывода результата
    # в новом окне
    def show_result(self):
        self.compare_images() # вызов метода для сравнения эталонов и тестовых
        self.get_accuracy() # вызов метода для вычисления точности распознования
        
        # флаг, типо переключателя для кнопки
        # которая останавливает цикл показа изображений
        self.flag = True
        
        # метод для переключение
        # этого самого переключателя
        def stop(event):
            self.flag = not self.flag
        
        # создание окна с результатами
        # и задание размеров (в скобках)
        fig = plt.figure('Результат', figsize=(16, 8))
        
        # добавление в фигуру мест
        # где будут выводится результатов.
        # в скобках задается их расположение
        ax1 = fig.add_subplot(2, 2, 1) # для тестового изображение
        ax2 = fig.add_subplot(2, 2, 2) # для эталона
        ax3 = fig.add_subplot(2, 1, 2) # для график
        
        # кнопка для остановки цикла
        ax4 = plt.axes([0.4625, 0.05, 0.1, 0.05])
        button = btn(ax4, 'Остановить', color='gray')
        button.on_clicked(stop)
        
        # массивы для динамического 
        # вывода точности 
        y = [] # ось y
        x = [] # ось х
        
        k = 0 # переменная для циклического вывода результатов
        # циклы для вывода результатов
        for i in range(self.m):
            for j in range(len(self.tests[0])):
                y.append(self.accuracy[i][j])
                x.append(k + 1)
                
                # вывод тестовых изображений
                ax1.cla()
                ax1.imshow(Image(i + 1, self.result[i][j][0] + 2, 
                                 'gray').get_image(), cmap='gray')
                ax1.set_xticks([])
                ax1.set_yticks([])
                ax1.set_title('Тестовое изображение')
                ax1.set_xlabel('Класс = ' + str(i + 1))
                
                # вывод эталонов
                ax2.cla()
                ax2.imshow(Image(self.result[i][j][1] + 1, 1,
                                 'gray').get_image(), cmap='gray')
                ax2.set_xticks([])
                ax2.set_yticks([])
                ax2.set_title('Эталон')
                ax2.set_xlabel('Класс = ' + str(self.result[i][j][1] + 1))
                
                # построение графика
                ax3.cla()
                ax3.plot(x, y)
                ax3.set_xlabel('Кол-во изображений')
                ax3.set_ylabel('Точность (%)')
                ax3.set_title('Точность распознавания')
                ax3.set_yticks(np.arange(0, 110, 10))
                
                plt.subplots_adjust(wspace=0.3, hspace=0.3,
                                    top=0.95, bottom=0.2)
                plt.show()
                
                # задания скорости в скобках
                # для выводя изобажений, т. е. 
                # с какой скоростью они будут меняться
                plt.pause(1)
                
                k += 1
                
                # переключатели для кнопки остановить
                if not self.flag:                  
                    break
            
            if not self.flag:                  
                break

# -----------------------------------------------------------------------------

# вызов интерфейса класса
app = QtWidgets.QApplication(sys.argv)
MainWindow = QtWidgets.QMainWindow()
ui = Ui_MainWindow()
ui.setupUi(MainWindow)
MainWindow.show()
sys.exit(app.exec_())