import sys
from PyQt6.QtCore import Qt
from PyQt6 import QtCore, QtGui
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QApplication, QLabel, QMainWindow, QPushButton, QVBoxLayout, QWidget, QFileDialog, \
    QMessageBox

from work import predict


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(200, 200)
        # Создаем кнопку выбора файла
        self.btn_select_image = QPushButton('Выбрать изображение')
        self.btn_select_image.clicked.connect(self.select_image)

        # Создаем кнопку помощи
        self.btn_help = QPushButton('Помощь')
        self.btn_help.clicked.connect(self.show_help)

        # Создаем виджет для отображения изображения
        self.label = QLabel()

        # Создаем метку для отображения предсказанного класса
        self.predicted_class_label = QLabel()

        # Создаем главный вертикальный layout и добавляем наши виджеты
        layout = QVBoxLayout()
        layout.addWidget(self.btn_select_image)
        layout.addWidget(self.btn_help)
        layout.addWidget(self.label)
        layout.addWidget(self.predicted_class_label)

        # Создаем контейнер-виджет и устанавливаем в него наш layout
        container = QWidget()
        container.setLayout(layout)

        # Устанавливаем наш контейнер-виджет в качестве центрального виджета главного окна
        self.setCentralWidget(container)

        # Текущее изображение
        self.current_image = None

        self.class_values = [
            "1.5 Пересечение с трамвайной линией",
            "1.11.1 Опасный поворот направо",
            "1.11.2 Опасный поворот налево",
            "1.12.2 Опасные повороты с первом поворотом налево",
            "1.15 ну крч машинка юзом идёт",
            "1.16 5 неровная дорога",
            "1.17 лежачий полицейский",
            "1.20.1 сужение дороги с обеих сторон",
            "1.20.2 сужение дороги справа",
            "1.20.3 сужение дороги слева",
            "1.22 пешеходный переход",
            "1.23 Осторожно! Дети",
            "1.25 Дорожные работы",
            "2.3.2 Примыкание к второстепенной дороге справа",
            "2.3.3 Примыкание к второстепенной дороге слева",
            "2.5 Движение без остановки запрещено",
            "3.1 Въезд запрещён",
            "3.2 Движение запрещено",
            "3.4 Движение грузовых автомобилей запрещено",
            "3.13 Ограничение высоты",
            "3.18.1 Поворот направо запрещён",
            "3.18.2 Поворот налево запрещён",
            "3.19 Разворот запрещён",
            "3.24 Ограничение максимальной скорости",
            "3.27 Остановка запрещена",
            "3.28 Стоянка запрещена",
            "3.29 Стоянка запрещена по нечётным числам месяца",
            "3.30 Стоянка запрещена по чётным числам месяца",
            "4.1.1 Движение прямо",
            "4.1.2 Движение направо",
            "4.1.4 Движение прямо и направо",
            "4.1.5 Движение прямо и налево",
            "4.2.1 Объезд препятствия справа",
            "4.2.2 Объезд препятствия слева",
            "4.2.3 Объезд препятствия слева или справа",
            "4.3 Круговое движение",
            "5.19 Пешеходный переход"
        ]

    def select_image(self):
        # Открываем диалог выбора файла
        file_name, _ = QFileDialog.getOpenFileName(self, 'Выбрать изображение', '', 'Image files (*.jpg *.png)')

        if file_name:
            # Если файл выбран, загружаем изображение и устанавливаем его на QLabel
            pixmap = QPixmap(file_name)

            if pixmap.width() > 512 or pixmap.height() > 512:
                pixmap = pixmap.scaled(512, 512, QtCore.Qt.AspectRatioMode.KeepAspectRatio)

            self.label.setPixmap(pixmap)

            # Обновляем значение self.current_image
            self.current_image = file_name

            # Меняем текст на кнопке на название открытого файла
            self.btn_select_image.setText(file_name.split('/')[-1])

            # Вызываем функцию изменения размера приложения, чтобы применить новое изображение
            self.resizeEvent(QtGui.QResizeEvent(self.size(), self.size()))

            # Используем модель для получения предсказаний
            predicted_class = predict(file_name)
            if predicted_class in range(len(self.class_values)):
                # Получаем значение из списка по индексу, соответствующему предсказанному классу
                predicted_value = self.class_values[predicted_class]
                self.predicted_class_label.setText('На картинке: {}'.format(predicted_value))
            else:
                self.predicted_class_label.setText('Predicted class: Unknown')

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.current_image is not None and not self.isMaximized() and not self.isFullScreen():
            pixmap = QPixmap(self.current_image)
            size = self.label.size()
            scaled_pixmap = pixmap.scaled(size, Qt.AspectRatioMode.KeepAspectRatio,
                                          Qt.TransformationMode.SmoothTransformation)
            self.label.setPixmap(scaled_pixmap)

    def show_help(self):
        text = """
        Вас приветствует программа распознавания дорожных знаков!

        Для использования программы, выполните следующие шаги:

        1. Нажмите кнопку "Выбрать изображение" для выбора изображения с дорожным знаком.
        2. Выберите изображение в формате JPG или PNG в диалоговом окне.
        3. После выбора изображения, оно отобразится в главном окне программы.
        4. Программа автоматически распознает класс дорожного знака и выводит его название.
        
        ВАЖНО! ваше изображение должно быть максимально обрезано, что бы знак занимал как можно больше площади

        Спасибо за использование программы!
        """
        QMessageBox.information(self, 'Помощь', text)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
