import socket

# False для отключения вывода информации
DEBUG = True


# Здесь вызываются функции из других файлов, возвращается data - массив управления (размера 4)
# Возвращаемый массив состоит из строк, символ разделения целой и дробной части - ","
# message - массив размера 6
# message[0, 3] - x, y, z - координаты дрона на текущем шаге (y - вверх)
# message[0, 3] - x, y, z - координаты поворота дрона
# сейчас в функции get_data находится пример
def get_data(message):
    data = list(map(float, message.replace(',', '.').split()))
    if data[1] < 100:
        rs = ['5', '5', '5,01', '5,01']
    else:
        rs = ['0', '0', '0,01', '0,01']
    return rs


def main():
    # Создаем сокет
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Получаем IP и порт сервера
    server_ip = '127.0.0.1'
    server_port = 12435

    # Подключаем сокет к указанному адресу и порту
    server_socket.bind((server_ip, server_port))
    server_socket.listen(1)

    if DEBUG:
        print("Ждем подключения клиента...")

    while True:
        # Принимаем подключение
        client_socket, addr = server_socket.accept()
        if DEBUG:
            print("Подключен клиент: ", addr)

        # Получаем данные от клиента
        data = client_socket.recv(1024)
        message = data.decode('utf-8')
        if DEBUG:
            print("Получено от клиента: ", message)

        # Получение массива управления
        data = get_data(message)

        # Отправляем ответ клиенту
        response = ' '.join(data)
        client_socket.send(response.encode('utf-8'))
        client_socket.close()


if __name__ == '__main__':
    main()
