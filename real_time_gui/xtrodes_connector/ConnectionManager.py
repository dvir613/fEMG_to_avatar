import socket
import threading
import time
from queue import Queue, Empty

class ConnectionManager:
    def __init__(self, host: str, port: int, timeout: float = 15):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.socket = None
        self.is_connected = False
        self.listen_thread = None
        self.data_queue = Queue()
        self.stop_listening = threading.Event()
        self.data_timeout = 10  # seconds to wait for data before considering a reconnection

    def connect(self):
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(self.timeout)
            self.socket.connect((self.host, self.port))
            self.is_connected = True
            print(f"Connection established with {self.host}:{self.port}")
            self.start_listening()
        except socket.error as e:
            print(f"Failed to connect: {e}")
            self.is_connected = False

    def disconnect(self):
        if self.socket:
            self.stop_listening.set()
            if self.listen_thread is not None:
                self.listen_thread.join()
            self.socket.close()
            self.is_connected = False
            print("Connection closed.")

    def only_socket_disconnect(self):
        if self.socket:
            self.stop_listening.set()
            self.socket.close()
            self.is_connected = False
            print("Connection closed.")

    def start_listening(self):
        """
        Starts a separate thread to listen for incoming data.
        """
        self.stop_listening.clear()
        self.listen_thread = threading.Thread(target=self.listen_for_data, daemon=True)
        self.listen_thread.start()

    def reconnect(self):
        self.only_socket_disconnect()
        while not self.is_connected:
            self.connect()
            if not self.is_connected:
                print("Reconnection attempt failed. Retrying in 5 seconds...")
                time.sleep(5)
        print("Reconnected successfully.")

    def listen_for_data(self):
        """
        The method run by the listening thread to continuously receive data.
        """
        last_data_time = time.time()
        while not self.stop_listening.is_set():
            try:
                data = self.socket.recv(1024)
                if data:
                    self.data_queue.put(data)
                    last_data_time = time.time()  # Reset the timer on receiving data
                else:
                    # No data received, connection might be closed
                    print("No data received. Connection might be closed.")
                    if time.time() - last_data_time > self.data_timeout:
                        print("No data received for too long. Reconnecting...")
                        #self.disconnect()
                        self.reconnect()
                        last_data_time = time.time()  # Reset the timer after reconnection
            except socket.timeout:
                self.stop_listening.set()
                #self.disconnect()
                self.reconnect()
                return
            except socket.error as e:
                print(f"Error receiving data: {e}")
                self.stop_listening.set()
                #self.disconnect()
                self.reconnect()
                return

    def get_data(self, timeout=1):
        """
        Attempts to get data from the queue within the specified timeout.

        :param timeout: Timeout in seconds to wait for data in the queue.
        :return: The data received or None if no data is available within the timeout.
        """
        try:
            return self.data_queue.get(timeout=timeout)
        except Empty:
            return None

    def send_data(self, data):
        if not self.is_connected:
            print("Not connected to any server.")
            return
        try:
            self.socket.sendall(data)
        except socket.error as e:
            print(f"Failed to send data: {e}")
            self.reconnect()
            self.socket.sendall(data)  # Try to send data again after reconnecting

    def is_alive(self) -> bool:
        return self.is_connected and self.listen_thread.is_alive()
