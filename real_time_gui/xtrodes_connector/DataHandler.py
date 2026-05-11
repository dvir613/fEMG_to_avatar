import logging

logging.basicConfig(level=logging.WARNING)
import threading
from xtrodes_connector.ConnectionManager import ConnectionManager  # Adjust import path as needed
from xtrodes_connector.StreamProcessor import StreamProcessor, parse_stream_packet_into_records  # Adjust import path as needed

class DataHandler:
    def __init__(self, host, port,queue):
        self.connection_manager = ConnectionManager(host, port,15)
        self.queue = queue
        self.stream_processor = StreamProcessor()
        self.running = threading.Event()
        self.running.clear()
        self.running_thread = None

    def start(self):
        self.running.set()
        self.connection_manager.connect()  # Starts connection and begins filling the queue
        self.running_thread = threading.Thread(target=self.process_data)
        self.running_thread.start()

    def process_data(self):
        while self.running.is_set():
            try:
                # Assuming your ConnectionManager has a method like get_data() to fetch from the queue
                # if self.connection_manager.is_connected==False:
                #     self.connection_manager.connect()
                data = self.connection_manager.get_data()
                if data:
                    packets = self.stream_processor.process(data)
                    for packet in packets:
                        packet_with_records = self.handle_packet(packet)  # Implement packet handling
                        if self.queue.full():
                            discarded = self.queue.get_nowait()
                            print(f"Discarded oldest data: {discarded}")
                        self.queue.put(packet_with_records)

            except Exception as e:
                print(f"Error processing data: {e}")

    def stop(self):
        self.running.clear()
        #self.thread.join()  # Wait for the thread to finish

        self.connection_manager.disconnect()  # Stops the connection and data reception
        self.running_thread.join()

    def handle_packet(self, packet):
        # Handle your packets here
        # for parsed_result in parsed_results:
        logging.debug(f"type: {packet.command_type}")
        sequence_number = packet.parsed_result_bytes[1:2]  # slicing to get one byte

        streamed_packet_with_records = parse_stream_packet_into_records(packet)
        #streamed_packet_with_records.sequence_number
        logging.debug(f"packet number: {streamed_packet_with_records.sequence_number}")
        return streamed_packet_with_records
        #print(f"Packet received: {packet}")
