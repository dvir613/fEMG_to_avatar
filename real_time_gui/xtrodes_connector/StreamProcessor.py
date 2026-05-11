from xtrodes_connector.StreamedPacket import CommunicationConstants, Record
import ParsedResult
from xtrodes_connector.StreamedPacket import StreamPacket


class StreamProcessor:
    last_sequence_number = 0
    lost_packets =0
    total_number_of_lost_packets=0
    temp_main_buffer=bytearray()
    def __init__(self):
        self.buffer = bytearray()

    def process(self, data):
        self.buffer.extend(data)
        packets = self._extract_packets()
        self.buffer.clear()
        return packets

    def _extract_packets(self):
        # Implement your packet extraction logic here
        # For simplicity, let's assume packets end with a newline character
        packets = []
        #packet = self.buffer[:delimiter_index + 1]
        #self.buffer = self.buffer[delimiter_index + 1:]

        # Assume packet is complete and can be parsed
        # Calling parse_buffer directly here with the packet
        #self.parse_buffer(packet, len(packet))  # Adjust if pars

        if(len(self.buffer) >=1024):
            parsed_results = parse_buffer(self.buffer,len(self.buffer))
            return parsed_results


        # while b'\n' in self.buffer:
        #     delimiter_index = self.buffer.index(b'\n')
        #     packet = self.buffer[:delimiter_index + 1]
        #     self.buffer = self.buffer[delimiter_index + 1:]
        #     packets.append(packet)
        # return packets


# class CommunicationConstants:
#     LOW_LEVEL_STR = 'SomeString'  # Adjust this to the appropriate constant value


def parse_buffer(main_bt_buffer, main_bt_buffer_pointer_to_next_empty):
    #str_data = main_bt_buffer[:main_bt_buffer_pointer_to_next_empty].decode('utf-8')

    #index_of_start_record = str_data.find(CommunicationConstants.LOW_LEVEL_STR)
    if 0xd in main_bt_buffer:
        parsed_results = get_parsed_results(main_bt_buffer, 0xd)
        return parsed_results
        # for parsed_result in parsed_results:
        #     logging.debug(f"type: {parsed_result.command_type}")
        #     sequence_number = parsed_result.parsed_result_bytes[1:2]  # slicing to get one byte
        #
        #     parse_stream_packet_into_records(parsed_result)
        #     logging.debug(f"record number: {sequence_number[0]}")

    #return 0


class RESPONSE_COMMANDS_TYPES:
    RESPONSE = 1
    REPORT = 2
    REPLY = 3


def find_next_index_as_the_start_of_buffer(bytes_received, localframeoccurrence,end_of_packet_index):
    return end_of_packet_index+1


def get_parsed_results(bytes_received, frame_to_check):
    start_index_to_look_for_record = 0
    parsed_results = []
    # str_main_bt_buffer = bytes_received.decode('ascii')

    #localframeoccurrences = all_indexes_of(str_main_bt_buffer, frame_to_check)
    bytes_received = StreamProcessor.temp_main_buffer+bytes_received
    localframeoccurrences = [i for i, byte in enumerate(bytes_received) if byte == frame_to_check]

    if not localframeoccurrences:
        StreamProcessor.temp_main_buffer=bytearray()
        return parsed_results

    for index, localframeoccurrence in enumerate(localframeoccurrences):
        if localframeoccurrence<start_index_to_look_for_record:#if the current occurence is in an areas which is not relevant anymore, need to continue
            continue
        if localframeoccurrence + CommunicationConstants.BLUETOOTH_PACKET_FIRST_INDEX_OF_LENGTH < len(bytes_received):
            length_bytes = bytes_received[localframeoccurrence + CommunicationConstants.BLUETOOTH_PACKET_FIRST_INDEX_OF_LENGTH:
                                          localframeoccurrence + CommunicationConstants.BLUETOOTH_PACKET_FIRST_INDEX_OF_LENGTH + 2]
            length = int.from_bytes(length_bytes, 'little')  # Adjust byte order if necessary

            if length <= CommunicationConstants.BLUETOOTH_PACKET_LENGTH_WITHOUT_HEADER:
                if localframeoccurrence + length < len(bytes_received):
                    if bytes_received[localframeoccurrence + CommunicationConstants.BLUETOOTH_PACKET_FIRST_INDEX_OF_LENGTH + 1 + length] == CommunicationConstants.LOW_LEVEL_PROTOCOL_RESPONSE_END_FRAME[0]:
                        start_index_to_look_for_record = localframeoccurrence + length + CommunicationConstants.BLUETOOTH_PACKET_LENGTH_HEADER

                        msg_type = bytes_received[localframeoccurrence + CommunicationConstants.BLUETOOTH_PACKET_MESSAGE_TYPE_INDEX]
                        if msg_type == RESPONSE_COMMANDS_TYPES.RESPONSE:
                            parsed_results.append(create_parsed_result(bytes_received, localframeoccurrence,
                                                                       localframeoccurrence + length + CommunicationConstants.BLUETOOTH_PACKET_FIRST_INDEX_OF_LENGTH+1, RESPONSE_COMMANDS_TYPES.RESPONSE))
                        elif msg_type == RESPONSE_COMMANDS_TYPES.REPORT:
                            parsed_results.append(create_parsed_result(bytes_received, localframeoccurrence,
                                                                       localframeoccurrence + length + CommunicationConstants.BLUETOOTH_PACKET_FIRST_INDEX_OF_LENGTH+1, RESPONSE_COMMANDS_TYPES.REPORT))
                        else:  # Handling REPLY and other types
                            parsed_results.append(create_parsed_result(bytes_received, localframeoccurrence,
                                                                       localframeoccurrence + length + CommunicationConstants.BLUETOOTH_PACKET_FIRST_INDEX_OF_LENGTH+1, RESPONSE_COMMANDS_TYPES.REPLY))
                        #next line will find the next index to look for more data
                        start_index_to_look_for_record=find_next_index_as_the_start_of_buffer(bytes_received, localframeoccurrence,localframeoccurrence + length + CommunicationConstants.BLUETOOTH_PACKET_FIRST_INDEX_OF_LENGTH+1)
                else:#here means the cancidate lenght provided exceeds the current byte received length
                    logging.debug("NO END OF FRAME: ")
                    start_index_to_look_for_record = localframeoccurrence
                    break#if there are not enough bytes in the buffer and the lenght candiate is less than 1024 better to wait for more data to come
            else:#mean lenght cancidate exceeds 1024 bytes long packet
                logging.debug(f"Received length not valid: {length}")
                start_index_to_look_for_record = localframeoccurrence

    # Adjust main buffer, simulation of Buffer.BlockCopy for a reset operation
    StreamProcessor.temp_main_buffer = bytes_received[start_index_to_look_for_record:]
    return parsed_results#//returns a list of parsed result which are simply packet with raw data

def all_indexes_of(source_str, substring):
    """ Utility function to find all occurrences of a substring """
    start = 0
    while True:
        start = source_str.find(substring, start)
        if start == -1: return
        yield start
        start += len(substring)  # use start += 1 to find overlapping matches

def create_parsed_result(buffer, start, end, command_type):
    """ Simulated function to create a parsed result """
    # Implement this function according to your actual parsing needs
    parsed_results = ParsedResult.ParsedResult(buffer[start:end+1], command_type)

    return parsed_results
    #return {'data': buffer[start:end], 'type': command_type}








import logging

def parse_stream_packet_into_records(parsed_result):
    number_of_records = parsed_result.parsed_result_bytes[CommunicationConstants.BLUETOOTH_PACKET_NUMBER_OF_RECORDS_FIELD:CommunicationConstants.BLUETOOTH_PACKET_NUMBER_OF_RECORDS_FIELD + 1]
    streamed_packet = StreamPacket(parsed_result)
    from xtrodes_connector.Record import parse_payload_into_recordings
    streamed_packet.records = parse_payload_into_recordings(parsed_result, None)
    if StreamProcessor.last_sequence_number==0:
        StreamProcessor.last_sequence_number = streamed_packet.sequence_number
        return streamed_packet
    if streamed_packet.sequence_number - StreamProcessor.last_sequence_number <0:
        StreamProcessor.lost_packets = StreamProcessor.lost_packets + streamed_packet.sequence_number + 2**8-1 - StreamProcessor.last_sequence_number
        StreamProcessor.total_number_of_lost_packets = StreamProcessor.total_number_of_lost_packets+ streamed_packet.sequence_number + 2**8-1 - StreamProcessor.last_sequence_number
    else:
        StreamProcessor.lost_packets = StreamProcessor.lost_packets + streamed_packet.sequence_number  - StreamProcessor.last_sequence_number-1
        StreamProcessor.total_number_of_lost_packets = StreamProcessor.total_number_of_lost_packets+ streamed_packet.sequence_number  - StreamProcessor.last_sequence_number-1
    StreamProcessor.last_sequence_number = streamed_packet.sequence_number
   # os.system('clear')
    logging.debug("Number of lost packet: " + str(StreamProcessor.total_number_of_lost_packets))

    # if is_save_file:
    #     Record.save_records(records_headers.records, file_handler)
    #     Record.save_records_to_csv(records_headers.records, csv_handlers_holder)

    # if is_real_time_streaming():
    #     if check_bt_streaming_packet(parsed_result):
    #         send_bt_streaming_packet_to_tcp(parsed_result)
    return streamed_packet
    if streamed_packet.records.data_records is not None:
        pass
        #process_records_data(records_headers)
from xtrodes_connector.Record import Record
def process_records_data(records_headers):
    is_error_in_metric = False
    records_message = ""
    for record in records_headers.records.data_records:
        if record.record_type == Record.RECORD_TYPE_ELECTRODES1P5:
            delta = abs(record.packet_index - last_a2_record_index) - 1 if abs(record.packet_index - last_a2_record_index) > 1 else 0
            number_of_lost_a2_records += delta
            is_error_in_metric = is_error_in_metric or delta > 0
            last_a2_record_index = record.packet_index
        elif record.record_type == Record.RECORD_TYPE_ELECTRODES_SAMPLING:
            delta = abs(record.packet_index - last_a0_record_index) - 1 if abs(record.packet_index - last_a0_record_index) > 1 else 0
            number_of_lost_a0_records += delta
            is_error_in_metric = is_error_in_metric or delta > 0
            last_a0_record_index = record.packet_index
        elif record.record_type == Record.RECORD_TYPE_IMU_SAMPLING:
            delta = abs(record.packet_index - last_a1_record_index) - 1 if abs(record.packet_index - last_a1_record_index) > 1 else 0
            number_of_lost_a1_records += delta
            is_error_in_metric = is_error_in_metric or delta > 0
            last_a1_record_index = record.packet_index

    if records_headers.number_of_records >= 9:
        records_message += f"Large # of records: {records_headers.number_of_records}"
        logging.debug("ADC records indexes: " + records_message)

def is_real_time_streaming():
    # Implement the real-time streaming check logic
    return True

def check_bt_streaming_packet(parsed_result):
    # Implement logic to check if this is a viable Stream packet
    return True

def send_bt_streaming_packet_to_tcp(parsed_result):
    # Implement logic to send packet to TCP
    pass

def update_metric(is_error, message):
    # Implement the logic to update metric with message
    pass
