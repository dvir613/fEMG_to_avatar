class CommunicationConstants:
    BLUETOOTH_PACKET_LENGTH_HEADER = 7
    LOW_LEVEL_PROTOCOL_RESPONSE = bytes([0xd])  # start of frame
    LOW_LEVEL_PROTOCOL_RESPONSE_END_FRAME = bytes([0xa])  # end of frame

    LOW_LEVEL_STR = LOW_LEVEL_PROTOCOL_RESPONSE.decode('ascii')
    BAUD_RATE = 9000000
    MAIN_BUFFER_SIZE = 20 * 1024  # 20 KB
    RECEIVED_BUFFER_SIZE = 4096
    MAX_MTU = 2048
    BLUETOOTH_PACKET_LENGTH_WITHOUT_HEADER = 1017
    BLUETOOTH_PACKET_FIRST_INDEX_OF_LENGTH = 5
    BLUETOOTH_PACKET_MESSAGE_TYPE_INDEX = 2
    BLUETOOTH_PACKET_NUMBER_OF_RECORDS_FIELD = 10
    COMMAND_SET_TIME = 0x2

    START_STREAMING_AND_LOGGING = bytes([0x0D, 0xF0, 0x00, 0x01, 0x01, 0x08, 0x00, 0x02, 0x03, 0x00, 0x03, 0x01, 0x03, 0x06, 0x0A])

# Example usage
#print(CommunicationConstants.LOW_LEVEL_STR)
#print(CommunicationConstants.START_STREAMING_AND_LOGGING)




class COMMUNICATION_STATE:
    START_FRAME_SYNC_WAIT = 1
    STOP_FRAME_WAIT = 2


class RESPONSE_COMMANDS_TYPES:
    RESPONSE = 0x02
    REPLY = 0x03
    REPORT = 0x04
    NA = 0xFF


class Record:
    OFFSET_OF_NUMBER_OF_RECORDS = CommunicationConstants.BLUETOOTH_PACKET_NUMBER_OF_RECORDS_FIELD
    # Placeholder for Record class implementation


class StreamPacket:
    def __init__(self, parsed_result):
        block = parsed_result.parsed_result_bytes
        self.start_byte = block[0]
        self.sequence_number = block[1]
        self.message_type = block[2]
        self.message_flags = block[3]
        self.number_of_payloads = block[4]
        self.length = int.from_bytes(block[5:7], 'little')
        self.number_of_records = block[Record.OFFSET_OF_NUMBER_OF_RECORDS]
        self.checksum = block[5 + 2 + self.length - 2]
        self.end_of_message = block[5 + 2 + self.length - 1]
        self.records = None  # Assuming it needs to be filled later


class ParseResult:
    # Placeholder for ParseResult class implementation
    pass
