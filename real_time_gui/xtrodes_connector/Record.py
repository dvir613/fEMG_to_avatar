
import datetime
from RecordHolder import RecordsHolder
from StreamedPacket import CommunicationConstants, RESPONSE_COMMANDS_TYPES


class Record:
    NUMBER_OF_CHANNELS = 16
    NUMBER_OF_BYTES_PER_0XA2 = 3
    OFFSET_OF_COMMAND = 7
    OFFSET_OF_LENGTH_OF_PAYLOAD = OFFSET_OF_COMMAND + 1
    OFFSET_OF_NUMBER_OF_RECORDS = OFFSET_OF_LENGTH_OF_PAYLOAD + 2
    IN_RECORDING_OFFSET_UNIX_TIME = 1
    IN_RECORDING_OFFSET_UNIX_MS = IN_RECORDING_OFFSET_UNIX_TIME + 4
    IN_RECORDING_OFFSET_OF_RECORD_LENGTH = IN_RECORDING_OFFSET_UNIX_MS + 2
    IN_RECORDING_OFFSET_OF_PACKET_INDEX = IN_RECORDING_OFFSET_OF_RECORD_LENGTH + 2
    IN_RECORDING_OFFSET_OF_CHANNEL_MAPPING = IN_RECORDING_OFFSET_OF_PACKET_INDEX + 2
    IN_RECORDING_OFFSET_OF_SAMPLE_RATE = IN_RECORDING_OFFSET_OF_CHANNEL_MAPPING + 2
    IN_RECORDING_OFFSET_DOWNSAMPLE = IN_RECORDING_OFFSET_OF_SAMPLE_RATE + 2
    IN_RECORDING_OFFSET_OF_DATA = IN_RECORDING_OFFSET_DOWNSAMPLE + 1
    RECORD_TYPE_ELECTRODES_SAMPLING = 0xa0
    RECORD_TYPE_ELECTRODES1P5 = 0xa2
    RECORD_TYPE_IMU_SAMPLING = 0xa1
    RECORD_TYPE_EVENT = 0xb0

    def __init__(self):
        self.record_type = 0
        self.unix_time_seconds = 0
        self.unix_time_milliseconds = 0
        self.record_length = 0
        self.packet_index = 0
        self.channel_mapping = 0
        self.sampling_rate = 0
        self.down_sample = 0
        self.data_samples = None
        self.data_samples_a2 = None
        self.record_to_xtr2_save = None
        self._original_record = None

    @property
    def original_record(self):
        return self._original_record

    @original_record.setter
    def original_record(self, value):
        self._original_record = value
        self.create_record_for_xtr(value)

    def create_record_for_xtr(self, original_array):
        record_for_xtr2 = bytearray(len(original_array) + 3)
        record_for_xtr2[0] = 0xD
        record_for_xtr2[1:-3] = original_array
        record_for_xtr2[-3:-1] = [0x0, 0x0]  # CRC placeholders
        record_for_xtr2[-1] = 0xA
        self.record_to_xtr2_save = bytes(record_for_xtr2)

    @staticmethod
    def parse_payload_into_recordings(parsed_result, mapping_array_helper):
        buffer = parsed_result.parsed_result_bytes
        # Process buffer as needed; this will involve conversion of specific .NET methods to Python
        pass

    # Additional methods (e.g., handle_reply, handle_set_time_response) need similar translation

    @staticmethod
    def concatenate(records):
        total_length = sum(len(record.record_to_xtr2_save) for record in records)
        result = bytearray(total_length)
        offset = 0
        for record in records:
            result[offset:offset+len(record.record_to_xtr2_save)] = record.record_to_xtr2_save
            offset += len(record.record_to_xtr2_save)
        return bytes(result)

    # More methods to translate based on what they do in C#

# Additional helper methods and classes as required


import struct

import struct

def extract_a2_electrodes(buffer, index_of_record_type):
    electrodes_recording = Record()
    electrodes_recording.record_type = buffer[index_of_record_type]
    electrodes_recording.unix_time_seconds = struct.unpack_from('<I', buffer, index_of_record_type + Record.IN_RECORDING_OFFSET_UNIX_TIME)[0]
    electrodes_recording.unix_time_milliseconds = struct.unpack_from('<H', buffer, index_of_record_type + Record.IN_RECORDING_OFFSET_UNIX_MS)[0]
    electrodes_recording.record_length = struct.unpack_from('<H', buffer, index_of_record_type + Record.IN_RECORDING_OFFSET_OF_RECORD_LENGTH)[0]
    electrodes_recording.packet_index = struct.unpack_from('<H', buffer, index_of_record_type + Record.IN_RECORDING_OFFSET_OF_PACKET_INDEX)[0]
    electrodes_recording.channel_mapping = struct.unpack_from('<H', buffer, index_of_record_type + Record.IN_RECORDING_OFFSET_OF_CHANNEL_MAPPING)[0]
    electrodes_recording.sampling_rate = struct.unpack_from('<H', buffer, index_of_record_type + Record.IN_RECORDING_OFFSET_OF_SAMPLE_RATE)[0]
    electrodes_recording.down_sample = buffer[index_of_record_type + Record.IN_RECORDING_OFFSET_DOWNSAMPLE]

    number_of_channels_sent = return_number_of_channels_sent(electrodes_recording.channel_mapping)
    number_of_samples_per_channel = (electrodes_recording.record_length - 7) // Record.NUMBER_OF_BYTES_PER_0XA2 // number_of_channels_sent
    samples = [[0] * number_of_samples_per_channel for _ in range(16)]

    loop_data_pointer = 0
    for sample_index in range(number_of_samples_per_channel):
        for channel_index in range(16):
            if (1 << channel_index) & electrodes_recording.channel_mapping:
                sample_value = struct.unpack_from('<I', buffer, index_of_record_type + Record.IN_RECORDING_OFFSET_OF_DATA + loop_data_pointer)[0] & 0x00FFFFFF
                if sample_value & 0x800000:
                    sample_value -= 0x01000000  # Subtract 2^24 to make it negative in 32-bit
                    #sample_value |= 0xFF000000  # Sign extend to 32 bits
                samples[channel_index][sample_index] = sample_value
                loop_data_pointer += Record.NUMBER_OF_BYTES_PER_0XA2

    electrodes_recording.data_samples_a2 = samples
    record_end = index_of_record_type + 2 + electrodes_recording.record_length + 1 + 6
    electrodes_recording.original_record = buffer[index_of_record_type:record_end]

    return electrodes_recording, record_end


import struct


def extract_imu_record(buffer, index_of_record_type):
    imu_recording = Record()

    imu_recording.record_type = buffer[index_of_record_type]
    imu_recording.unix_time_seconds = \
    struct.unpack_from('<I', buffer, index_of_record_type + Record.IN_RECORDING_OFFSET_UNIX_TIME)[0]
    imu_recording.unix_time_milliseconds = \
    struct.unpack_from('<H', buffer, index_of_record_type + Record.IN_RECORDING_OFFSET_UNIX_MS)[0]
    imu_recording.record_length = \
    struct.unpack_from('<H', buffer, index_of_record_type + Record.IN_RECORDING_OFFSET_OF_RECORD_LENGTH)[0]
    imu_recording.packet_index = \
    struct.unpack_from('<H', buffer, index_of_record_type + Record.IN_RECORDING_OFFSET_OF_PACKET_INDEX)[0]
    imu_recording.channel_mapping = \
    struct.unpack_from('<H', buffer, index_of_record_type + Record.IN_RECORDING_OFFSET_OF_CHANNEL_MAPPING)[0]
    imu_recording.sampling_rate = \
    struct.unpack_from('<H', buffer, index_of_record_type + Record.IN_RECORDING_OFFSET_OF_SAMPLE_RATE)[0]
    imu_recording.down_sample = buffer[index_of_record_type + Record.IN_RECORDING_OFFSET_DOWNSAMPLE]

    number_of_channels_sent = return_number_of_channels_sent(imu_recording.channel_mapping)
    number_of_samples_per_channel = (imu_recording.record_length - 7) // 2 // number_of_channels_sent
    imu_recording.data_samples = [[0] * number_of_samples_per_channel for _ in range(number_of_channels_sent)]

    loop_data_pointer = 0
    for sample_index in range(number_of_samples_per_channel):
        for channel_index in range(number_of_channels_sent):
            sample_offset = index_of_record_type + Record.IN_RECORDING_OFFSET_OF_DATA + loop_data_pointer
            imu_recording.data_samples[channel_index][sample_index] = struct.unpack_from('>h', buffer, sample_offset)[0]
            loop_data_pointer += 2

    end_of_record = index_of_record_type + 2 + imu_recording.record_length + 1 + 6
    imu_recording.original_record = buffer[index_of_record_type:end_of_record]

    return imu_recording, end_of_record


def extract_a0_electrodes(buffer, index_of_record_type):
    electrodes_recording = Record()
    electrodes_recording.record_type = buffer[index_of_record_type]
    electrodes_recording.unix_time_seconds = struct.unpack_from('<I', buffer, index_of_record_type + Record.IN_RECORDING_OFFSET_UNIX_TIME)[0]
    electrodes_recording.unix_time_milliseconds = struct.unpack_from('<H', buffer, index_of_record_type + Record.IN_RECORDING_OFFSET_UNIX_MS)[0]
    electrodes_recording.record_length = struct.unpack_from('<H', buffer, index_of_record_type + Record.IN_RECORDING_OFFSET_OF_RECORD_LENGTH)[0]
    electrodes_recording.packet_index = struct.unpack_from('<H', buffer, index_of_record_type + Record.IN_RECORDING_OFFSET_OF_PACKET_INDEX)[0]
    electrodes_recording.channel_mapping = struct.unpack_from('<H', buffer, index_of_record_type + Record.IN_RECORDING_OFFSET_OF_CHANNEL_MAPPING)[0]
    electrodes_recording.sampling_rate = struct.unpack_from('<H', buffer, index_of_record_type + Record.IN_RECORDING_OFFSET_OF_SAMPLE_RATE)[0]
    electrodes_recording.down_sample = buffer[index_of_record_type + Record.IN_RECORDING_OFFSET_DOWNSAMPLE]

    number_of_channels_sent = return_number_of_channels_sent(electrodes_recording.channel_mapping)
    number_of_samples_per_channel = (electrodes_recording.record_length - 7) // 2 // number_of_channels_sent
    samples = [[0] * number_of_samples_per_channel for _ in range(number_of_channels_sent)]

    loop_data_pointer = 0
    for sample_index in range(number_of_samples_per_channel):
        for channel_index in range(number_of_channels_sent):
            if (1 << channel_index) & electrodes_recording.channel_mapping:
                sample_value = struct.unpack_from('<h', buffer, index_of_record_type + Record.IN_RECORDING_OFFSET_OF_DATA + loop_data_pointer)[0]
                samples[channel_index][sample_index] = sample_value
                loop_data_pointer += 2

    electrodes_recording.data_samples = samples
    record_end = index_of_record_type + 2 + electrodes_recording.record_length + 1 + 6
    electrodes_recording.original_record = buffer[index_of_record_type:record_end]

    return electrodes_recording, record_end

def return_number_of_channels_sent(channel_mapping):
    return bin(channel_mapping).count('1')
from RecordEvent import RecordEvent

def parse_payload_into_recordings(parsed_result, mapping_array_helper):
    buffer = parsed_result.parsed_result_bytes
    data = RecordsHolder()

    message_type_index =  CommunicationConstants.BLUETOOTH_PACKET_MESSAGE_TYPE_INDEX

    if buffer[message_type_index] == RESPONSE_COMMANDS_TYPES.REPORT:
        if buffer[Record.OFFSET_OF_COMMAND] != 0xF0:
            print("parse payload: not a streaming packet")
            return None

        number_of_records = buffer[Record.OFFSET_OF_NUMBER_OF_RECORDS]
        index_of_record_type = Record.OFFSET_OF_NUMBER_OF_RECORDS + 1

        for index in range(number_of_records):
            record_type = buffer[index_of_record_type]
            if record_type == Record.RECORD_TYPE_ELECTRODES_SAMPLING or record_type == (
                    Record.RECORD_TYPE_ELECTRODES_SAMPLING | 0x8):
                electrodes_recording, index_of_record_type = extract_a0_electrodes(buffer, index_of_record_type)
                data.data_records.append(electrodes_recording)
            elif record_type == Record.RECORD_TYPE_IMU_SAMPLING:
                imu_recording, index_of_record_type = extract_imu_record(buffer, index_of_record_type)
                data.data_records.append(imu_recording)
            elif record_type == Record.RECORD_TYPE_ELECTRODES1P5 or record_type == (
                    Record.RECORD_TYPE_ELECTRODES1P5 | 0x8):
                a2_recording, index_of_record_type = extract_a2_electrodes(buffer, index_of_record_type)
                data.data_records.append(a2_recording)
            elif record_type == Record.RECORD_TYPE_EVENT:
                event_record, index_of_record_type = RecordEvent.extract_b0_record(buffer, index_of_record_type)
                data.event_records.append(event_record)

    elif buffer[message_type_index] == RESPONSE_COMMANDS_TYPES.RESPONSE:
        pass
        #handle_set_time_response(buffer)
    elif buffer[message_type_index] == RESPONSE_COMMANDS_TYPES.REPLY:
        #reply = handle_reply(buffer)
        #data.reply_messages.append(reply)
        pass
   # current_sequence_number = parsed_result.
    return data

# Additional methods need to be defined to support this:
# - extract_a0_electrodes
# - extract_imu_sampling
# - extract_a2_electrodes
# - extract_event_record
# - handle_set_time_response
# - handle_reply
