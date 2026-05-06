class RecordEvent:
    IN_RECORDING_OFFSET_UNIX_TIME = 1
    IN_RECORDING_OFFSET_UNIX_MS = IN_RECORDING_OFFSET_UNIX_TIME + 4
    IN_RECORDING_OFFSET_OF_RECORD_LENGTH = IN_RECORDING_OFFSET_UNIX_MS + 2
    IN_RECORDING_OFFSET_OF_EVENT_TYPE = IN_RECORDING_OFFSET_OF_RECORD_LENGTH + 2
    IN_RECORDING_OFFSET_OF_EVENT_DATA_LENGTH = IN_RECORDING_OFFSET_OF_EVENT_TYPE + 1
    IN_RECORDING_OFFSET_OF_EVENT_DATA = IN_RECORDING_OFFSET_OF_EVENT_DATA_LENGTH + 2

    EVENT_CHARGER = 0
    EVENT_STATUS = 1

    def __init__(self):
        self.record_type = 0
        self.unix_time_seconds = 0
        self.unix_time_milliseconds = 0
        self.record_length = 0
        self.event_type = 0
        self.event_data_length = 0
        self.event_data = None
        self.record_to_xtr2_save = None
        self._original_record = None

    @staticmethod
    def extract_b0_record(buffer, index_of_record_type, record_type):
        status_record = RecordEvent()
        status_record.record_type = record_type
        status_record.unix_time_seconds = int.from_bytes(buffer[index_of_record_type + RecordEvent.IN_RECORDING_OFFSET_UNIX_TIME:index_of_record_type + RecordEvent.IN_RECORDING_OFFSET_UNIX_TIME + 4], 'little')
        status_record.unix_time_milliseconds = int.from_bytes(buffer[index_of_record_type + RecordEvent.IN_RECORDING_OFFSET_UNIX_MS:index_of_record_type + RecordEvent.IN_RECORDING_OFFSET_UNIX_MS + 2], 'little')
        status_record.record_length = int.from_bytes(buffer[index_of_record_type + RecordEvent.IN_RECORDING_OFFSET_OF_RECORD_LENGTH:index_of_record_type + RecordEvent.IN_RECORDING_OFFSET_OF_RECORD_LENGTH + 2], 'little')
        status_record.event_type = buffer[index_of_record_type + RecordEvent.IN_RECORDING_OFFSET_OF_EVENT_TYPE]
        status_record.event_data_length = int.from_bytes(buffer[index_of_record_type + RecordEvent.IN_RECORDING_OFFSET_OF_EVENT_DATA_LENGTH:index_of_record_type + RecordEvent.IN_RECORDING_OFFSET_OF_EVENT_DATA_LENGTH + 2], 'little')
        start = index_of_record_type + RecordEvent.IN_RECORDING_OFFSET_OF_EVENT_DATA
        end = start + status_record.event_data_length
        status_record.event_data = buffer[start:end]
        status_record._original_record = buffer[index_of_record_type:index_of_record_type + 2 + status_record.record_length + 6]
        return status_record

    def __str__(self):
        event_type_str = "CHARGER" if self.event_type == RecordEvent.EVENT_CHARGER else "STATUS"
        return f"Event {event_type_str} arrived"
