class RESPONSE_COMMANDS_TYPES:
    NA = 0  # Assuming NA is a type, add other types as necessary.

class ParsedResult:
    PACKET_LENGTH_INDEX = 5
    BLUETOOTH_HEADER_LENGTH = 5

    def __init__(self,buffer=None,command_type=RESPONSE_COMMANDS_TYPES.NA):
        self.command_type = command_type
        self.parsed_result_bytes = buffer

