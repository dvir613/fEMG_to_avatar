import numpy as np

# Hardware constants
N_PHYSICAL_CHANNELS_EXG = 16
N_PHYSICAL_CHANNELS_IMU = 3

# Digital-to-physical conversions
PHYS_MAX_EXG = 12582.912
PHYS_MIN_EXG = -PHYS_MAX_EXG
PHYS_MAX_ACC = 16  # 16g
PHYS_MIN_ACC = -PHYS_MAX_ACC
PHYS_MAX_GYR = 2000  # deg/sec
PHYS_MIN_GYR = -PHYS_MAX_GYR
EXG_BITS = 15
IMU_BITS = 16
DATA_RESOLUTION = {"EXG": (PHYS_MAX_EXG - PHYS_MIN_EXG) / 2**EXG_BITS,
                   "Acc": (PHYS_MAX_ACC - PHYS_MIN_ACC) / 2**IMU_BITS,
                   "Gyro": (PHYS_MAX_GYR - PHYS_MIN_GYR) / 2**IMU_BITS,
                   }
EXPECTED_SAMPLES_PER_RECORD = {250: {"EXG": 7, "IMU": 5, "Acc": 10, "Gyro": 10},
                               500: {"EXG": 15, "IMU": 10, "Acc": 20, "Gyro": 20}}

# BT packet mapping constants
N_BYTES_RECORD_TYPE = 1
N_BYTES_UNIX_TIME_SECS = 4
N_BYTES_UNIX_TIME_MS = 2
N_BYTES_RECORD_LENGTH = 2
N_BYTES_PACKET_IDX = 2
N_BYTES_CHANNEL_MAPPING = 2
N_BYTES_SAMPLE_RATE = 2
N_BYTES_DOWNSAMPLE = 1
N_BYTES_PER_SAMPLE = 2
RECORD_HEADER_LEN = N_BYTES_RECORD_TYPE + N_BYTES_UNIX_TIME_SECS + N_BYTES_UNIX_TIME_MS + N_BYTES_RECORD_LENGTH
RECORD_TYPES = {"EXG": 0xa0,
                "IMU": 0xa1}
RESPONSE_COMMANDS = {"RESPONSE": 0x2,
                     "REPORT": 0x4,
                     "N.A.": 0xff}


class Record:

    def __init__(self,
                 record_type: str = None,
                 data: np.ndarray = None,
                 unix_time_secs: int = None,
                 unix_time_ms: int = None,
                 channel_mapping: int = None,
                 fs: int = None,
                 downsample: int = None,
                 packet_idx: int = None,
                 record_len: int = None):

        self.record_type = record_type
        self.unix_time_secs = unix_time_secs
        self.unix_time_ms = unix_time_ms
        self.channel_mapping = channel_mapping
        self.fs = fs / downsample
        self.record_len = record_len
        self.packet_idx = packet_idx

        if record_type == "EXG":
            self.data = DATA_RESOLUTION["EXG"] * (data - 2**(EXG_BITS-1))
        elif record_type == "Acc":
            self.data = DATA_RESOLUTION["Acc"] * data
        elif record_type == "Gyro":
            self.data = DATA_RESOLUTION["Gyro"] * data
        elif record_type == "IMU":
            data1 = DATA_RESOLUTION["Acc"] * data[:, :3]
            data2 = DATA_RESOLUTION["Gyro"] * data[:, 3:]
            self.data = np.hstack((data1, data2))


def get_nchannels(channel_mapping):
    bit_mask = 0b0001
    n_active = 0
    for n in range(N_PHYSICAL_CHANNELS_EXG):
        if channel_mapping & bit_mask:
            n_active += 1
        bit_mask <<= 1

    return n_active


def parse_byte_arr(byte_arr):

    start_signal = byte_arr[0]  # 0xd
    assert start_signal == 0xd
    sequence_num = byte_arr[1]  # 0-255
    message_type = hex(byte_arr[2])  # RESPONSE_COMMANDS["REPORT"] = (0x4)
    message_flags = byte_arr[3]
    n_payloads = byte_arr[4]
    n_bytes_remaining = byte_arr[5:7]  # should be 1017 if message_type=="REPORT" (0x4)
    n_bytes_remaining = int.from_bytes(list(n_bytes_remaining), 'little')

    header = byte_arr[7]  # 0xf0
    assert header == 0xf0
    data_len = byte_arr[8:10]  # = 1012
    data_len = int.from_bytes(list(data_len), 'little')
    n_records = byte_arr[10]

    header_len = 10
    pointer = header_len + 1
    record_len = 0
    records = []
    for n_record in range(n_records):

        record_type = byte_arr[pointer]  # EXG = 0xa0; IMU = 0xa1
        assert record_type in RECORD_TYPES.values()
        record_type = "EXG" if record_type == RECORD_TYPES["EXG"] else \
                      "IMU" if record_type == RECORD_TYPES["IMU"] else "?"
        pointer += N_BYTES_RECORD_TYPE

        # if record_type != "EXG":
        #     breakpoint()

        unix_time_secs = np.frombuffer(byte_arr[pointer:pointer + N_BYTES_UNIX_TIME_SECS], dtype=np.uint32).item()
        pointer += N_BYTES_UNIX_TIME_SECS

        unix_time_ms = np.frombuffer(byte_arr[pointer:pointer + N_BYTES_UNIX_TIME_MS], dtype=np.uint16).item()
        pointer += N_BYTES_UNIX_TIME_MS

        record_len = np.frombuffer(byte_arr[pointer:pointer + N_BYTES_RECORD_LENGTH], dtype=np.uint16).item()
        pointer += N_BYTES_RECORD_LENGTH

        packet_idx = np.frombuffer(byte_arr[pointer:pointer + N_BYTES_PACKET_IDX], dtype=np.uint16).item()
        pointer += N_BYTES_PACKET_IDX

        channel_mapping = np.frombuffer(byte_arr[pointer:pointer + N_BYTES_CHANNEL_MAPPING], dtype=np.uint16).item()
        pointer += N_BYTES_CHANNEL_MAPPING  # gyro = 1 (0b01), acc = 2 (0b10), both = 3 (0b11)

        record_type = "Gyro" if record_type == "IMU" and channel_mapping == 1 else\
                      "Acc" if record_type == "IMU" and channel_mapping == 2 else \
                      record_type

        sampling_rate = np.frombuffer(byte_arr[pointer:pointer + N_BYTES_SAMPLE_RATE], dtype=np.uint16).item()
        pointer += N_BYTES_SAMPLE_RATE

        downsample = np.frombuffer(byte_arr[pointer:pointer + N_BYTES_DOWNSAMPLE], dtype=np.uint8).item()
        pointer += N_BYTES_DOWNSAMPLE

        n_channels = get_nchannels(channel_mapping)
        n_channels = n_channels if record_type == "EXG" else 3*n_channels  # multiplier for IMU channel mapping
        n_samples_per_channel = (record_len - (RECORD_HEADER_LEN - N_BYTES_RECORD_LENGTH)) / n_channels / N_BYTES_PER_SAMPLE
        assert n_samples_per_channel - int(n_samples_per_channel) == 0  # ensure n_samples is int
        n_samples_per_channel = int(n_samples_per_channel)

        samples = [[] for _ in range(n_channels)]
        for sample_index in range(n_samples_per_channel):
            for channel_index in range(n_channels):
                if record_type == "EXG":

                    # Sanity check that (may be) relevant only for EXG?
                    channel_mask = 1 << channel_index
                    if not (channel_mask & channel_mapping):
                        samples[channel_index].extend([np.nan])
                        continue

                # Read sample, store in `samples`, increment pointer
                dtype = np.dtype(np.int16)
                dtype = dtype if record_type == "EXG" else dtype.newbyteorder('>')  # IMU = big-endian; EXG = little-endian
                sample_buffer = np.frombuffer(byte_arr[pointer:pointer + N_BYTES_PER_SAMPLE], dtype=dtype)
                samples[channel_index].extend(sample_buffer)
                pointer += N_BYTES_PER_SAMPLE

        record = Record(record_type=record_type,
                        data=np.array(samples).T,
                        unix_time_secs=unix_time_secs,
                        unix_time_ms=unix_time_ms,
                        channel_mapping=channel_mapping,
                        fs=sampling_rate,
                        downsample=downsample,
                        packet_idx=packet_idx,
                        record_len=record_len)
        records.append(record)

    return records