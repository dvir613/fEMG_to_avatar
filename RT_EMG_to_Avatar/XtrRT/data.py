import os
import ctypes
import socket
import warnings
import pyedflib
import subprocess
import numpy as np
from typing import Union
from threading import Thread
from itertools import groupby
from datetime import datetime, timedelta

from .record import parse_byte_arr, \
    EXPECTED_SAMPLES_PER_RECORD, PHYS_MAX_EXG, PHYS_MAX_ACC, PHYS_MAX_GYR, EXG_BITS, IMU_BITS


class ConnectionTimeoutError(ConnectionRefusedError):

    def __init__(self,
                 message="\nMake sure \"Bluetooth Low Energy C# sample\" is running and connected. Then do this and run again:"
                         "\n  1. Copy: CheckNetIsolation.exe LoopbackExempt -is -p=S-1-15-2-2022722280-4131399851-3337013219-4054732753-2439233258-3605005605-669734301"
                         "\n  2. Open PowerShell as Administrator"
                         "\n  3. Right click inside the PowerShell window to paste; hit Enter. (Keep this window open!)"
                 ):
        self.message = message

    def __str__(self):
        return self.message

    def __repr__(self):
        return str(type(self))


class Data(Thread):

    VALID_EXTENSIONS = ['.edf']

    def __init__(self,
                 host_name: str,
                 port: int,
                 timeout_secs: float = None,
                 verbose: bool = False,
                 save_as: str = None):

        # Make sure `save_as` file path is valid
        if isinstance(save_as, str):
            fp, ext = os.path.splitext(save_as)
            if ext.lower() not in Data.VALID_EXTENSIONS:
                warnings.warn(f"Invalid extension {ext}. Saving as {Data.VALID_EXTENSIONS[0]} instead.")
                fp = fp + Data.VALID_EXTENSIONS[0]
            dir_ = os.path.dirname(fp)
            os.makedirs(dir_, exist_ok=True) if dir_ else None
        elif save_as is None:
            pass
        else:
            type_ = type(save_as)
            raise ValueError(f"Invalid input type {type_} for parameter `save_as` (must be str)")

        # Initialize thread
        Thread.__init__(self)

        # Initialize properties
        self.has_data = False
        self.exg_data = None
        self.imu_data = None
        self.fs_exg = None
        self.fs_imu = None
        self.is_connected = False
        self.start_time = None
        self.annotations = []
        self.save_as = save_as
        self._client = None
        self._verbose = verbose
        self._n_bytes = 1024
        self._timeout_secs = timeout_secs
        self._current_packet_exg = (0, 0)
        self._current_packet_imu = (0, 0)

        # Initialize client
        self._init_client(host_name, port)

    def _init_client(self, host_name: str, port: int):
        """
        Initialize socket connection

        :param host_name: IP address or proxy URL
        :param port: computer's port number to use to connect
        :return:
        """

        try:
            is_admin = os.getuid() == 0
        except AttributeError:
            is_admin = ctypes.windll.shell32.IsUserAnAdmin() != 0

        if is_admin:
            # Run background process
            cmd = "CheckNetIsolation.exe LoopbackExempt -is -p=S-1-15-2-2022722280-4131399851-3337013219-4054732753-2439233258-3605005605-669734301"
            subprocess.Popen(cmd)

        # create an INET, STREAMing socket
        self._client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._client.setblocking(False)
        self._client.settimeout(self._timeout_secs)
        ip = socket.gethostbyname(host_name)
        # now connect to the web server on port `port` -- normal http port = 80
        try:
            self._client.connect((ip, port))
            self.is_connected = True
            print(f"Connected to streamer {ip} via port {port}")
        except ConnectionRefusedError:
            raise ConnectionTimeoutError
        except:
            # What else might raise errors?
            raise ConnectionTimeoutError

    @staticmethod
    def _print_records(records: list):

        info = {key: np.sum(r.data.shape[0] for r in result) for key, result in
                groupby([r for r in records if r.record_type == "EXG"], key=lambda rec: rec.packet_idx)}
        [print(f'Record #{k}: {v} new EXG samples') for k, v in info.items()]
        info = {key: np.sum(r.data.shape[0] for r in result) for key, result in
                groupby([r for r in records if r.record_type != "EXG"], key=lambda rec: rec.packet_idx)}
        [print(f'Record #{k}: {v} new IMU samples') for k, v in info.items()]

    def _parse_incoming_records(self):

        # Receive the next self._n_bytes bytes of data from socket
        try:
            byte_arr = self._client.recv(self._n_bytes)
        except ConnectionResetError as e:
            self.is_connected = False
            self.save_data()
            raise e
        except socket.timeout as e:
            self.is_connected = False
            self.save_data()
            raise socket.timeout(f"More than {self._timeout_secs} seconds have passed without receiving any data.")

        # Parse most recent packet into list of Records
        records = parse_byte_arr(byte_arr)

        # Print packet details if verbose==True
        if self._verbose:
            Data._print_records(records)

        return records

    def run(self):
        """
        Overrides Thread.run().

        Commands what to do when a Data thread is started: continuously receive data from the socket, parse it into
        Records, (print details of received packet if verbose==True), and add data to growing data matrix.
        """

        while self.is_connected:

            # Receive incoming record(s)
            records = self._parse_incoming_records()

            # Add newly received data to main data matrix
            self._add_to_data(records)

        print('Connection terminated.')
        self.save_data()

    def save_data(self):

        if self.save_as is None:
            return

        fp, ext = os.path.splitext(self.save_as)
        success = False
        if ext.lower() == '.edf':

            success = self._write_edf()
            print(f"Data saved to {self.save_as}")

        if not success:
            warnings.warn("Failed to save data.")

    def _make_edf_header(self,
                         technician: str = '',
                         recording_additional: str = '',
                         patientname: str = '',
                         patient_additional: str = '',
                         patientcode: str = '',
                         equipment: str = '',
                         admincode: str = '',
                         gender: str = '',
                         birthdate: Union[datetime, str] = ''):

        startdate = self.start_time
        annotations = self.annotations

        assert startdate is None or isinstance(startdate, datetime), 'must be datetime or None, is {}: {}'.format(type(startdate), startdate)
        assert birthdate == '' or isinstance(birthdate, (datetime, str)), 'must be datetime or empty, is {}'.format(type(birthdate))
        if startdate is None:
            now = datetime.now()
            startdate = datetime(now.year, now.month, now.day, now.hour, now.minute, now.second)
            del now
        if isinstance(birthdate, datetime): birthdate = birthdate.strftime('%d %b %Y')
        local = {k: v for k, v in locals().items() if k != 'self'}
        header = {}
        for var in local:
            if isinstance(local[var], datetime):
                header[var] = local[var]
            else:
                header[var] = str(local[var])
        return header

    def _preprocess_edf_signals(self):

        # Add data to EDF-writer object
        channels = []
        headers = []
        for dataset, data, fs in zip(("EXG", "IMU"),
                                     (self.exg_data, self.imu_data),
                                     (self.fs_exg, self.fs_imu)):
            if data is not None:

                # Add data to list of channels to write (must be list and not 2D matrix in case multiple sampling rates)
                n_samples, n_channels = data.shape
                data[np.isnan(data)] = 0  # replace NaNs with 0
                channels.extend([ch for ch in data.T])

                labels = [f'Channel {nch}' for nch in range(n_channels)] if dataset == "EXG" else \
                         ["Acc X", "Acc Y", "Acc Z", "Gyro X", "Gyro Y", "Gyro Z"]

                # If there are n_bits of data, then signed data can range from -2^n_bits/2 to 2^n_bits/2-1
                # 2^n_bits/2 = 2^(n_bits-1)
                n_bits = EXG_BITS if dataset == 'EXG' else IMU_BITS if dataset == 'IMU' else 0
                digital_max = int(2 ** (n_bits-1) - 1)
                digital_min = int(- 2 ** (n_bits-1))

                for nch in range(n_channels):

                    label = labels[nch]
                    if dataset == 'EXG':
                        units = 'uV'
                        physical_max = PHYS_MAX_EXG
                    elif 'Acc' in label:
                        units = 'g'
                        physical_max = PHYS_MAX_ACC
                    elif 'Gyro' in label:
                        units = 'deg/sec'
                        physical_max = PHYS_MAX_GYR
                    else:
                        raise ValueError("Unknown data type. Failed to save data.")

                    physical_min = -physical_max

                    header = {
                        'label':  label,                # channel label (string, <= 16 characters, must be unique)
                        'dimension':  units,            # physical dimension (e.g., mV) (string, <= 8 characters)
                        'sample_rate':  fs,             # sample frequency in hertz (int). Deprecated: use 'sample_frequency' instead.
                        'sample_frequency':  fs,        # number of samples per record (int)
                        'physical_max':  physical_max,  # maximum physical value (float)
                        'physical_min':  physical_min,  # minimum physical value (float)
                        'digital_max':  digital_max,    # maximum digital value (int, -2**n_bits <= x < 2**n_bits)
                        'digital_min':  digital_min,    # minimum digital value (int, -2**n_bits <= x < 2**n_bits)
                    }
                    headers.append(header)

        return channels, headers

    def _write_edf(self):
        """
        Write an edf file using pyEDFlib

        :param signal_headers: a list with one signal header for each signal (see function `make_signal_header)
        :param is_digital: whether signals are presented digitally or in physical values
        """

        filepath = self.save_as
        header = self._make_edf_header()
        signals, signal_headers = self._preprocess_edf_signals()

        if not signals:
            print('No data to save.')
            return False

        assert header is None or isinstance(header, dict), 'header must be dictioniary'
        assert isinstance(signal_headers, list), 'signal headers must be list'
        assert len(signal_headers) == len(signals), 'signals and signal_headers must be same length'

        n_channels = len(signals)

        with pyedflib.EdfWriter(filepath, n_channels=n_channels) as edf:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                edf.setSignalHeaders(signal_headers)
                edf.setHeader(header)
                edf.writeSamples(signals, digital=False)  # digital = False if physical values; True if int
                for time, dur, txt in self.annotations:
                    edf.writeAnnotation(time, dur, txt)

        SUCCESS = os.path.isfile(filepath) and os.path.getsize(filepath) > min([len(sig) for sig in signals])
        return SUCCESS

    def _insert_stray_packet(self, record):

        if record.record_type == "EXG":
            fs = self.fs_exg
            current_packet_idx, current_packet_len = self._current_packet_exg
        elif record.record_type == "IMU":
            fs = self.fs_imu
            current_packet_idx, current_packet_len = self._current_packet_imu
        else:
            return

        # Account for packet index reset after 2**16
        record_packet_idx = record.packet_idx
        if current_packet_idx - record.packet_idx > 2 ** 15:
            record_packet_idx += 2 ** 16

        last_sample_idx = current_packet_idx * EXPECTED_SAMPLES_PER_RECORD[fs][record.record_type] + current_packet_len
        incoming_sample_idx = record_packet_idx * EXPECTED_SAMPLES_PER_RECORD[fs][record.record_type]

        # Missed packet
        if current_packet_idx < record_packet_idx:
            missing_samples = incoming_sample_idx - last_sample_idx
            padded_record = np.pad(record.data, ((missing_samples, 0), (0, 0)), constant_values=np.nan)
            if record.record_type == "EXG":
                self.exg_data = np.vstack((self.exg_data, padded_record))
            else:  # record.record_type in ("IMU", "Acc", "Gyro"):
                self.imu_data = np.vstack((self.imu_data, padded_record))

        # Filling missed packet (this never actually happens)
        elif current_packet_idx >= record_packet_idx:
            # TODO: what if the 2nd half of a partial packet is inserted? How to know which half it is?
            samples_ago = last_sample_idx - incoming_sample_idx
            n_inserted_samples = record.data.shape[0]
            if record.record_type == "EXG":
                insertion_idx = self.exg_data.shape[0] - samples_ago
                # assert all(np.isnan(self.exg_data[insertion_idx:insertion_idx + n_inserted_samples, :]))  # TODO
                self.exg_data[insertion_idx:insertion_idx + n_inserted_samples, :] = record.data
            else:
                insertion_idx = self.imu_data.shape[0] - samples_ago
                # assert all(np.isnan(self.imu_data[insertion_idx:insertion_idx + n_inserted_samples, :]))  # TODO
                self.imu_data[insertion_idx:insertion_idx + n_inserted_samples, :] = record.data

    def _add_to_data(self, records: list):
        """
        Concatenates incoming samples to ever-expanding data matrix.

        (Note: should eventually be updated for improved efficiency...)

        :param records: list of Record objects
        """

        for record in records:

            is_valid = self._validate_record(record)

            if record.record_type == "EXG":

                # No EXG data yet
                if self.exg_data is None:
                    self.exg_data = record.data
                # Everything is as expected; simply append new data to existing data matrix
                elif is_valid:
                    self.exg_data = np.vstack((self.exg_data, record.data))
                # Weird shenanigans...
                else:
                    self._insert_stray_packet(record)

                # Establish/check sampling rate of received data
                if self.fs_exg is None:
                    self.fs_exg = record.fs
                else:
                    assert self.fs_exg == record.fs

            else:  # record.record_type in ("IMU", "Acc", "Gyro")

                # No IMU data yet
                if self.imu_data is None:
                    self.imu_data = record.data
                # Everything is as expected; simply append new data to existing data matrix
                elif is_valid:
                    self.imu_data = np.vstack((self.imu_data, record.data))
                # Weird shenanigans...
                else:
                    self._insert_stray_packet(record)

                # Establish/check sampling rate of received data
                if self.fs_imu is None:
                    self.fs_imu = record.fs
                else:
                    assert self.fs_imu == record.fs

            # Save time of first received record, regardless of type
            if self.start_time is None:
                self.start_time = datetime.fromtimestamp(records[0].unix_time_secs) + \
                                  timedelta(milliseconds=records[0].unix_time_ms)

            if not self.has_data and (self.exg_data is not None and self.imu_data is not None):
                self.has_data = True
                print("Streaming EXG and IMU data...")

    def _update_current_packet_info(self, record_type: str, packet_idx: int, n_samples: int):

        if record_type == "EXG":
            self._current_packet_exg = (packet_idx, n_samples)
        else:  # record_type in ("IMU", "Acc", "Gyro"):
            self._current_packet_imu = (packet_idx, n_samples)

    def _validate_record(self, record):
        """
        Each record should contain EXPECTED_SAMPLES_PER_RECORD samples, but can be broken up across multiple packets, so
        in a given packet, we should receive either a full record with EXPECTED_SAMPLES_PER_RECORD samples, immediately
        preceded by packet_idx-1 and immediately followed by packet_idx+1, OR a record with fewer than
        EXPECTED_SAMPLES_PER_RECORD samples. In the latter case, sequentially received records should have the same
        index and their sum of samples should equal EXPECTED_SAMPLES_PER_RECORD.
        (Aside from this, there are some edge case scenarios described below.)
        """

        is_valid = True  # Default

        # Existing data info
        if record.record_type == "EXG":
            current_packet = self._current_packet_exg
            existing_samples = 0 if self.exg_data is None else self.exg_data.shape[0]
            fs = self.fs_exg
        else:
            current_packet = self._current_packet_imu
            existing_samples = 0 if self.imu_data is None else self.imu_data.shape[0]
            fs = self.fs_imu

        # Incoming data info
        n_samples = record.data.shape[0]
        record_idx = record.packet_idx
        while record_idx + 2**15 < current_packet[0]:
            record_idx += 2**16

        # Usually, numerous packets are already sent from the DAU by the time this app connects to the streamer,
        # so we can ignore an initial gap in indices:
        if current_packet == (0, 0) and not self.has_data:
            self._update_current_packet_info(record.record_type, record_idx, n_samples)

        # Receiving a (full or partial) packet with index 1 greater than the previous packet, which is complete:
        elif record_idx == current_packet[0] + 1 and current_packet[1] == EXPECTED_SAMPLES_PER_RECORD[fs][record.record_type]:
            self._update_current_packet_info(record.record_type, record_idx, n_samples)

        # Receiving a partial packet with same index as prior packet received, whose sum of samples is at most
        # the number expected in a full packet:
        elif record_idx == current_packet[0] and n_samples + current_packet[1] <= EXPECTED_SAMPLES_PER_RECORD[fs][record.record_type]:
            self._update_current_packet_info(record.record_type, record_idx, n_samples + current_packet[1])

        # After max uint16 (=65535), index restarts at 0, so we allow the following:
        elif current_packet == (65535, EXPECTED_SAMPLES_PER_RECORD[fs][record.record_type]) and record_idx == 0:
            self._update_current_packet_info(record.record_type, record_idx, n_samples)

        # Data collection began with a partial packet, so it's OK that a new packet follows:
        elif record_idx == current_packet[0] + 1 and existing_samples < EXPECTED_SAMPLES_PER_RECORD[fs][record.record_type]:
            self._update_current_packet_info(record.record_type, record_idx, n_samples)

        # Something is awry.
        else:

            s_rec = 's' if n_samples > 1 else ''
            s_cur = 's' if current_packet[1] > 1 else ''

            # Attempt to retroactively insert packet that was skipped (never actually happens)
            if current_packet[0] > record_idx:
                warnings.warn(f"Wrongly inserted packet index {record_idx} ({n_samples} sample{s_rec} after packet index {current_packet[0]} ({current_packet[1]} sample{s_cur}).")
                self._update_current_packet_info(record.record_type, record_idx, n_samples)
                is_valid = False

            # Attempting to insert packet after missing packets
            elif current_packet[0] <= record_idx:
                missed_samples = record_idx * EXPECTED_SAMPLES_PER_RECORD[fs][record.record_type] - (current_packet[0] * EXPECTED_SAMPLES_PER_RECORD[fs][record.record_type] + current_packet[1])
                s_miss = 's' if missed_samples > 1 else ''
                warnings.warn(f"Missing {missed_samples} sample{s_miss} between last packet index {current_packet[0]} ({current_packet[1]} sample{s_cur}) and current packet index {record_idx} ({n_samples} sample{s_rec}).")
                self._update_current_packet_info(record.record_type, record_idx, n_samples)
                is_valid = False

            # Shouldn't happen!
            else:
                warnings.warn(f"Unexpected error attempting to insert packet index {record.packet_idx} ({n_samples} samples) after packet index {current_packet[0]} ({current_packet[1]} samples).")
                is_valid = False

        return is_valid

    def add_annotation(self, text: str,
                       time: Union[datetime, float] = None,
                       duration: Union[timedelta, float] = 0):

        if time is None:
            secs_since_start = 0 if self.exg_data is None and self.imu_data is None \
                                 else self.imu_data.shape[0]/self.fs_imu if self.exg_data is None \
                                 else self.exg_data.shape[0]/self.fs_exg
            time = secs_since_start
        elif isinstance(time, (int, float)):
            assert time > 0, "Attempting to insert annotation before data collection began."
        elif isinstance(time, datetime):
            time = self.start_time if time < self.start_time else time
            dif = time - self.start_time
            time = dif.seconds + dif.microseconds/1e6
        else:
            raise ValueError

        if isinstance(duration, (int, float)):
            # ms = duration - int(duration)
            # secs = int(duration)
            # duration = timedelta(seconds=secs, milliseconds=ms)
            assert duration >= 0
        elif isinstance(duration, timedelta):
            duration = duration.seconds + duration.microseconds/1e6
        else:
            raise ValueError

        self.annotations.append((time, duration, text))

    def stop(self):
        self.is_connected = False