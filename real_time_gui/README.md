# xtrodes_connector

The **xtrodes_connector** Python SDK enables seamless communication with Xtroses-powered external systems that stream real-time physiological or sensor data. It hides all low-level protocol, networking, and data handling complexity, exposing a minimal and clean interface.

Designed for integration into real-time visualizations, signal processing, or research environments.

---

## 🔧 Installation

### Install from GitHub (source build)

```bash
pip install git+https://github.com/yourusername/xtrodes_connector.git

## 📦 Usage

To use the SDK, instantiate the `DataHandler` class with the following arguments:

```python
from xtrodes_connector import DataHandler
import queue

# Create a thread-safe queue for streaming data
data_queue = queue.Queue()

# Define IP and port of the external streaming server
ip = "192.168.0.10"
port = 12345

# Instantiate handler with configuration
handler = DataHandler(ip=ip, port=port, data_queue=data_queue)

# Start the data stream
handler.start()

# Read streamed data from the queue (example)
while True:
    data = data_queue.get()
    print("Received:", data)

# Stop the data stream
handler.stop()
