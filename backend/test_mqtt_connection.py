
import os
import threading
from pathlib import Path

import paho.mqtt.client as mqtt
from dotenv import load_dotenv

# Load .env from the project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

HOST = os.getenv("MQTT_BROKER_HOST")
PORT = int(os.getenv("MQTT_BROKER_PORT", "8883"))
USERNAME = os.getenv("MQTT_USERNAME")
PASSWORD = os.getenv("MQTT_PASSWORD")
KEEPALIVE = int(os.getenv("MQTT_KEEPALIVE", "60"))

# Check that required settings exist
if not all([HOST, USERNAME, PASSWORD]):
    raise ValueError(
        "Missing MQTT settings. Check MQTT_BROKER_HOST, "
        "MQTT_USERNAME and MQTT_PASSWORD in your .env file."
    )

connected_event = threading.Event()
connection_error = None


def on_connect(client, userdata, flags, reason_code, properties):
    global connection_error

    if reason_code.is_failure:
        connection_error = str(reason_code)
        print("MQTT connection failed:", reason_code)
    else:
        print("Successfully connected to HiveMQ Cloud!")
        connected_event.set()


client = mqtt.Client(
    callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
    client_id="stresssense-connection-test"
)

client.username_pw_set(USERNAME, PASSWORD)

# Enable TLS encryption
client.tls_set()

client.on_connect = on_connect

try:
    print("Connecting to HiveMQ Cloud...")
    client.connect(HOST, PORT, KEEPALIVE)
    client.loop_start()

    if connected_event.wait(timeout=10):
        print("CONNECTION TEST PASSED")
    elif connection_error:
        print("CONNECTION TEST FAILED:", connection_error)
    else:
        print("CONNECTION TEST FAILED: Timed out.")

except Exception as error:
    print("CONNECTION TEST FAILED:", error)

finally:
    client.loop_stop()
    client.disconnect()
