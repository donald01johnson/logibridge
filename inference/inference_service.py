#!/usr/bin/env python3
"""Dockerised LogiBridge TFLite MQTT inference service."""

import json
import logging
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import paho.mqtt.client as mqtt
import tensorflow as tf

from preprocessing import (
    FEATURE_NAMES,
    SensorSample,
    SlidingWindowProcessor,
    load_training_statistics,
)


LOGGER = logging.getLogger(
    "logibridge.inference"
)

CLASS_NAMES = [
    "Normal",
    "Warning",
    "Critical",
]


class TFLiteModel:
    """Load and invoke a float32 TFLite classifier."""

    def __init__(self, model_path):
        self.model_path = Path(
            model_path
        )

        if not self.model_path.exists():
            raise FileNotFoundError(
                "Model not found: "
                + str(self.model_path)
            )

        self.interpreter = tf.lite.Interpreter(
            model_path=str(
                self.model_path
            )
        )

        self.interpreter.allocate_tensors()

        self.input_details = (
            self.interpreter
            .get_input_details()
        )

        self.output_details = (
            self.interpreter
            .get_output_details()
        )

        if list(
            self.input_details[0]["shape"]
        ) != [1, 6]:
            raise ValueError(
                "Expected model input shape [1, 6]"
            )

        if list(
            self.output_details[0]["shape"]
        ) != [1, 3]:
            raise ValueError(
                "Expected model output shape [1, 3]"
            )

    def predict(self, features):
        """Return class probabilities and predicted class."""

        input_value = np.asarray(
            features,
            dtype=np.float32,
        ).reshape(1, 6)

        self.interpreter.set_tensor(
            self.input_details[0]["index"],
            input_value,
        )

        start_time = time.perf_counter()

        self.interpreter.invoke()

        latency_ms = (
            time.perf_counter()
            - start_time
        ) * 1000.0

        probabilities = (
            self.interpreter.get_tensor(
                self.output_details[0][
                    "index"
                ]
            )[0]
        )

        predicted_label = int(
            np.argmax(probabilities)
        )

        confidence = float(
            probabilities[
                predicted_label
            ]
        )

        return {
            "predicted_label": predicted_label,
            "predicted_class": (
                CLASS_NAMES[
                    predicted_label
                ]
            ),
            "confidence": confidence,
            "probabilities": [
                float(value)
                for value in probabilities
            ],
            "latency_ms": float(
                latency_ms
            ),
        }


class InferenceService:
    """MQTT preprocessing and TFLite inference service."""

    def __init__(self):
        self.broker_host = os.getenv(
            "MQTT_HOST",
            "host.docker.internal",
        )

        self.broker_port = int(
            os.getenv(
                "MQTT_PORT",
                "1883",
            )
        )

        self.truck_id = os.getenv(
            "TRUCK_ID",
            "TRUCK-001",
        )

        self.mqtt_client_id = os.getenv(
            "MQTT_CLIENT_ID",
            (
                "logibridge-inference-"
                + self.truck_id
            ),
        )

        self.model_path = os.getenv(
            "MODEL_PATH",
            "/app/model.tflite",
        )

        self.statistics_path = os.getenv(
            "STATS_PATH",
            "/app/training_stats.npy",
        )

        self.qos = int(
            os.getenv(
                "MQTT_QOS",
                "1",
            )
        )

        self.input_topic = (
            "logibridge/trucks/"
            + self.truck_id
            + "/sensors/combined"
        )

        self.output_topic = (
            "logibridge/trucks/"
            + self.truck_id
            + "/inference"
        )

        self.model = TFLiteModel(
            self.model_path
        )

        statistics = (
            load_training_statistics(
                self.statistics_path
            )
        )

        self.processor = (
            SlidingWindowProcessor(
                statistics=statistics
            )
        )

        self.client = mqtt.Client(
            callback_api_version=(
                mqtt.CallbackAPIVersion.VERSION2
            ),
            client_id=self.mqtt_client_id,
            protocol=mqtt.MQTTv311,
        )

        self.client.on_connect = (
            self.on_connect
        )

        self.client.on_disconnect = (
            self.on_disconnect
        )

        self.client.on_message = (
            self.on_message
        )

        self.running = True
        self.connected = False
        self.inference_sequence = 0
        self.last_vibration_value = None

    def on_connect(
        self,
        client,
        userdata,
        flags,
        reason_code,
        properties,
    ):
        del userdata
        del flags
        del properties

        if reason_code != 0:
            LOGGER.error(
                "MQTT connection failed: %s",
                reason_code,
            )
            return

        self.connected = True

        client.subscribe(
            self.input_topic,
            qos=self.qos,
        )

        LOGGER.info(
            "Subscribed to %s",
            self.input_topic,
        )

        LOGGER.info(
            "Publishing inference to %s",
            self.output_topic,
        )

    def on_disconnect(
        self,
        client,
        userdata,
        disconnect_flags,
        reason_code,
        properties,
    ):
        del client
        del userdata
        del disconnect_flags
        del properties

        self.connected = False

        if self.running:
            LOGGER.warning(
                "MQTT disconnected: %s",
                reason_code,
            )

    def vibration_is_new(self, value):
        """Detect a new 0.5 Hz vibration observation."""

        numeric_value = float(value)

        if self.last_vibration_value is None:
            self.last_vibration_value = (
                numeric_value
            )
            return True

        changed = not np.isclose(
            numeric_value,
            self.last_vibration_value,
            rtol=0.0,
            atol=1e-12,
        )

        if changed:
            self.last_vibration_value = (
                numeric_value
            )

        return bool(changed)

    def on_message(
        self,
        client,
        userdata,
        message,
    ):
        del client
        del userdata

        try:
            payload = json.loads(
                message.payload.decode(
                    "utf-8"
                )
            )

            timestamp_text = str(
                payload["timestamp"]
            )

            timestamp = (
                datetime.fromisoformat(
                    timestamp_text.replace(
                        "Z",
                        "+00:00",
                    )
                ).timestamp()
            )

            vibration_value = float(
                payload[
                    "vibration_rms_g"
                ]
            )

            vibration_updated = payload.get(
                "vibration_updated"
            )

            if vibration_updated is None:
                vibration_updated = (
                    self.vibration_is_new(
                        vibration_value
                    )
                )

            sample = SensorSample(
                timestamp=float(
                    timestamp
                ),
                temperature_c=float(
                    payload[
                        "temperature_c"
                    ]
                ),
                vibration_rms_g=(
                    vibration_value
                ),
                door_state=str(
                    payload.get(
                        "door_state",
                        "CLOSE",
                    )
                ),
                vibration_updated=bool(
                    vibration_updated
                ),
            )

            windows = (
                self.processor.add_sample(
                    sample
                )
            )

            for window in windows:
                self.publish_inference(
                    window
                )

        except Exception:
            LOGGER.exception(
                "Rejected sensor message"
            )

    def publish_inference(self, window):
        """Run inference and publish one result."""

        normalized_features = window[
            "normalized_features"
        ]

        if normalized_features is None:
            raise RuntimeError(
                "Normalised features unavailable"
            )

        prediction = self.model.predict(
            normalized_features
        )

        self.inference_sequence += 1

        result = {
            "schema_version": "1.0",
            "truck_id": self.truck_id,
            "timestamp": datetime.now(
                timezone.utc
            ).isoformat(
                timespec="milliseconds"
            ).replace(
                "+00:00",
                "Z",
            ),
            "inference_sequence": (
                self.inference_sequence
            ),
            "window_start_unix": float(
                window[
                    "window_start"
                ]
            ),
            "window_end_unix": float(
                window[
                    "window_end"
                ]
            ),
            "sample_count": int(
                window[
                    "sample_count"
                ]
            ),
            "feature_names": (
                FEATURE_NAMES
            ),
            "raw_features": [
                float(value)
                for value in window[
                    "raw_features"
                ]
            ],
            "normalized_features": [
                float(value)
                for value in normalized_features
            ],
            "model_path": (
                self.model_path
            ),
            "predicted_label": (
                prediction[
                    "predicted_label"
                ]
            ),
            "predicted_class": (
                prediction[
                    "predicted_class"
                ]
            ),
            "confidence": prediction[
                "confidence"
            ],
            "probabilities": prediction[
                "probabilities"
            ],
            "latency_ms": prediction[
                "latency_ms"
            ],
        }

        publish_result = (
            self.client.publish(
                self.output_topic,
                json.dumps(
                    result,
                    separators=(",", ":"),
                ),
                qos=self.qos,
                retain=False,
            )
        )

        if (
            publish_result.rc
            != mqtt.MQTT_ERR_SUCCESS
        ):
            raise RuntimeError(
                "Inference publication failed"
            )

        LOGGER.info(
            "INFERENCE %03d | class=%s | "
            "confidence=%.4f | latency=%.3f ms",
            self.inference_sequence,
            prediction[
                "predicted_class"
            ],
            prediction["confidence"],
            prediction["latency_ms"],
        )

    def stop(self):
        self.running = False

    def run(self):
        """Connect and run until interrupted."""

        LOGGER.info(
            "Loading model from %s",
            self.model_path,
        )

        LOGGER.info(
            "Loading statistics from %s",
            self.statistics_path,
        )

        LOGGER.info(
            "Connecting to MQTT broker %s:%d",
            self.broker_host,
            self.broker_port,
        )

        LOGGER.info(
            "MQTT client ID: %s",
            self.mqtt_client_id,
        )

        self.client.connect(
            self.broker_host,
            self.broker_port,
            keepalive=60,
        )

        self.client.loop_start()

        deadline = (
            time.monotonic() + 15.0
        )

        while not self.connected:
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "MQTT connection timed out"
                )

            time.sleep(0.05)

        try:
            while self.running:
                time.sleep(0.25)
        finally:
            self.running = False
            self.client.loop_stop()
            self.client.disconnect()

            LOGGER.info(
                "Inference service stopped"
            )


def main():
    """Program entry point."""

    logging.basicConfig(
        level=os.getenv(
            "LOG_LEVEL",
            "INFO",
        ),
        format=(
            "%(asctime)s | %(levelname)-8s | "
            "%(name)s | %(message)s"
        ),
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    service = InferenceService()

    def stop_service(
        signal_number,
        frame,
    ):
        del frame

        LOGGER.info(
            "Received signal %s",
            signal_number,
        )

        service.stop()

    signal.signal(
        signal.SIGINT,
        stop_service,
    )

    signal.signal(
        signal.SIGTERM,
        stop_service,
    )

    service.run()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
