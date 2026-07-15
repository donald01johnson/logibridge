#!/usr/bin/env python3
"""Monitor LogiEdge output confidence using rolling-window PSI."""

import argparse
import csv
import json
import logging
import signal
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import paho.mqtt.client as mqtt

from psi_common import (
    PSI_BIN_LABELS,
    calculate_psi,
    confidence_distribution,
)


LOGGER = logging.getLogger(
    "logibridge.drift_monitor"
)

ROLLING_WINDOW_SIZE = 100
MONITOR_INTERVAL_SECONDS = 60.0
DRIFT_ALERT_THRESHOLD = 0.25
RECOVERY_THRESHOLD = 0.10


class DriftMonitor:
    """Subscribe to inference results and compute rolling PSI."""

    def __init__(
        self,
        broker_host,
        broker_port,
        truck_id,
        reference_path,
        output_csv,
    ):
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.truck_id = truck_id

        self.reference_path = Path(
            reference_path
        )

        self.output_csv = Path(
            output_csv
        )

        self.reference = self.load_reference()

        self.reference_proportions = (
            np.asarray(
                self.reference[
                    "proportions"
                ],
                dtype=np.float64,
            )
        )

        self.inference_topic = (
            "logibridge/trucks/"
            + self.truck_id
            + "/inference"
        )

        self.scores = deque(
            maxlen=ROLLING_WINDOW_SIZE
        )

        self.lock = threading.Lock()

        self.running = True
        self.connected = False
        self.alert_active = False

        self.last_evaluation_time = (
            time.monotonic()
        )

        self.client = mqtt.Client(
            callback_api_version=(
                mqtt.CallbackAPIVersion.VERSION2
            ),
            client_id=(
                "logibridge-drift-monitor-"
                + self.truck_id
            ),
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

        self.prepare_csv()

    def load_reference(self):
        """Load and validate reference_dist.json."""

        if not self.reference_path.exists():
            raise FileNotFoundError(
                str(self.reference_path)
            )

        reference = json.loads(
            self.reference_path.read_text(
                encoding="utf-8"
            )
        )

        if reference[
            "reference_window_count"
        ] != 300:
            raise ValueError(
                "Reference must contain 300 windows"
            )

        if reference["bins"] != [
            0.0,
            0.25,
            0.5,
            0.75,
            1.0,
        ]:
            raise ValueError(
                "Reference bins are incorrect"
            )

        if len(
            reference["proportions"]
        ) != 4:
            raise ValueError(
                "Reference must contain four proportions"
            )

        return reference

    def prepare_csv(self):
        """Create the monitoring evidence CSV when absent."""

        self.output_csv.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if self.output_csv.exists():
            return

        with self.output_csv.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:
            writer = csv.writer(handle)

            writer.writerow(
                [
                    "timestamp_utc",
                    "sample_count",
                    "psi",
                    "bin_0_025",
                    "bin_025_050",
                    "bin_050_075",
                    "bin_075_100",
                    "state",
                ]
            )

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
            self.inference_topic,
            qos=1,
        )

        LOGGER.info(
            "Subscribed to %s",
            self.inference_topic,
        )

        LOGGER.info(
            "Monitoring score: probabilities[0] = P(Normal)"
        )

        LOGGER.info(
            "Rolling window=%d, interval=%.0fs",
            ROLLING_WINDOW_SIZE,
            MONITOR_INTERVAL_SECONDS,
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

            probabilities = payload[
                "probabilities"
            ]

            if len(probabilities) != 3:
                raise ValueError(
                    "Expected three class probabilities"
                )

            normal_confidence = float(
                probabilities[0]
            )

            if (
                normal_confidence < 0.0
                or normal_confidence > 1.0
            ):
                raise ValueError(
                    "Normal confidence is outside [0, 1]"
                )

            with self.lock:
                self.scores.append(
                    normal_confidence
                )

        except Exception:
            LOGGER.exception(
                "Rejected inference message"
            )

    def evaluate(self):
        """Calculate and print PSI for the current rolling window."""

        with self.lock:
            score_values = list(
                self.scores
            )

        if len(
            score_values
        ) < ROLLING_WINDOW_SIZE:
            LOGGER.info(
                "PSI waiting for rolling window: %d/%d inferences",
                len(score_values),
                ROLLING_WINDOW_SIZE,
            )
            return

        counts, current_proportions = (
            confidence_distribution(
                score_values
            )
        )

        psi_value, contributions = (
            calculate_psi(
                self.reference_proportions,
                current_proportions,
            )
        )

        print(
            "Current PSI="
            + format(
                psi_value,
                ".3f",
            ),
            flush=True,
        )

        LOGGER.info(
            "Current bins=%s counts=%s contributions=%s",
            PSI_BIN_LABELS,
            counts.tolist(),
            np.round(
                contributions,
                6,
            ).tolist(),
        )

        if psi_value > DRIFT_ALERT_THRESHOLD:
            print(
                "[LOGIBRIDGE DRIFT ALERT] PSI="
                + format(
                    psi_value,
                    ".3f",
                ),
                flush=True,
            )

            self.alert_active = True
            state = "DRIFT"

        elif (
            self.alert_active
            and psi_value < RECOVERY_THRESHOLD
        ):
            print(
                "[LOGIBRIDGE RECOVERY] PSI="
                + format(
                    psi_value,
                    ".3f",
                ),
                flush=True,
            )

            self.alert_active = False
            state = "RECOVERED"

        else:
            state = "STABLE"

        with self.output_csv.open(
            "a",
            newline="",
            encoding="utf-8",
        ) as handle:
            writer = csv.writer(handle)

            writer.writerow(
                [
                    datetime.now(
                        timezone.utc
                    ).isoformat(),
                    len(score_values),
                    format(
                        psi_value,
                        ".9f",
                    ),
                    *[
                        format(
                            float(value),
                            ".9f",
                        )
                        for value in current_proportions
                    ],
                    state,
                ]
            )

    def stop(self):
        self.running = False

    def run(self):
        """Connect to MQTT and evaluate PSI every 60 seconds."""

        LOGGER.info(
            "Reference distribution: %s",
            self.reference_proportions.tolist(),
        )

        LOGGER.info(
            "Connecting to MQTT broker %s:%d",
            self.broker_host,
            self.broker_port,
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
                now = time.monotonic()

                if (
                    now
                    - self.last_evaluation_time
                    >= MONITOR_INTERVAL_SECONDS
                ):
                    self.evaluate()

                    self.last_evaluation_time = (
                        now
                    )

                time.sleep(0.25)

        finally:
            self.running = False
            self.client.loop_stop()
            self.client.disconnect()

            LOGGER.info(
                "Drift monitor stopped"
            )


def build_parser():
    """Build command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Monitor LogiEdge confidence-score PSI"
        )
    )

    parser.add_argument(
        "--broker",
        default="127.0.0.1",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=1883,
    )

    parser.add_argument(
        "--truck-id",
        default="TRUCK-001",
    )

    parser.add_argument(
        "--reference",
        default=(
            "monitoring/reference_dist.json"
        ),
    )

    parser.add_argument(
        "--output-csv",
        default=(
            "monitoring/psi_events.csv"
        ),
    )

    return parser


def main():
    """Program entry point."""

    arguments = build_parser().parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | %(levelname)-8s | "
            "%(name)s | %(message)s"
        ),
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    monitor = DriftMonitor(
        broker_host=arguments.broker,
        broker_port=arguments.port,
        truck_id=arguments.truck_id,
        reference_path=arguments.reference,
        output_csv=arguments.output_csv,
    )

    def stop_monitor(
        signal_number,
        frame,
    ):
        del frame

        LOGGER.info(
            "Received signal %s",
            signal_number,
        )

        monitor.stop()

    signal.signal(
        signal.SIGINT,
        stop_monitor,
    )

    signal.signal(
        signal.SIGTERM,
        stop_monitor,
    )

    monitor.run()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
