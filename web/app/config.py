import os

CONFIG_DIR = os.environ.get("HEIMDALL_CONFIG_DIR", "/app/config")
CAMERAS_DIR = os.path.join(CONFIG_DIR, "cameras")
HEIMDALL_CONFIG = os.path.join(CONFIG_DIR, "heimdall.jsonc")
APRILTAG_CONFIG = os.path.join(CONFIG_DIR, "apriltag_layout.jsonc")
LOGS_DIR = os.environ.get("HEIMDALL_LOGS_DIR", "/app/logs")
TRACKER_LOG = os.path.join(LOGS_DIR, "tracker_log.csv")
HEIMDALL_CONTAINER_NAME = os.environ.get("HEIMDALL_CONTAINER", "heimdall")
