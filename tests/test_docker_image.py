import requests
import time
import sys

API_URL = "http://localhost:8000/health"
MLFLOW_URL = "http://localhost:5000"


def wait_for_service(url, name, timeout=120):
    print(f"[INFO] Waiting for {name} at {url} ...")
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                print(f"[SUCCESS] {name} is up! Response: {resp.text[:200]}")
                return True
            else:
                print(f"[WARN] {name} returned status {resp.status_code}")
        except Exception as e:
            print(f"[WAIT] {name} not ready: {e}")
        time.sleep(3)
    print(f"[ERROR] {name} did not become ready in {timeout} seconds.")
    return False


def main():
    api_ok = wait_for_service(API_URL, "FastAPI API")
    mlflow_ok = wait_for_service(MLFLOW_URL, "MLflow UI")
    if not (api_ok and mlflow_ok):
        print("[FAIL] One or more services failed to start.")
        sys.exit(1)
    print("[PASS] All Dockerized services are healthy!")

if __name__ == "__main__":
    main() 