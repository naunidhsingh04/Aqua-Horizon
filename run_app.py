"""
Starts a small web server for the website in web/ and shows it.

  python run_app.py            local PC: opens http://localhost:8000 (or the next free port up to 8019)
  python run_app.py --check    starts the server, requests every page / data file the site needs,
                               prints OK or MISSING for each, then stops (no browser needed)
  %run run_app.py              Google Colab: serves the site in the background and shows it inside the notebook
"""
import sys
import time
import functools
import threading
import webbrowser
import http.server
import socketserver
import urllib.request
from pathlib import Path

PORT = 8000
WEB_DIR = Path(__file__).resolve().parent / "web"

# Everything the page loads while starting up (index.html + scripts + data files)
REQUIRED_FILES = [
    "index.html", "css/style.css", "js/app.js",
    "data/india_districts.geojson", "data/flood_intelligence_data.json",
    "data/hybrid_benchmarks.json", "data/kfold_hybrid_benchmarks.json",
    "data/combo_hybrid_benchmarks.json", "data/image_benchmark_metrics.json",
]


def start_server():
    """Start the server on the first free port; returns (server, port)."""
    class Handler(http.server.SimpleHTTPRequestHandler):
        def end_headers(self):
            # Never cache, so a refreshed model result shows up immediately
            self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
            super().end_headers()

        def log_message(self, *args):   # keep the console / notebook output clean
            pass

    # Threaded so several browser requests (map, JSON files) are served at the same time
    class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
        daemon_threads = True
        allow_reuse_address = True

    # `directory=` serves web/ without changing the current folder (important inside a notebook)
    handler = functools.partial(Handler, directory=str(WEB_DIR))
    for port in range(PORT, PORT + 20):
        try:
            return ThreadedTCPServer(("", port), handler), port
        except OSError:
            continue
    raise RuntimeError("No free port between 8000 and 8019")


def check_files():
    """Request every file the site needs and report whether it is served."""
    httpd, port = start_server()
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    time.sleep(0.5)
    print("=" * 65)
    print(f" WEBSITE FILE CHECK (server on port {port})")
    print("=" * 65)
    missing = 0
    for name in REQUIRED_FILES:
        try:
            with urllib.request.urlopen(f"http://localhost:{port}/{name}", timeout=10) as r:
                print(f"  OK       {name:<40} {len(r.read()):>10,} bytes")
        except Exception as e:
            missing += 1
            print(f"  MISSING  {name:<40} ({e})")
    httpd.shutdown()
    print("-" * 65)
    print("All website files are present." if missing == 0 else f"{missing} file(s) missing - run the earlier steps first.")
    return missing


def main():
    if not WEB_DIR.exists():
        print(f"Error: Directory {WEB_DIR} does not exist.")
        sys.exit(1)

    if "--check" in sys.argv:
        sys.exit(1 if check_files() else 0)

    httpd, port = start_server()
    url = f"http://localhost:{port}"
    print("=" * 65)
    print(" AQUA HORIZON - FLOOD INTELLIGENCE WEB SERVER")
    print("=" * 65)
    print(f" Serving web app from: {WEB_DIR}")
    print(f" Local URL:            {url}")

    if "google.colab" in sys.modules:
        # Google Colab: no browser on the server, so run in the background and show the site in the notebook
        from google.colab import output
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        print(" Running inside Google Colab - the site is shown below.")
        print("=" * 65)
        output.serve_kernel_port_as_iframe(port, height=720)
        return

    print(" Opening browser now...")
    print(" Press Ctrl+C to terminate.")
    print("=" * 65)
    webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer gracefully stopped.")
        sys.exit(0)


if __name__ == '__main__':
    main()
