import os
import sys
import webbrowser
import http.server
import socketserver
from pathlib import Path

PORT = 8000

def main():
    web_dir = Path(__file__).parent / "web"
    if not web_dir.exists():
        print(f"Error: Directory {web_dir} does not exist.")
        sys.exit(1)

    os.chdir(web_dir)

    class CustomHandler(http.server.SimpleHTTPRequestHandler):
        def end_headers(self):
            # Disable cache for hackathon dev ease
            self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
            super().end_headers()

    class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
        daemon_threads = True

    # Find free port if 8000 is taken
    port = PORT
    for p in range(PORT, PORT + 20):
        try:
            httpd = ThreadedTCPServer(("", p), CustomHandler)
            port = p
            break
        except OSError:
            continue

    url = f"http://localhost:{port}"
    print("=" * 65)
    print(" NATIONAL FLOOD INTELLIGENCE SYSTEM (NFIS) - WEB SERVER")
    print("=" * 65)
    print(f" Serving web app from: {web_dir}")
    print(f" Local URL:            {url}")
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
