import http.server
import socketserver
import os
import sys

PORT = 8085

# Ensure directory is web-demo
web_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(web_dir)

class NoCacheHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate, max-age=0')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Expires', '0')
        self.send_header('Access-Control-Allow-Origin', '*')
        super().end_headers()

try:
    with socketserver.TCPServer(("", PORT), NoCacheHTTPRequestHandler) as httpd:
        print(f"Serving web-demo at http://localhost:{PORT} with NO-CACHE headers...")
        httpd.serve_forever()
except OSError:
    PORT = 8090
    with socketserver.TCPServer(("", PORT), NoCacheHTTPRequestHandler) as httpd:
        print(f"Serving web-demo at http://localhost:{PORT} with NO-CACHE headers...")
        httpd.serve_forever()
