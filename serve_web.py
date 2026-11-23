import functools
import http.server
import os
import socket
import socketserver


def get_lan_ips():
    ips = set()
    try:
        hostname = socket.gethostname()
        for info in socket.getaddrinfo(hostname, None, socket.AF_INET):
            ip = info[4][0]
            if not ip.startswith("127."):
                ips.add(ip)
    except OSError:
        pass
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            if not ip.startswith("127."):
                ips.add(ip)
    except OSError:
        pass
    return sorted(ips)


class ReusableTCPServer(socketserver.TCPServer):
    allow_reuse_address = True


def main():
    port = 8501
    proto_dir = os.path.dirname(os.path.abspath(__file__))
    handler = functools.partial(
        http.server.SimpleHTTPRequestHandler, directory=proto_dir
    )
    lan_ips = get_lan_ips()

    print("=" * 64)
    print("Server Prototipe PulmoSense (Prediksi Risiko TB) Aktif!")
    print(f" - Buka di Laptop : http://127.0.0.1:{port}/index.html")
    if lan_ips:
        for ip in lan_ips:
            print(f" - Buka di HP (Wi-Fi sama) : http://{ip}:{port}/index.html")
    else:
        print(f" - Buka di HP (Wi-Fi sama) : http://<IP-Laptop-Anda>:{port}/index.html")
    print("=" * 64)
    print("Tekan Ctrl+C untuk menghentikan server.", flush=True)

    with ReusableTCPServer(("0.0.0.0", port), handler) as httpd:
        httpd.serve_forever()


if __name__ == "__main__":
    main()
