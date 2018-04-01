from gevent import monkey;monkey.patch_all()
import gevent
import socket
import time

file = open('b64_img.txt', 'rb')
b64_img = file.readline() + b'\n'
file.close()
 
def do_connect(addr):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(addr)
    sock.send(b64_img)
    print(sock.recv(1024))
    sock.close()
 
addr = ('127.0.0.1', 16000)

since = time.time()
greenlets = [gevent.spawn(do_connect, addr) for i in range(1)]
gevent.joinall(greenlets, timeout=5)
print(time.time() - since)