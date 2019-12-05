import mysql.connector
#import bencode
#import binascii
#import hashlib
import os
import sys

conn = mysql.connector.connect(host="localhost",user="root",password="", database="database13")
cursor = conn.cursor()
# path = "define the path of file where  file data is exit"
# dirs = os.listdir(path)
# for file in dirs:
#         try:
#                 with open(os.path.join(path, file), 'rb') as torrentfile:
#                         torrent = bencode.bdecode(torrentfile.read())
#                         user = ("torrent['info']['name']","torrent['info']['length']","(hashlib.sha1(bencode.bencode(torrent['info'])).hexdigest())")
#                         cursor.execute("""INSERT INTO torrent_infos (Name, Size, Hash) VALUES(%s, %s, %s)""", user)
#         except bencode.BTL.BTFailure:
#                 continue


conn.close()