"""
res = r.get('http://ws.audioscrobbler.com/2.0/?method=track.getinfo&api_key=b25b959554ed76058ac220b7b2e0a026&artist=cher&track=believe&format=json').json()
"""
from __future__ import print_function
import json
import pdb
import requests as req
import sys

API_URL = 'http://ws.audioscrobbler.com/2.0'
API_KEY = 'd4437f516dc33560c35b8038de33ee86'
METHOD = 'track.getinfo'
FORMAT = 'json'

class APIError(Exception):
    def __init__(self, message, **kwargs):
        super(APIError, self).__init__(message, **kwargs)
        self.message = message

genre_dict = {'errors': []}
def main():
    with open('wikifonia_songs.json') as fp:
        song_data = json.loads(fp.read())
    song_count = 0
    for artist in song_data:
        for song in song_data[artist]:
            song_count += 1
            try:
                res = req.get(API_URL, params={'method': METHOD, 'api_key': API_KEY,
                                               'artist': artist, 'track': song,
                                               'format': FORMAT})
                # pdb.set_trace()
                if res.status_code != 200:
                    raise APIError("status code: %d, body: %s" % (res.status_code,
                        res.body))
                elif 'error' in res.json().keys():
                    raise APIError(res.json()['message'].encode('utf-8'))

                for tag_dict in res.json()['track']['toptags']['tag']:
                    tag = tag_dict['name']
                    if tag not in genre_dict.keys():
                        genre_dict[tag] = []
                    genre_dict[tag].append("%s - %s" % (artist, song))
            except APIError as e:
                # pdb.set_trace()
                genre_dict['errors'].append({'song': song, 'artist': artist, 'error': e.message})
                # if len(genre_dict['errors'])%50 == 0:
                print('%d errors / %d tracks'% (len(genre_dict['errors']), song_count))
            except TypeError as e:
                genre_dict['errors'].append({'song': song, 'artist': artist,
                                             'error': "TypeError: ProtocolError(Connection Aborted)"})
                # if len(genre_dict['errors'])%50 == 0:
                print('%d errors / %d tracks'% (len(genre_dict['errors']), song_count))

    write_data()

def write_data():
    with open('wikifonia_genre_data.json', 'w') as fp:
        fp.write(json.dumps(genre_dict, sort_keys=True, indent=4, encoding='utf-8'))

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt as e:
        write_data()
