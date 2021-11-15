"""
curl https://blockstream.info/api/block-height/634787
curl https://blockstream.info/api/block/0000000000000000000880a61b2ee8cc205fed9d111435c340d366a4f9d595c3/txid/771
curl https://blockstream.info/api/tx/58b279b045215582ec947cbe509263e71dad4bd24d1fcc782c2864db9ce0f3f3
Then look at the index in vout i.e. index 0 -> this is the channel balance in sat

Check if channel is still open:
curl https://blockstream.info/api/tx/58b279b045215582ec947cbe509263e71dad4bd24d1fcc782c2864db9ce0f3f3/outspend/0
or
curl https://blockstream.info/api/tx/58b279b045215582ec947cbe509263e71dad4bd24d1fcc782c2864db9ce0f3f3/outspends
"""
import logging
import networkx as nx
import requests as re
import os


def importFromGraphML(filePath):
    logging.info("Starting " + filePath)
    g = nx.read_graphml(filePath)

    for edge in g.edges(data=True):
        try:
            scid = (edge[2]['scid'])
            scid_components = scid.split('x')
            block_height = scid_components[0]
            tx_index = scid_components[1]
            output = scid_components[2].split('/')[0]

            # print('##########################')
            # print('Starting requests for channel ' + scid)

            response = re.get('https://blockstream.info/api/block-height/' + block_height)
            block_hash = response.text

            response = re.get('https://blockstream.info/api/block/' + block_hash + '/txid/' + tx_index)
            tx_hash = response.text

            response = re.get('https://blockstream.info/api/tx/' + tx_hash)
            response_json = response.json()
            channel_balance = response_json['vout'][int(output)]['value']

            response = re.get('https://blockstream.info/api/tx/' + tx_hash + '/outspend/' + output)
            response_json = response.json()
            is_spent = response_json['spent']

            edge[2]['channel_balance'] = channel_balance
            edge[2]['open'] = is_spent

            # print('Channel ' + scid + ' was funded with ' + str(channel_balance))
            # print('Is channel ' + scid + ' still open? ' + str(is_spent) + '\n')

        except re.exceptions.RequestException as e:
            print(e)
            pass

    filePath_split = filePath.split('/')
    filePath_split[2] = 'updated_' + filePath_split[2]
    filePath = '/'.join(filePath_split)
    # print(filePath)

    nx.write_graphml(g, filePath)
    logging.info("Saving  " + filePath)


format = "%(asctime)s: %(message)s"
logging.basicConfig(format=format, level=logging.INFO,
                    datefmt="%H:%M:%S")
filepath = '../graphs'
filenames = next(os.walk(filepath), (None, None, []))[2]  # [] if no file
for filename in filenames:
    # print(filename)
    importFromGraphML(filepath + '/' + filename)
