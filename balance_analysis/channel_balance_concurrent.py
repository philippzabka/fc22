import networkx as nx
import requests as re
import os
import logging
import threading
import time
import multiprocessing


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

        except re.exceptions.RequestException as e:
            print(e)
            pass

    filePath_split = filePath.split('/')
    filePath_split[2] = 'updated_' + filePath_split[2]
    filePath = '/'.join(filePath_split)

    nx.write_graphml(g, filePath)
    logging.info("Saving  " + filePath)


if __name__ == "__main__":
    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO,
                        datefmt="%H:%M:%S")

    filepath = '../graphs'
    # filenames = next(os.walk(filepath), (None, None, []))[2]  # [] if no file
    filenames = ['1564653600_lngraph.graphml', '1572606000_lngraph.graphml', '1585735200_lngraph.graphml',
                 '1596276000_lngraph.graphml', '1606820400_lngraph.graphml', '1609498800_lngraph.graphml']

    threads = list()
    for index in range(len(filenames)):
        logging.info("Main    : create and start thread %d.", index)
        filename = filenames[index]
        print(filepath + '/' + filename)
        x = threading.Thread(target=importFromGraphML, args=(filepath + '/' + filename,))
        threads.append(x)
        x.start()
