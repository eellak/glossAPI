import downloader_app as dl
import extractor_app as xt
import asyncio
import json
import os

def read_filename_index(json_file="./filename_index.json"):
    """
    Reads the filename index JSON file and creates a list of args objects.
    """
    with open(json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    args_list = []
    for filename, attributes in data.items():
        args_obj = type('args', (object,), {
            'filename': filename,
            'sleep': 1,
            'req': attributes.get('req', 'get'),
            'type': attributes.get('type', 'pdf'),
            'batch': 10
        })()
        args_list.append(args_obj)
    return args_list

def remove_from_list(remove_these, list):
    for e in remove_these:
        list.remove(e)
    return list

async def main():
    reached_end_of_file = False
    args_list = read_filename_index()
    completed = []
    i = 0
    current_path = os.getcwd()
    while args_list and i < 2:
        for args in args_list:
            reached_end_of_file = False
            reached_end_of_file = await dl.run(args)
            if reached_end_of_file:
                completed.append(args)
        args_list = remove_from_list(completed, args_list)
        completed = []
        i += 1
        xt.run(current_path)


asyncio.run(main())