import json
import os
from pathlib import Path
import sys

from glossapi import Corpus
from glossapi.ledger.writer import open_readonly

def main():
    root, mode = Path(sys.argv[1]), sys.argv[2]
    corpus = Corpus(root/'empty',root/'output')
    try:
        if mode == 'start':
            result = corpus.run([Path(sys.argv[3])],stages=('extract','clean','ocr','export'),settings={'debug':os.environ.get('PROBE_DEBUG','off'),'extract':{'repair_fonts':False}})
            (root/'runid').write_text(result['run_id'])
        else:
            result = corpus.run(resume=(root/'runid').read_text(),retry_failed=True)
        data = dict(state=result['state'],counts=result.get('document_counts'))
    except Exception as error:
        data = dict(error_type=type(error).__name__,error_code=getattr(error,'code',None),message=str(error))
    with open_readonly(root/'output/corpus.sqlite3') as connection:
        data['attempts']=connection.execute('SELECT COUNT(*) FROM attempts').fetchone()[0]
        data['errors']=[list(row) for row in connection.execute(f"SELECT event_kind,reason_code,status FROM {connection.event_table} WHERE reason_code IN ('input_hash_changed','output_hash_mismatch')")]
        data['ocr_tasks']=[list(row) for row in connection.execute("SELECT operation,status,reason_code FROM current_task_state WHERE stage='ocr'")]
    (root/(mode+'.json')).write_text(json.dumps(data,indent=2))
    print(json.dumps(data))

if __name__=='__main__':
    main()
