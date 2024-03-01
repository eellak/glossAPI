import argparse
import os
from lxml import etree
import pickle
import re
import pandas

parser = argparse.ArgumentParser(
        prog = "This solves the issue of multiple XML files in a nested directory",
        description = "This program will help parse OPUS XML files that are nested into directories and consist of several hundred XML files",
        epilog = "This is experimental work in progress, use with caution. No warranty whatsoever."
        )

parser.add_argument('-d', '--dir', help='Define the directory containing the files', default="xml/el")
parser.add_argument('-o', '--output-directory', help='Set the directory where the pickle files will be siphoned to.', default="./text_pickles")
parser.add_argument('-s', '--suffix', help='Choose a suffix to prepend to the output pickle filenames; this is mandatory.', default="no-suffix-given")
parser.add_argument('-t', '--stats-filename', help='Set a filename to save document statistics to, via Pandas.', default="document_record_stats.csv")
parser.add_argument('-x', '--xpath-selector', help='Xpath expression used to select nodes', default="//w")
parser.add_argument('-v', '--verbose', help='Do you want trial by trial updates?', default=True)
parser.add_argument('-p', '--sentences', help='(From p for Greek protasis) Parse at the sentence level?', default=False)
args = parser.parse_args()

# text data stats

td_stats = list()

def process_single_file(fname, base_dir, suffix=args.suffix, xe=args.xpath_selector, serialize=True):

    # xe: xpath expression
    fi = open(fname, 'r')
    te = etree.parse(fi)
    fi.close()
    el = te.xpath(xe)
    if args.sentences:
        out = []
        if xe != "//s":
            print("Make sure the Xpath Selector selects sentences, I would expect s, but not mandatory")
        for s in el:
            content = ' '.join([w.text for w in s.findall('./w')])
            out.append(content)
    else:
        out = [wd.text for wd in el]

    # pickle filename prepare
    fn_stem = re.sub("\.xml$", "", fname) # remove the extension
    if args.sentences:
        pfnM1 = "{}_{}_se.p"
    else:
        pfnM1 = "{}_{}.p"
    pfn0 = pfnM1.format(suffix, fn_stem ) # concat with suffix
    pfn = os.path.join(base_dir, args.output_directory, pfn0) #  join with output dir

    # save stats
    td_stats.append({'file': pfn, 'n': len(out)})
    
    # serialize via pickle
    pfi = open(pfn, 'wb') 
    pickle.dump(out, pfi)
    pfi.close()

    return out

# switch to directory containing multiple files
base_dir = os.getcwd()
os.chdir(args.dir)

al_f = os.listdir()

cnt = 0
for fi in al_f:
    # if cnt > 5:
    #     break
    if os.path.splitext(fi)[1] == ".xml":
        process_single_file(fi, base_dir)
        cnt+=1
    if args.verbose:
        print("Trial No.: {}".format(cnt))

# print(td_stats)

df = pandas.DataFrame.from_records(td_stats)
# print(df)

df.to_csv(os.path.join(base_dir, args.stats_filename), sep=';', index=False)
print("Program complete")

