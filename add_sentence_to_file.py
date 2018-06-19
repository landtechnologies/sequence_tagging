import argparse
import re

parser = argparse.ArgumentParser(description='Add sentences to a training file.')
parser.add_argument('-f', '--file', dest='file', metavar='FILE', type=str, required=True,
                    help='File to add sentence to')
parser.add_argument('-s', dest='sentence', metavar='SENTENCE', type=str, required=True, help='Sentence to add')

args = parser.parse_args()

if len(args.sentence):
  with open(args.file, "a") as toAppendTo:
    lines = map(
      lambda word: "{} 0".format(word), 
      filter(
        lambda word: word is not None and word != '' and word != '\n', 
        re.split(r'([.,\/#!$%\^&\*;:{}=\-_`~()\n])| ', args.sentence)
      )
    )
    appendText = '\n' + '\n'.join(lines)
    appendText += '\n'
    toAppendTo.write(appendText)
else:
  print("No sentence, doing nothing")