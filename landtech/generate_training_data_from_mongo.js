

const fs = require('fs-extra'),
      co = require('co'),
      vm = require('vm'),
      MongoClient = require('mongodb').MongoClient,
      { compact } = require('lodash');

var argv = require('yargs')
  .usage('Usage: $0 -d [dir]')
  .example('$0 count -d ../data')
  .help('h')
  .alias('d', 'dir')
  .demandOption(['d'])
  .nargs('d', 1)
  .describe('d', 'Output dir for data')
  .argv;

const MONGO_CONNECTION_STRING = process.env.MONGO_CONNECTION_STRING; //Must include correct database

co(function*() {
  let db = yield MongoClient.connect(MONGO_CONNECTION_STRING);

  let policyTags = db.collection('policy_tags'); 

  let totalCount = yield policyTags.count();

  let testCount = Math.ceil(totalCount * 0.2);
  let devCount = Math.ceil(totalCount * 0.2);
  let trainCount = totalCount - testCount - devCount;

  let trainIndex = 0;
  let devIndex = trainIndex + trainCount;
  let testIndex = devIndex + devCount;

  let totalProcessed = 0;

  let trainSet = [];
  let devSet = [];
  let testSet = [];
  let cursor = policyTags.find({});
  while(yield cursor.hasNext()) {
    let doc = yield cursor.next();
    if (totalProcessed >= testIndex) {
      testSet.push(doc);
    } else if (totalProcessed >= devIndex) {
      devSet.push(doc);
    } else {
      trainSet.push(doc);
    }

    totalProcessed++;
  }

  function processSet(docSet, fileName) {
    let text = docSet.map(doc => {
      let docText = doc.tagged_words
        .filter(taggedWord => !taggedWord.is_non_word)
        .map(taggedWord => `${taggedWord.word} ${taggedWord.tag}`);
      return docText.join('\n');
    }).join('\n\n');
    return fs.writeFile(`${argv.d}/${fileName}`, text);
  }

  yield processSet(trainSet, 'policy.train.txt');
  yield processSet(testSet, 'policy.test.txt');
  yield processSet(devSet, 'policy.dev.txt');

}).then(() => {
  process.exit(0);
}, err => {
  console.error(err);
  process.exit(1);
});