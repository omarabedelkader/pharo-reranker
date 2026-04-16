#!/usr/bin/env sh
set -eu

mkdir -p image
cd image
curl -fsSL https://get.pharo.org/140+vm | bash
curl -fLO https://files.pharo.org/image/140/Pharo14.0-SNAPSHOT.build.415.sha.359c9be46a.arch.64bit.zip
unzip -o Pharo14.0-SNAPSHOT.build.415.sha.359c9be46a.arch.64bit.zip
./pharo Pharo14.0-SNAPSHOT-64bit-359c9be46a.image eval "2+2"
echo "IMAGE OK"



./pharo Pharo14.0-SNAPSHOT-64bit-359c9be46a.image eval --save "
Metacello new
    githubUser: 'omarabedelkader'
    project: 'AI-Sorter'
    commitish: 'main'
    path: 'src';
    baseline: 'AISorter';
    load."

echo "AI-SORTER OK"

echo "Done Bench installation"
./pharo Pharo14.0-SNAPSHOT-64bit-359c9be46a.image eval --save "CooStaticBenchmarksVariablesSorter nec."
./pharo Pharo14.0-SNAPSHOT-64bit-359c9be46a.image eval --save "CooStaticBenchmarksVariablesSorter ast."

./pharo Pharo14.0-SNAPSHOT-64bit-359c9be46a.image eval --save "CooStaticBenchmarksMessageSorter nec."
./pharo Pharo14.0-SNAPSHOT-64bit-359c9be46a.image eval --save "CooStaticBenchmarksMessageSorter ast."

echo "BENCH OK"