#!/bin/bash

if (( $# > 0 )); then
    TESTS=$@
else
    TESTS=`find . -type d -name test_\*`
fi

for test in $TESTS ; do
    echo $test
    cd $test
    python ../../main.py --single-shot --dump-prompts > chatbot.log
    python ../../run_golden.py
    cd ..
done
