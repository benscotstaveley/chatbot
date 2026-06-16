#!/bin/bash


for test in `find . -type d -name test_\*`; do
    echo $test
    cd $test
    python ../../main.py --single-shot --dump-prompts > chatbot.log
    python ../../run_golden.py
    cd ..
done
