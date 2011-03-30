#!/bin/bash

for d in `find . -name "?_*"`; do 
	echo "syncing $d"
	cd $d
	./down.sh
	cd -
done
