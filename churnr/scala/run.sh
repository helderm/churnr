#!/bin/bash
RELPATH="`dirname \"$0\"`"
ABSPATH="`( cd \"$RELPATH\" && pwd )`"
cd $ABSPATH; sbt -Dbigquery.project=$1 "runMain com.spotify.churnr.Parser \
--project=$1 --runner=DataflowRunner \
--zone=europe-west1-d --output=$2"; cd --
