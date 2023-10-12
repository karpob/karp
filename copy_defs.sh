#!/bin/bash
export ECCODESDIR=$1
export OUTDIR=$2
export THISDIR=$PWD

if [ -d "$OUTDIR" ]; then
  cd $OUTDIR
else 
  mkdir -p $OUTDIR
  cd $OUTDIR
fi
echo $PWD
cp -r ${ECCODESDIR}/share/eccodes/definitions/ $OUTDIR

cd ${OUTDIR}/definitions/bufr/tables/0/wmo/12/

cat ${THISDIR}/ncepiasi616.seq >> sequence.def
