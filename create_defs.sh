#!/bin/bash

if [ $# -eq 0 ]; then
    echo "Usage: create_defs.sh {path to eccodes directory, or directory above 'share'} {output}"
fi

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

mkdir definitions

cd definitions 

ln -s ${ECCODESDIR}/share/eccodes/definitions/boot.def

ln -s ${ECCODESDIR}/share/eccodes/definitions/budg

ln -s ${ECCODESDIR}/share/eccodes/definitions/cdf

ln -s ${ECCODESDIR}/share/eccodes/definitions/CMakeLists.txt

ln -s ${ECCODESDIR}/share/eccodes/definitions/common

ln -s ${ECCODESDIR}/share/eccodes/definitions/diag

ln -s ${ECCODESDIR}/share/eccodes/definitions/empty_template.def

ln -s ${ECCODESDIR}/share/eccodes/definitions/grib1/

ln -s ${ECCODESDIR}/share/eccodes/definitions/grib2/

ln -s ${ECCODESDIR}/share/eccodes/definitions/grib3/

ln -s ${ECCODESDIR}/share/eccodes/definitions/gts/

ln -s ${ECCODESDIR}/share/eccodes/definitions/hdf5/

ln -s ${ECCODESDIR}/share/eccodes/definitions/installDefinitions.sh

ln -s ${ECCODESDIR}/share/eccodes/definitions/mars

ln -s ${ECCODESDIR}/share/eccodes/definitions/mars_param.table

ln -s ${ECCODESDIR}/share/eccodes/definitions/metar

ln -s ${ECCODESDIR}/share/eccodes/definitions/parameters_version.def

ln -s ${ECCODESDIR}/share/eccodes/definitions/param_id.table

ln -s ${ECCODESDIR}/share/eccodes/definitions/param_limits.def

ln -s ${ECCODESDIR}/share/eccodes/definitions/stepUnits.table

ln -s ${ECCODESDIR}/share/eccodes/definitions/taf

ln -s ${ECCODESDIR}/share/eccodes/definitions/tide

ln -s ${ECCODESDIR}/share/eccodes/definitions/wrap

mkdir bufr

cd bufr/

ln -s ${ECCODESDIR}/share/eccodes/definitions/bufr/boot.def

ln -s ${ECCODESDIR}/share/eccodes/definitions/bufr/boot_edition_0.def

ln -s ${ECCODESDIR}/share/eccodes/definitions/bufr/boot_edition_1.def

ln -s ${ECCODESDIR}/share/eccodes/definitions/bufr/boot_edition_2.def

ln -s ${ECCODESDIR}/share/eccodes/definitions/bufr/boot_edition_3.def

ln -s ${ECCODESDIR}/share/eccodes/definitions/bufr/boot_edition_4.def

ln -s ${ECCODESDIR}/share/eccodes/definitions/bufr/dataKeys.def

ln -s ${ECCODESDIR}/share/eccodes/definitions/bufr/old_section.1.def

ln -s ${ECCODESDIR}/share/eccodes/definitions/bufr/rdb_key_28.def

ln -s ${ECCODESDIR}/share/eccodes/definitions/bufr/rdb_key.def

ln -s ${ECCODESDIR}/share/eccodes/definitions/bufr/section.0.def

ln -s ${ECCODESDIR}/share/eccodes/definitions/bufr/section.1.1.def

ln -s ${ECCODESDIR}/share/eccodes/definitions/bufr/section.1.2.def

ln -s ${ECCODESDIR}/share/eccodes/definitions/bufr/section.1.3.def

ln -s ${ECCODESDIR}/share/eccodes/definitions/bufr/section.1.4.def

ln -s ${ECCODESDIR}/share/eccodes/definitions/bufr/section1_flags.table

ln -s ${ECCODESDIR}/share/eccodes/definitions/bufr/section.2.def

ln -s ${ECCODESDIR}/share/eccodes/definitions/bufr/section.3.def

ln -s ${ECCODESDIR}/share/eccodes/definitions/bufr/section3_flags.table

ln -s ${ECCODESDIR}/share/eccodes/definitions/bufr/section.4.def

ln -s ${ECCODESDIR}/share/eccodes/definitions/bufr/section.5.def

ln -s ${ECCODESDIR}/share/eccodes/definitions/bufr/templates/

mkdir tables

cd tables/

ln -s ${ECCODESDIR}/share/eccodes/definitions/bufr/tables/3/

ln -s ${ECCODESDIR}/share/eccodes/definitions/bufr/tables/operators.table

mkdir 0

cd 0/

ln -s ${ECCODESDIR}/share/eccodes/definitions/bufr/tables/0/local/

mkdir wmo

cd wmo/

ln -s ${ECCODESDIR}/share/eccodes/definitions/bufr/tables/0/wmo/10

ln -s ${ECCODESDIR}/share/eccodes/definitions/bufr/tables/0/wmo/11

ln -s ${ECCODESDIR}/share/eccodes/definitions/bufr/tables/0/wmo/13

ln -s ${ECCODESDIR}/share/eccodes/definitions/bufr/tables/0/wmo/14

ln -s ${ECCODESDIR}/share/eccodes/definitions/bufr/tables/0/wmo/15

ln -s ${ECCODESDIR}/share/eccodes/definitions/bufr/tables/0/wmo/16

ln -s ${ECCODESDIR}/share/eccodes/definitions/bufr/tables/0/wmo/17

ln -s ${ECCODESDIR}/share/eccodes/definitions/bufr/tables/0/wmo/18

ln -s ${ECCODESDIR}/share/eccodes/definitions/bufr/tables/0/wmo/19

ln -s ${ECCODESDIR}/share/eccodes/definitions/bufr/tables/0/wmo/2

ln -s ${ECCODESDIR}/share/eccodes/definitions/bufr/tables/0/wmo/20

ln -s ${ECCODESDIR}/share/eccodes/definitions/bufr/tables/0/wmo/21

ln -s ${ECCODESDIR}/share/eccodes/definitions/bufr/tables/0/wmo/22

ln -s ${ECCODESDIR}/share/eccodes/definitions/bufr/tables/0/wmo/23

ln -s ${ECCODESDIR}/share/eccodes/definitions/bufr/tables/0/wmo/24

ln -s ${ECCODESDIR}/share/eccodes/definitions/bufr/tables/0/wmo/25

ln -s ${ECCODESDIR}/share/eccodes/definitions/bufr/tables/0/wmo/26

ln -s ${ECCODESDIR}/share/eccodes/definitions/bufr/tables/0/wmo/27

ln -s ${ECCODESDIR}/share/eccodes/definitions/bufr/tables/0/wmo/28

ln -s ${ECCODESDIR}/share/eccodes/definitions/bufr/tables/0/wmo/29

ln -s ${ECCODESDIR}/share/eccodes/definitions/bufr/tables/0/wmo/30/

ln -s ${ECCODESDIR}/share/eccodes/definitions/bufr/tables/0/wmo/31/

ln -s ${ECCODESDIR}/share/eccodes/definitions/bufr/tables/0/wmo/32/

ln -s ${ECCODESDIR}/share/eccodes/definitions/bufr/tables/0/wmo/33/

ln -s ${ECCODESDIR}/share/eccodes/definitions/bufr/tables/0/wmo/34/

ln -s ${ECCODESDIR}/share/eccodes/definitions/bufr/tables/0/wmo/35/

ln -s ${ECCODESDIR}/share/eccodes/definitions/bufr/tables/0/wmo/36/

ln -s ${ECCODESDIR}/share/eccodes/definitions/bufr/tables/0/wmo/37/

ln -s ${ECCODESDIR}/share/eccodes/definitions/bufr/tables/0/wmo/38/

ln -s ${ECCODESDIR}/share/eccodes/definitions/bufr/tables/0/wmo/6/

ln -s ${ECCODESDIR}/share/eccodes/definitions/bufr/tables/0/wmo/7

ln -s ${ECCODESDIR}/share/eccodes/definitions/bufr/tables/0/wmo/8

ln -s ${ECCODESDIR}/share/eccodes/definitions/bufr/tables/0/wmo/9

mkdir 12

cd 12/

ln -s ${ECCODESDIR}/share/eccodes/definitions/bufr/tables/0/wmo/12/codetables/

ln -s ${ECCODESDIR}/share/eccodes/definitions/bufr/tables/0/wmo/12/element.table

cp ${ECCODESDIR}/share/eccodes/definitions/bufr/tables/0/wmo/12/sequence.def .

cat ${THISDIR}/ncepiasi616.seq >> sequence.def
