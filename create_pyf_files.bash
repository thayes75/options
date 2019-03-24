#!/bin/bash
#
#  Quick wrapper to create the pyf files
#
#  Tyler Hayes - 2014-05-17
#
#------------------------------------------------------------------------
FLIST=`ls -1 *f90`
for F90FILE in `echo $FLIST`
do
    MODNAME=`echo $F90FILE | cut -f1 -d\.`
    SIGFLNM=$MODNAME.pyf
    f2py $F90FILE -m $MODNAME -h $SIGFLNM
done
