#!/usr/bin/env python3

from __future__ import print_function
import traceback
import sys
from eccodes import *
import numpy as np
import configparser
import json,h5py
def initDict(dictKeys):
    d = {}
    for k in dictKeys:
        d[k] = []
    return d

def createMap(inDict,outDict):
    m = {}
    for k in list(inDict.keys()):
        m[k] = {}
        for i,kk in enumerate(inDict[k]):
            m[k][kk] = outDict[k][i]
    return m


def bufr_decode(input_file,\
                imagerKeys,\
                imagerKeysArray,\
                keysUsedButDropped,\
                keysUsedButDroppedArray,\
                keysModified,\
                keysModifiedArray,\
                keysPassed,\
                keysPassedArray,\
                keysUnused,\
                keysUnusedArray):

    imagerData = initDict(imagerKeys)
    imagerDataArray = initDict(imagerKeysArray)
    usedButDroppedData = initDict(keysUsedButDropped) 
    usedButDroppedDataArray = initDict(keysUsedButDroppedArray) 
    modifiedData = initDict(keysModified)
    modifiedDataArray = initDict(keysModifiedArray)
    passedData = initDict(keysPassed)
    passedDataArray = initDict(keysPassedArray)
    unusedData = initDict(keysUnused) 
    unusedArray = initDict(keysUnusedArray)

    f = open(input_file, 'rb')
    # Message number 1
    # -----------------
    iii=0
    #while (True):
    for izzz in list(range(0,1)):
        #print ('Decoding message number {}'.format(iii))
        iii+=1
        try:
            ibufr = codes_bufr_new_from_file(f)
        except:
            break
        # exit while if we reach end of file.
        # going off example from ECMWF tutorial, not convinced 
        # this is the cleanest way to do this.
        if(ibufr is None):
            break
        
        codes_set(ibufr, 'unpack', 1)
        #need this because of some weirdness in bufr message which tries to put multiple scanlines in one message.
        if( 'scanLineNumber' in list( modifiedData.keys() ) ):
            cnt = codes_get_array(ibufr,'scanLineNumber')
            if(len(cnt)>1):
                continue
        #sometimes it just has 1 crosstrack point which is garbage, skip it.
        skip = False
        for k in list(usedButDroppedDataArray.keys()):
            if '#nonNormalizedPrincipalComponentScore' in k:
                tmp = codes_get_array(ibufr,k)
                if len(tmp) == 1:
                    skip = True
        if(skip):
            continue
        for k in list(imagerData.keys()):
            imagerData[k].append(codes_get(ibufr, k))
        for k in list(imagerDataArray.keys()):
            imagerDataArray[k].append(codes_get_array(ibufr,k))
        for k in list(usedButDroppedData.keys()):
            usedButDroppedData[k].append(codes_get(ibufr,k))
        for k in list(modifiedData.keys()):
            modifiedData[k].append(codes_get(ibufr,k))
        for k in list(usedButDroppedDataArray.keys()):
            tmp = codes_get_array(ibufr,k)
            usedButDroppedDataArray[k].append(tmp)

        for k in list(passedData.keys()):
            passedData[k].append(codes_get(ibufr,k))
        for k in list(modifiedDataArray.keys()):
            tmp = codes_get_array(ibufr,k)
#           if('hour' in k or 'minute' in k or 'second' in k):
#                print(k,tmp)
            modifiedDataArray[k].append(tmp)
        for k in list(passedDataArray.keys()):
            tmp = codes_get_array(ibufr,k)
            passedDataArray[k].append(tmp)
        for k in list(unusedData.keys()):
            tmp = codes_get(ibufr,k)
            if('databaseIdentification' in k):
                if tmp!=104:
                    print(k,tmp)
            unusedData[k].append(tmp)
        for k in list(unusedArray):
            unusedArray[k].append(codes_get_array(ibufr,k))
        
        codes_release(ibufr)
    f.close()
    return imagerData,\
           imagerDataArray,\
           usedButDroppedData,\
           usedButDroppedDataArray,\
           modifiedData,\
           modifiedDataArray,\
           passedData,\
           passedDataArray,\
           unusedData,\
           unusedArray


def main(matrix, subset, band_limits, pc_limits, scale_limits, scale_values, ioIn, ioBufOut, ioNcOut):
   
    if len(sys.argv) < 2:
        print('Usage: ', sys.argv[0], ' BUFR_file_in')
        sys.exit(1)
    infn = sys.argv[1]
    outfn = 'none' 
    print('Input File: ',infn)
    
    try:
        imagerData,\
        imagerDataArray,\
        usedButDroppedData,\
        usedButDroppedDataArray,\
        modifiedData,\
        modifiedDataArray,\
        passedData,\
        passedDataArray,\
        unusedData,\
        unusedArray = bufr_decode(infn,\
                                  ioIn['imager'],\
                                  ioIn['imagerArray'],\
                                  ioIn['usedButDropped'],\
                                  ioIn['usedButDroppedArray'],\
                                  ioIn['modified'],\
                                  ioIn['modifiedArray'],\
                                  ioIn['passed'],\
                                  ioIn['passedArray'],\
                                  ioIn['unused'],\
                                  ioIn['unusedArray'])

    except CodesInternalError as err:
        traceback.print_exc(file=sys.stderr)
        return 1
    

if __name__ == "__main__":
    cfg = configparser.ConfigParser()
    cfg.read( 'channel_subset.cfg' )


    matrix = {}
    subset = {}
    band_limits = {}
    pc_limits = {}
    scale_limits = {}
    scale_values = {}
    verbose = False
    for k in list(cfg['subset'].keys()):
        if(verbose):print(k)
        subset[k] = json.loads(cfg['subset'][k])
        if(verbose):print(k,subset[k])
    for k in list(cfg['matrix'].keys()):
        matrix[k] = cfg['matrix'][k]
        if(verbose):print(k,matrix[k])
    for k in list(cfg['band_limits']):
        a,b = cfg['band_limits'][k].split(',')
        band_limits[k] = [int(a),int(b)]
    for k in list(cfg['pc_limits']):
        a,b = cfg['pc_limits'][k].split(',')
        pc_limits[k] = [int(a),int(b)]
    for k in list(cfg['scale_channel_limits']):
        a,b = cfg['scale_channel_limits'][k].split(',')
        scale_limits[k] = [int(a),int(b)]
    for k in list(cfg['scale_values']):
        a = float(cfg['scale_values'][k])
        scale_values[k] = a 

    iocfg = configparser.ConfigParser()
    iocfg.read( 'iasi_io_map.cfg' )
    flagsCombine = json.loads(iocfg['flagsCombine']['input'])
    ioIn = {}
    ioBufOut = {}
    ioNcOut = {}
    for k in iocfg.keys():
        #already have flags combined handled above
        if k=='flagsCombine': continue
        if ('input' in iocfg[k].keys()):
            ioIn[k] = json.loads(iocfg[k]['input'])
        else:
            ioIn[k] = []
        if('outBuf' not in list(iocfg[k].keys())):
            ioBufOut[k] = ioIn[k]
        else:
            ioBufOut[k] = json.loads(iocfg[k]['outBuf'])
        if('ncOut' not in list(iocfg[k].keys()) ):
            ioNcOut[k] = ioIn[k]
        else:
            ioNcOut[k] = json.loads(iocfg[k]['outNc']) 


    sys.exit( main(matrix, subset, band_limits, pc_limits, scale_limits, scale_values, ioIn, ioBufOut, ioNcOut) )
