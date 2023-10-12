#!/usr/bin/env python3

from __future__ import print_function
import traceback
import sys
from eccodes import *
import numpy as np
import configparser,argparse
import json,h5py
import os
#print (help(codes_set_definitions_path))
#print ("Before ECCODES_DEFINITION_PATH ", os.environ['ECCODES_DEFINITION_PATH'])
#sys.exit()
def readMatrix(f):
    h5 = h5py.File(f,'r')
    # Matrix is pc# by channel #
    if('ReconstructionOperator' in list(h5.keys())):
        ReconstructionOperator = np.asarray(h5['ReconstructionOperator'])
        Mean = np.zeros(ReconstructionOperator.shape[1])
        Nedr = np.ones(ReconstructionOperator.shape[1])
    elif('Eigenvectors' in list(h5.keys())):
        ReconstructionOperator = np.asarray(h5['Eigenvectors'])
        Mean = np.asarray(h5['Mean'])
        Nedr = np.asarray(h5['Nedr'])
    else:
        print("Reconstruction operator data unknown.")
        sys.exit(1)
    return ReconstructionOperator,Mean,Nedr
def applyPc(f,subset,scores,q=[0.5]):
    R,meanz,Nedr = readMatrix(f)
    s = np.asarray(scores)
    # just use max q since every FOV has it's own scale which is the same
    qmax = max(q)
    rads = np.zeros([R[:,subset].shape[1],s.shape[1],s.shape[2]])
    mterm =  Nedr[subset]*meanz[subset]
    factor = Nedr[subset]
    RR = R[:,subset]
    l,m,n = s.shape
    k = R[:,subset].shape[1]
    rads1 = np.zeros([R[:,subset].shape[1],m*n])
    ss = s.reshape(l,m*n)
    for ii in list(range(m*n)):
        r = qmax * np.matmul( ss[:,ii], RR )
        rads1[:,ii] = factor*r + mterm #+ meanNedr[subset]*1e-3
    rads = rads1.reshape(k,m,n)
    return rads

def makeScaledInt(rads,scale_factor):
    r = np.round(rads/scale_factor,0).astype('int')
    return r

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
    while (True):
    #for iziz in range(0,1):
        print ('Decoding message number {}'.format(iii))
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
            unusedData[k].append(codes_get(ibufr,k))
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

def matchDims(modifiedDataArray,nscan=120):
    for k in list(modifiedDataArray.keys()):
        tmp = modifiedDataArray[k]
        modifiedDataArray[k] = []
        for t in tmp:
            if len(t)==1:
                ttmp = t*np.ones(nscan)
                modifiedDataArray[k].append(ttmp.tolist())
            else:
                modifiedDataArray[k].append(t)
    return modifiedDataArray     


def bufr_encode(imagerData,\
                imagerDataArray,\
                usedButDroppedData,\
                usedButDroppedDataArray,\
                modifiedData,\
                modifiedDataArray,\
                passedData,\
                passedDataArray,\
                unusedData,\
                unusedArray,\
                scaledRadianceSubset,\
                allChansInSubset,\
                outfn,\
                scale_limits,\
                scale_values,\
                mapBuf):

    nchunk = 30
    nfov = 4
    #scaled radiances dims chan,scan,fov# (1-120)
    nscan = scaledRadianceSubset.shape[1]
    for iscn in list(range(nscan)):
        fov_start = 0
        for ifov in list(range(nfov)):
            ibufr = codes_bufr_new_from_samples('BUFR3')
            codes_set_array(ibufr, 'inputExtendedDelayedDescriptorReplicationFactor', (616,))
            codes_set(ibufr, 'edition', 3)
            codes_set(ibufr, 'masterTableNumber', 0)
            codes_set(ibufr, 'bufrHeaderSubCentre', 0)
            codes_set(ibufr, 'bufrHeaderCentre', 160)
            codes_set(ibufr, 'updateSequenceNumber', 0)
            codes_set(ibufr, 'dataCategory', 21)
            codes_set(ibufr, 'dataSubCategory', 241)
            codes_set(ibufr, 'masterTablesVersionNumber', 12)
            codes_set(ibufr, 'localTablesVersionNumber', 0)

            codes_set(ibufr, 'numberOfSubsets', nchunk)
            codes_set(ibufr, 'observedData', 1)
            codes_set(ibufr, 'compressedData', 1)

            # Create the structure of the data section
            codes_set(ibufr, 'unexpandedDescriptors', 361207)
            for k in list(passedData.keys()):
                tmp = int(passedData[k][iscn])
                codes_set(ibufr, mapBuf['passed'][k], tmp)
            for k in list(passedDataArray.keys()):
                tmp = passedDataArray[k][iscn]
                codes_set(ibufr, mapBuf['passedArray'][k], tmp)
            for k in list(modifiedData.keys()):
                codes_set(ibufr,mapBuf['modified'][k],modifiedData[k][iscn])
            for k in list(modifiedDataArray.keys()):
                tmp = np.asarray(modifiedDataArray[k][iscn][fov_start:fov_start+nchunk])
                codes_set_array(ibufr, mapBuf['modifiedArray'][k], tmp)
            for iss,c in enumerate(allChansInSubset):
                codes_set(ibufr, '#{}#channelNumber'.format(iss+1), int(c))
                vec1 = np.asarray(scaledRadianceSubset[iss,iscn,fov_start:fov_start+nchunk]).astype('int32')-5000
                vec2 = np.asarray(scaledRadianceSubset[iss,iscn,fov_start:fov_start+nchunk]).astype('int32')-5000
                idx, = np.where(vec1.astype('int32')<-5000) 
                #print(vec1.min(),vec1.max())
                if(len(idx)>0):
                    idx, = np.where(vec1<-5000)
                    #instead of doing what AAPP does, just set it to a really high value.
                    vec2[idx]=60535
                codes_set_array(ibufr, '#{}#scaledIasiRadiance'.format(iss+1), vec2)
            for ib,b in enumerate(list(scale_limits.keys())):
                if(scale_limits[b][0]>0 and scale_limits[b][1]>0):
                    codes_set(ibufr, '#{}#startChannel'.format(ib+1), int(scale_limits[b][0]))
                    codes_set(ibufr, '#{}#endChannel'.format(ib+1), int(scale_limits[b][1]))
                    codes_set(ibufr, '#{}#channelScaleFactor'.format(ib+1), int(-1.0*np.log10(float(scale_values[b]))))
 

            for k in list(imagerData.keys()):
                # NCEP's message goes through 616 channel numbers before it gets to AVHRR
                # eumetsat's bufr doesn't go through channels, because it's all PCs
                kk = mapBuf['imager'][k]
                codes_set(ibufr,kk,imagerData[k][iscn])
            for k in list(imagerDataArray.keys()):
                kk = mapBuf['imagerArray'][k]
                codes_set_array(ibufr,kk,imagerDataArray[k][iscn][fov_start:fov_start+nchunk])
            fov_start+=nchunk
            # Encode the keys back in the data section
            codes_set(ibufr, 'pack', 1)
            if(iscn == 0 and ifov == 0):
                outfile = open(outfn, 'wb')
                print ("Created output BUFR file {}".format(outfile))
            else:
                outfile = open(outfn, 'ab')
            print ("Writing Scan {} FOR {} in {}".format(iscn+1,ifov+1,outfile))
            codes_write(ibufr, outfile)
            codes_release(ibufr)

def main(matrix, subset, band_limits, pc_limits, scale_limits, scale_values, ioIn, ioBufOut, ioNcOut, infn, outfn, defs):
   
    codes_set_definitions_path(defs)

    print('Input File: ',infn)
    print('Output File', outfn)
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
    
    #setup band channel numbers, and band idx starting at 0.
    bandChans = {}
    bandIdx = {}
    for k in list(subset.keys()):
        bandChans[k] = np.arange(int(band_limits[k][0]),int(band_limits[k][1])+1,1)
        tmp = []
        for i in subset[k]:
            tmp.append(np.where(bandChans[k]==i)[0][0])
        bandIdx[k] = tmp


    # Initialize an array to hold PCs in each band, add 1 to upper limit for python behavior of last value in list  
    pcBand = {}
    for k in list(matrix.keys()):
        pcBand[k] = []
        #add 1 to limits to follow python convention for range.
        pc_limits[k][1]+=1

    # for each band, grab Qs and PCs
    score_q = {}
    bandNames = list(pcBand.keys())
    bandNames.sort()
    for k in bandNames:
        score_q[k] = usedButDroppedDataArray['#{}#scoreQuantizationFactor'.format(k.replace('band',''))]
        for i in range(pc_limits[k][0],pc_limits[k][1]):
            pcBand[k].append(usedButDroppedDataArray['#{}#nonNormalizedPrincipalComponentScore'.format(i)])

    # for each band apply PCs and make a big matrix of reconstructed radiances
    subsetRad = []
    for b in bandNames:
        tmp = applyPc(matrix[b], bandIdx[b], pcBand[b], q=score_q[b])
        if(len(subsetRad)==0):
            subsetRad = tmp
        else:
            subsetRad = np.append(subsetRad,tmp,axis=0)
    # get channel numbers associated with subset 
    allChansInSubset = []
    ks = list(subset.keys())
    ks.sort()
    for k in ks:
        allChansInSubset.extend(subset[k])
    allChansInSubset = np.asarray(allChansInSubset)

    #Apply scale factors
    scaleArr = np.zeros(subsetRad.shape)
    for l in list(scale_limits.keys()):
        idx, = np.where( (allChansInSubset>=scale_limits[l][0]) &(allChansInSubset<=scale_limits[l][1]) )
        scaleArr[idx,:,:] = scale_values[l]
    scaledRadianceSubset = np.round(subsetRad/scaleArr,0).astype(int)
    # sometimes people like fov number to start with 0, the GSI does not (JEDI doesn't either), so add 1.
    if(modifiedDataArray['fieldOfViewNumber'][0][0] == 0):
        tmp = np.asarray(modifiedDataArray['fieldOfViewNumber']) +1
        modifiedDataArray['fieldOfViewNumber'] = tmp

    hf = h5py.File('quick_dump.h5', 'w')
    hf.create_dataset('scaledRadiance', data=scaledRadianceSubset)
    hf.create_dataset('longitude', data=np.asarray(modifiedDataArray['longitude']))
    hf.create_dataset('latitude', data=np.asarray(modifiedDataArray['latitude']))
    hf.create_dataset('fieldOfViewNumber', data=np.asarray(modifiedDataArray['fieldOfViewNumber'])+1)
    hf.create_dataset('satelliteZenithAngle', data=np.asarray(modifiedDataArray['satelliteZenithAngle']))
    hf.close()

    modifiedDataArray = matchDims(modifiedDataArray)
    imagerDataArray = matchDims(imagerDataArray)
  
    mapNc = createMap(ioIn, ioNcOut)
    mapBuf = createMap(ioIn, ioBufOut)   

    bufr_encode(imagerData,\
                imagerDataArray,\
                usedButDroppedData,\
                usedButDroppedDataArray,\
                modifiedData,\
                modifiedDataArray,\
                passedData,\
                passedDataArray,\
                unusedData,\
                unusedArray,\
                scaledRadianceSubset,\
                allChansInSubset,\
                outfn,\
                scale_limits,\
                scale_values,\
                mapBuf)

if __name__ == "__main__":
    parser = argparse.ArgumentParser( description = 'Run PCC reconstructed radiances.')
    parser.add_argument('--input', help = 'experiment', required = True, dest = 'input')
    parser.add_argument('--output',help = 'control', required = True, dest='output')
    parser.add_argument('--definitions', help="control name.", dest='definitions', default='definitions' )
    aa = parser.parse_args()

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


    main( matrix, subset, band_limits, pc_limits, scale_limits, scale_values, ioIn, ioBufOut, ioNcOut, aa.input, aa.output, aa.definitions)
