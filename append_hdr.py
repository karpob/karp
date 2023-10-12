#!/usr/bin/env python3
import argparse, os, sys, glob,shutil
from datetime import timedelta, date, datetime

def getFilesInWindow(files,swin,ewin):
    filesOut = []
    for f in files:
        fn = os.path.basename(f)
        fns = fn.split('-')
        fstart = datetime.strptime(fns[5].split('.')[0],'%Y%m%d%H%M%S')
        fend = datetime.strptime(fns[6],'%Y%m%d%H%M%S')
        if ( (fstart >= swin or fend >= swin) and (fstart <= ewin)):
            filesOut.append(f)
    return filesOut

def appendHdr(fi, hdr, outpath, dtg, prefix, suffix):
     
    #gdas1.%y2%m2%d2.t%h2z.mtiasi.tm00.bufr_d 
    fext=open(os.path.join(outpath,prefix+'.'+dtg.strftime("%y%m%d.t%Hz")+'.'+suffix),"wb")
    fo = open(hdr,"rb")
    shutil.copyfileobj(fo, fext)
    fo.close()
    fo = open(fi,"rb")
    shutil.copyfileobj(fo,fext)
    fo.close()

    fext.close()        
def main(inpath, outpath, start, end, prefix, suffix, hdr):
    startYear, startMonth, startDay, startHour = int(start[0:4]), int(start[4:6]), int(start[6:8]), int(start[8:10])
    endYear, endMonth, endDay, endHour = int(end[0:4]), int(end[4:6]), int(end[6:8]), int(end[8:10])
    startDtg = datetime(startYear, startMonth, startDay, startHour)
    endDtg = datetime(endYear, endMonth, endDay, endHour)
    currentDtg = startDtg
    while(currentDtg<endDtg):
        fileIn = os.path.join(inpath,prefix+'.'+currentDtg.strftime("%y%m%d.t%Hz")+'.'+suffix)
        appendHdr( fileIn, hdr, outpath, currentDtg, prefix, suffix)
        currentDtg += timedelta(hours=6)        
if __name__ == "__main__":
    #gdas1.%y2%m2%d2.t%h2z.mtiasi.tm00.bufr_d 
    parser = argparse.ArgumentParser( description = 'cat together bufr files by assimilation windows.')
    parser.add_argument('--path', help = 'path to input bufr', required = True, dest = 'path')
    parser.add_argument('--outpath', help = 'path to output bufr', required = True, dest = 'outpath')
    parser.add_argument('--start',help = 'start dtg', required = True, dest='start')
    parser.add_argument('--end', help="end dtg",required=True,dest='end')
    parser.add_argument('--prefix', help="what to stick on front of outputfile",required=False,dest='prefix',default='gdas')
    parser.add_argument('--suffix', help="what to stick on the end of outputfile",required=False,dest='suffix',default='mtiasi.tm00.bufr_d')
    parser.add_argument('--header', help="internal bufr header for gsi",required=False,dest='hdr',default='iasi.hdr')
    a = parser.parse_args()
    main(a.path, a.outpath, a.start, a.end, a.prefix, a.suffix, a.hdr)

