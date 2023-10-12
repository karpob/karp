from datetime import timedelta, date, datetime

dtg = datetime(2022,4,24,0)
maxDtg = datetime(2022,4,24,6)
while dtg<maxDtg:
    date= dtg.strftime("%y%m%d")
    hr = dtg.strftime("%H")
    cmd = "./karrpp.py --input /discover/nobackup/projects/gmao/obsdev/bkarpowi/iasipc_dumps/gdas.{}.t{}z.mtiasi.tm00.bufr_d --output /discover/nobackup/projects/gmao/obsdev/bkarpowi/iasipc_dumps/rr/gdas.{}.t{}z.mtiasi.tm00.bufr_d".format(date,hr,date,hr)
    print(cmd)
    dtg+=timedelta(hours=6)

