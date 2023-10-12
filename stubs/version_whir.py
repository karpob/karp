from datetime import timedelta, date, datetime

dtg = datetime(2022,4,24,00)
maxDtg = datetime(2022,5,7,00)
while dtg<maxDtg:
    date= dtg.strftime("%y%m%d")
    hr = dtg.strftime("%H")
    cmd = "./version_dump.py /discover/nobackup/projects/gmao/obsdev/bkarpowi/iasipc_dumps/gdas.{}.t{}z.mtiasi.tm00.bufr_d".format(date,hr)
    print(cmd)
    dtg+=timedelta(hours=6)

