import xlrd
import sys
import logging
import json


workbook = xlwt.Workbook(encoding= 'ascii')

    
worksheet = workbook.add_sheet("My new Sheet")

    # 往表格写入内容
worksheet.write(0,0, "内容1")