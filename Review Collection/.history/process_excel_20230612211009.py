import xlrd
import sys
import logging
import json
import xlwt;


workbook = xlwt.Workbook(encoding= 'ascii')

    
worksheet = workbook.add_sheet("Sheet1")

   
worksheet.write(0,0, "内容1")