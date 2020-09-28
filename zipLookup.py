import sys
import pyodbc as oc
import requests as req
import xml.etree.ElementTree as ET


def zipLookup(tgt):
	obj = tgt[0]
	addr = tgt[1].strip()

	query = '<ZipCodeLookupRequest USERID="XXXXXXXXX"> <Address ID="1"><Address1></Address1><Address2>'+addr+'</Address2><City>New York</City><State>NY</State><Zip5></Zip5><Zip4></Zip4></Address></ZipCodeLookupRequest>'
	ret = req.get('https://secure.shippingapis.com/ShippingAPI.dll?API= ZipCodeLookup&XML='+query)

	tree = ET.fromstring(ret.text)
	all_name_elements = tree.findall('*/Zip5')
	print(int(all_name_elements[0].text) if len(all_name_elements)>0 else 0)
	return [obj, addr, int(all_name_elements[0].text) if len(all_name_elements)>0 else 0]


db_file_location = r'D:\database\Database1.accdb'
connection = oc.connect(rf'Driver={{Microsoft Access Driver (*.mdb, *.accdb)}};DBQ={db_file_location};')

cursor = connection.cursor()
cursor.execute("select ID,ADDRESS from salescleaned where [ZIP CODE]=0")
results = cursor.fetchall()

zipCode = list(map(zipLookup, results))

for i in zipCode:
	if i[2] > 0:
		cursor.execute('update salescleaned set [ZIP CODE]='+str(i[2])+' where ID='+str(i[0]))

connection.commit()