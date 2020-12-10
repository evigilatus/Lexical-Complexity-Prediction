import xml.etree.ElementTree as ET
lang = 'English'
root = ET.fromstring(open('bible.xml').read())
with open('bible.txt', 'w', encoding='utf-8') as out:
    for n in root.iter('seg'):
        out.write(n.text.strip() + '\n')