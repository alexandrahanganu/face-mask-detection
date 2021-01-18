import xml.etree.ElementTree as ET

maski = 1
masklessi = 1
for count in range(853):
    tree = ET.parse("D:\\projectAI\\annotations\\maksssksksss" + str(count) + ".xml")
    all_name_elements = tree.findall('object')
    root = tree.getroot()
    ok_mask = 0
    for child in root:
        # print(child)
        for ch in child:
            # print(ch)
            for atr in ch.iter('name'):
                # print(atr)
                if atr.text == 'with_mask':
                    ok_mask = 1
                    print("with mask")
                elif atr.text == 'without_mask':
                    print("without mask")
                    ok_mask = 0
            if ch.tag == 'bndbox':
                for atr in ch.iter('xmin'):
                    xmin = atr.text
                for atr in ch.iter('xmax'):
                    xmax = atr.text
                for atr in ch.iter('ymin'):
                    ymin = atr.text
                for atr in ch.iter('ymax'):
                    ymax = atr.text
                from PIL import Image

                img = Image.open("D:\\projectAI\\images\\maksssksksss" + str(count) + ".png")
                img2 = img.crop((int(xmin), int(ymin), int(xmax), int(ymax)))
                if (ok_mask == 1):
                    img2.save("D:\\projectAI\\with_mask\\with_mask" + str(maski) + ".png")
                    maski += 1
                else:
                    img2.save("D:\\projectAI\\without_mask\\without_mask" + str(masklessi) + ".png")
                    masklessi += 1
