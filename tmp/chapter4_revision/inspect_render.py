from pathlib import Path
import pypdfium2 as pdf
from pypdf import PdfReader
from PIL import Image, ImageDraw, ImageOps
root=Path(__file__).resolve().parent
out=root/'final_pages';out.mkdir(exist_ok=True)
source=root/'chapter_four_qa.pdf'
d=pdf.PdfDocument(source)
reader=PdfReader(source)
rows=[]
for i,page in enumerate(d):
    im=page.render(scale=1.5).to_pil()
    im.save(out/f'page-{i+1:02}.png')
    txt=reader.pages[i].extract_text()
    rows.append(f'{i+1}: {len(txt.split())} words | '+txt[:75].replace('\n',' ')+' ... '+txt[-65:].replace('\n',' '))
for start in range(0,len(d),8):
    contact=Image.new('RGB',(1200,1600),'#dddddd'); draw=ImageDraw.Draw(contact)
    for j,i in enumerate(range(start,min(start+8,len(d)))):
        thumb=ImageOps.contain(Image.open(out/f'page-{i+1:02}.png'),(390,500))
        x=(j%3)*400;y=(j//3)*533
        contact.paste(thumb,(x,y+23));draw.text((x+8,y+5),f'Page {i+1}',fill='black')
    contact.save(out/f'contact-{start+1:02}.png')
root.joinpath('page_audit.txt').write_text('\n'.join(rows),encoding='utf-8')
print('\n'.join(rows))
